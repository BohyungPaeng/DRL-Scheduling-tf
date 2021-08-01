from model.util_nn import *
import tensorflow as tf
import numpy as np

class BaseNetwork(object):
    def __init__(self, sess, input_dim, action_dim, update_option, name, optimizer, tau,
                 initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN")):
        """
        Abstarct class for creating networks
        :param input_dim:
        :param action_dim:
        :param stddev:
        """

        # if use soft update, tau should not be None
        self.tau = tau

        self.update_option = update_option
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.initializer = initializer
        self.sess = sess

        # build network
        self.build(name)
        self.network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                              "target" not in v.name]

        # build target
        self.build_target("target_%s" % name)
        self.target_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                             "target" in v.name]

        self.gradients = None

        # optimizer
        self.optimizer = optimizer

    def create_update_op(self):
        import time
        st = time.time()
        print('start time', st)
        if self.update_option == "soft_update":
            update_op = [tf.assign(target_param, (1 - self.tau) * target_param + self.tau * network_param)
                         for target_param, network_param in zip(self.target_param, self.network_param)]
        else:
            update_op = [tf.assign(target_param, network_param)
                         for target_param, network_param in zip(self.target_param, self.network_param)]
        print('elapsed time', time.time()-st)
        return update_op

    def create_train_op(self):
        return self.optimizer.apply_gradients([(g, v) for g, v in zip(self.gradients, self.network_param)])

    def build(self, name):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def build_target(self, name):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def compute_gradient(self):
        """
        Abstract method, compute gradient in order to be used by self.optimizer
        """
        raise NotImplementedError("Not implemented")
    def init_summary(self, weight_hist, summary_dir, name):
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        self.summary_writer_hist = None
        self.hist_freq = weight_hist
        summary_list = []
        if summary_dir:
            if self.is_train:
                summary_dir = os.path.join(summary_dir, "summaries_{}".format(name))
            else:
                summary_dir = os.path.join(summary_dir, "summaries_test_{}".format(name))

            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir+'/scalar')
            if self.is_train and self.hist_freq: self.summary_writer_hist = tf.summary.FileWriter(summary_dir+'/hist')
        # Summaries for Tensorboard
        if weight_hist:
            var_list = tf.trainable_variables()
            # self.grads = optimizer.compute_gradients(self.loss, var_list=var_list)
            # for grad, var in self.grads:
            #     if var.op.name.split("/")[0] != name:
            #         continue
            #     if grad is not None:
            #         if 'weights' in var.op.name:
            #             summary_list.append(tf.summary.histogram(var.op.name + "/gradients", grad))

            for var in var_list:
                if var.op.name.split("/")[0] != name:
                    continue
                if 'weights' in var.op.name or 'kernel' in var.op.name or 'bias' in var.op.name:
                    weight_name = var.op.name.split("/")[1] + "/" + var.op.name.split("/")[2]
                    with tf.variable_scope(name, reuse=True):
                        weight = tf.get_variable(weight_name)
                    summary_list.append(tf.summary.histogram(var.op.name, weight))
        summary_list.extend([
            # tf.summary.histogram('fc1_weight ' % name , (weights)),
            # tf.summary.histogram('fc1_bias' , b),
            tf.summary.scalar('logits/max_predicted_q_value', tf.reduce_max(self.predicted_Q)),
            # tf.summary.scalar('loss/batch_loss_%s' % name, self.loss),
            tf.summary.scalar('loss/batch_loss', self.loss),
            ])
        if self.is_duel:
            summary_list.extend([
                tf.summary.scalar('logits/value_estimated', tf.reduce_max(self.value[0])),
                tf.summary.scalar('logits/max_predicted_a_value', tf.reduce_max(self.adv[0]))
            ])

        print('Setting {} length of default summary'.format(len(summary_list)))
        self.summary = tf.summary.merge(summary_list)


class PDQN(BaseNetwork):
    def __init__(self, sess, input_dim, action_dim, auxin_dim, tau, optimizer, name, global_step,
                 is_duel=False, is_train=True, layers=[64,32], summary_dir=None, weight_hist=False):
        """
        Initialize critic network. The critic network maintains a copy of itself and target updating ops
        Args
            input_dim: dimension of input space, if is length one, we assume it is low dimension.
            action_dim: dimension of action space.
            stddev: standard deviation for initializing network params.
        """
        self.is_train = is_train
        self.is_duel = is_duel
        self.auxin_dim = auxin_dim
        self.layers=layers
        self.name = name
        # else:
        #     self.name = name+'_test'

        super(PDQN, self).__init__(sess, input_dim, action_dim, update_option="soft_update",
                                            name=self.name, optimizer=optimizer, tau=tau)
        self.update_op = self.create_update_op()

        self.is_training = tf.placeholder(dtype=tf.bool, name="bn_is_train")

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.target_Q = tf.placeholder(tf.float32, shape=[None, 1], name="target_q")

        self.action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="selected_action")
        # self.tp = tf.transpose(self.action)

        self.predicted_Q = tf.reduce_sum(tf.multiply(self.action, self.net), axis=1, keep_dims=True) # batch_size

        self.global_step = global_step
        self.loss = self._loss()
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.init_summary(weight_hist, summary_dir, name)

    def _loss(self):
        # bellman loss
        diff = self.target_Q - self.predicted_Q
        delta = 0.5
        loss = tf.where(tf.abs(diff) < delta, 0.5 * tf.square(diff), delta * tf.abs(diff) - 0.5 * (delta ** 2))
        # For Prioritized Replay buffer(Importance Sampling)
        self.weight = tf.placeholder(tf.float32, shape=[None, 1], name="weight_is")
        loss= tf.reduce_mean(tf.multiply(self.weight, loss))
        self.loss_comp=[loss]
        return loss

    def base_encoder_cells(self, x, name='', reuse=False):
        if len(self.input_dim)==2:
            net = None
            for i in range(self.input_dim[0]):
                _input = tf.squeeze(tf.slice(x, [0,i,0],[-1,1,-1]),axis=1)
                # print(i, self.sess.run([_input, tf.shape(_input)], feed_dict={self.x:np.zeros((2,10,13))}))
                net_temp = self._base_encoder_cells(_input, name=name, reuse=tf.AUTO_REUSE)
                if net is None: net = net_temp
                else: net = tf.concat([net,net_temp],1)
        else: net = self._base_encoder_cells(x, name=name, reuse=reuse)
        return net
    def _base_encoder_cells(self, x, name='', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            net = tf.identity(x)
            # Default initializer is xavier_initializer
            for l in range(len(self.layers)):
                h_dim = self.layers[l]
                # net = tf.contrib.layers.fully_connected(net, num_outputs=h_dim, activation_fn=tf.nn.relu, scope="fc{}".format(l)
                #                                         # ,normalizer_fn=tf.contrib.layers.layer_norm
                #                                         )
                net = dense_layer(net, output_dim=h_dim, activation_fn=tf.nn.relu, scope="fc{}".format(l))

        return net

    def value_layer(self, net, name='', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # net = dense_layer(net, output_dim=8, activation_fn=tf.sigmoid, scope="fc_val")
            # net = dense_layer(net, output_dim=1, activation_fn=None, scope="value")
            net = tf.contrib.layers.fully_connected(net, activation_fn=None, num_outputs=self.action_dim,
                                                    # weights_regularizer=regularizer,
                                                    weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                                    biases_initializer=tf.constant_initializer(.0, dtype=tf.float32),
                                                    scope="q", reuse=reuse)
        return net
    def fusion_layer(self, net, auxin, name='', reuse=False):
        net = tf.concat([net, auxin], 1)
        # with tf.variable_scope(name, reuse=reuse):
        #     net = dense_layer(net, output_dim=100, activation_fn=tf.nn.relu, scope="fc_fus")
            # net = dense_layer(net, output_dim=16, activation_fn=tf.nn.relu, scope="fc_fus2")
        return net

    def build(self, name):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_dim], name="%s_input" % name)
        net = self.base_encoder_cells(self.x, name)
        if self.auxin_dim > 0:
            self.auxin = tf.placeholder(dtype=tf.float32, shape=[None, self.auxin_dim], name="%s_auxin" % name)
            # net = tf.concat([net, self.auxin], 1)
            # net = dense_layer(net, output_dim=64, activation_fn=tf.nn.relu, scope="%s_fc_fus" % name)
            net = self.fusion_layer(net, self.auxin, name)

        # last layer
        if self.is_duel:
            with tf.variable_scope(name):
                # net1 = tf.identity(net); net2 = tf.identity(net)
                net1 = dense_layer(net, output_dim=8, activation_fn=tf.sigmoid, scope="fc_val")
                self.value = dense_layer(net1, output_dim=1, activation_fn=None, scope="value")
                net2 = dense_layer(net, output_dim=8, activation_fn=tf.sigmoid, scope="fc_adv")
                # value = dense_layer(net1, 1,
                #                     weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                #                          scope="value", use_bias=True)
                # adv = dense_layer(net2, self.action_dim,
                #                         weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                #                              scope="advantage", use_bias=True)
                # value = self.value_layer(net, name)
                self.adv = dense_layer(net2, self.action_dim, activation_fn = None, scope="advantage")
            self.net = self.value + (self.adv - tf.reduce_mean(self.adv, reduction_indices=[1, ], keep_dims=True))

        else:
            self.net = self.value_layer(net, name=name)
    def build_target(self, name):
        self.target_x = tf.placeholder(dtype=tf.float32, shape=[None, *self.input_dim], name="%s_input" % name)
        net = self.base_encoder_cells(self.target_x, name)
        if self.auxin_dim > 0:
            self.target_auxin = tf.placeholder(dtype=tf.float32, shape=[None, self.auxin_dim], name="%s_auxin" % name)
            # net = tf.concat([net, self.target_auxin], 1)
            # net = dense_layer(net, output_dim=16, activation_fn=tf.nn.relu, scope="%s_fc_fus" % name)
            net = self.fusion_layer(net, self.target_auxin, name)
        # last layer
        if self.is_duel:

            with tf.variable_scope(name):
                # net1 = tf.identity(net); net2 = tf.identity(net)
                net1 = dense_layer(net, output_dim=8, activation_fn=tf.sigmoid, scope="fc_val")
                value = dense_layer(net1, output_dim=1, activation_fn=None, scope="value")
                net2 = dense_layer(net, output_dim=8, activation_fn=tf.sigmoid, scope="fc_adv")
                # value = dense_layer(net1, 1,
                #                     weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                #                          scope="value", use_bias=True)
                # adv = dense_layer(net2, self.action_dim,
                #                         weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                #                              scope="advantage", use_bias=True)
                # value = self.value_layer(net, name)
                adv = dense_layer(net2, self.action_dim, activation_fn=None, scope="advantage")
            self.target_net = value + (adv - tf.reduce_mean(adv, reduction_indices=[1, ], keep_dims=True))

        else:
            self.target_net = self.value_layer(net, name=name)


    def compute_gradient(self):
        grad = tf.gradients(self.loss, self.network_param, name="critic_gradients")
        # action_grad = tf.gradients(self.net, self.action, name="action_gradient")
        return grad

    # def action_gradients(self, inputs, actions):
    #     return self.sess.run(self.action_grads, {self.x: inputs, self.action: actions})
    def critic_predict(self, state, auxin, feasibility=None):
        return self._critic_predict(state, auxin)

    def critic_target_predict(self, state, auxin, feasibility=None):
        return self._critic_target_predict(state, auxin)

    def _critic_predict(self, state, auxin, summary=False):
        """
        If summary is True, we also get the q value. This is used for logging.
        """
        # if summary:
        #     return self.sess.run(self.q_summary, feed_dict={self.critic.action: action, self.critic.x: state})
        # else:
        feed_dict = {self.x: state}
        if self.auxin_dim > 0: feed_dict.update({self.auxin: auxin})

        if self.is_duel:
            net, adv, value = self.sess.run([self.net, self.adv, self.value],
                                            feed_dict=feed_dict)
            # print("adv", net[1], "value", net[2], "Q", net[0])
            # net = self.sess.run(self.net, feed_dict={self.x: state})
            # print(self.sess.run(self.advantage))
            # print(net)
            return net, adv, value
        else: return self.sess.run(self.net, feed_dict=feed_dict)

    def _critic_target_predict(self, state, auxin):
        feed_dict = {self.target_x: state}
        if self.auxin_dim > 0: feed_dict.update({self.target_auxin: auxin})
        # if self.use_lstm: feed_dict.update({self.target_c_in: self.lstm_state_out_target[0], self.target_h_in:self.lstm_state_out_target[1]})
        return self.sess.run(self.target_net, feed_dict=feed_dict)

    def critic_train(self, weight, inputs, auxin, action, target_q_value, train_step):
        feed_dict = {self.weight: weight, self.target_Q: target_q_value, self.action: action}
        if self.auxin_dim > 0:
            feed_dict.update({self.auxin:auxin})
        feed_dict.update({self.x:inputs})

        summary, loss, loss_comp, train_op, predicted_Q, target_Q, action = \
            self.sess.run(
                [self.summary, self.loss, self.loss_comp, self.train_op, self.predicted_Q, self.target_Q, self.action],
                feed_dict=feed_dict)

        # print(loss, loss_b, loss_r)
        # print(self.sess.run())
        if self.summary_writer_hist:
            if train_step % self.hist_freq == 0:
                self.summary_writer_hist.add_summary(summary, global_step=train_step)
                print('write hist summary', len(summary))
        return loss, train_op, predicted_Q, target_Q, action
    # def critic_train(self, estimated_q_value, predicted_q_value):
    #     return self.sess.run([self.loss, self.train],
    #                          feed_dict={self.y: predicted_q_value, self.q: estimated_q_value})
    def update_critic(self):
        self.sess.run(self.update_op)
        # print(self.sess.run(self.target_param))
    def getSummary(self):
        return self.summary_writer

    def get_action(self):
        raise NotImplementedError("Not implemented")