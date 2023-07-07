from agent.replay_buffer import *
from utils.visualize.logger import instance_log
from config import *
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
import random
from model import *
import sys

class Trainer(object):

    def __init__(self, sess, optimizer, idx=0, exp_idx=0, global_step=None, use_buff=True, use_hist=False):
        self.Epsilon = args.eps
        self.eps = self.Epsilon
        self.Epsilon_epoch = args.max_episode * args.eps_ratio
        self.TAU = 1
        self.Episode = 1
        self.TimeStep = 0
        self.RememberFlag = False
        self.cutday = -1

        """For experiences"""
        self.dat = ReplayBuffer(args.max_buffersize)
        self.temp_buffer = None

        """For checking training performance"""
        self.reward_history = []
        self.reward_total = 0
        self.trigger_cnt = 0
        self.decision_cnt = 0
        self.cumQ = 0
        self.cumV_real = 0
        self.setupNum = 0
        self.RewardMemo = ""

        """Network objects"""
        self.summary_writer = None
        self.summary_dir = args.summary_dir
        self.nn = None
        is_conv = True if type(args.state_dim) == list and len(args.state_dim)>2 else False
        print(args.auxin_dim)
        self.nn = PDQN(sess, action_dim=args.action_dim, input_dim=args.state_dim, auxin_dim=args.auxin_dim,
                       optimizer=optimizer, tau=self.TAU, name="dqn_pms_{}".format(exp_idx), layers=args.hid_dims,
                       is_train=args.is_train, is_duel = args.is_duel,
                       global_step=global_step, summary_dir=self.summary_dir, weight_hist=use_hist)
        # self.nn = PolicyModel(input_dims = args.state_dim, input_dim_str='1', # original paper applies 2x1 conv on 2xs state
        #                          hidden_dims = args.hid_dims, action_dim=args.action_dim,
        #                          GAMMA=args.GAMMA, ALPHA=args.lr,
        #                          )
        self.use_buff = use_buff
        if args.is_train:
            self.record = instance_log(args.gantt_dir, 'instances_{}'.format(args.timestamp))
        else:
            self.record = instance_log(args.gantt_dir, 'test_instances_{}'.format(args.timestamp))

    def getDecisionNum(self):
        return self.decision_cnt

    def SetEpisode(self, episode, num_decision=None):
        self.reward_total = 0
        if num_decision == None:
            self.is_terminated=False
        else:
            if self.trigger_cnt != 0: self.num_decision = self.trigger_cnt
            else: self.num_decision = num_decision
        self.trigger_cnt = 0
        self.decision_cnt = 0
        self.setupNum = 0
        self.cumQ = 0
        self.cumV_real = 0
        self.loss_history = np.zeros(300, dtype=float)
        self.Episode = episode
        self.eps = self.Epsilon
        self.record.clearInfo()
        self.targetupFlag=True

        # self.memory()

    def memory(self):
        import os, psutil
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory use:', memoryUse)

    def getSummary(self):
        if isinstance(self.nn, PDQN):
            return self.nn.getSummary()
        else:
            return False

    def writeSummary(self):
        summ = self.getSummary()
        if summ is False:
            print("There is no summary writer")
            return
        episode_summary = tf.compat.v1.Summary()
        episode_summary.value.add(simple_value=self.reward_total, node_name='reward/cumulative_reward', tag='reward/cumulative_reward')
        episode_summary.value.add(simple_value=self.cumQ, node_name='reward/cumulative_Q', tag='reward/cumulative_Q')
        L_avg = 0
        if len(self.loss_history) > 0:
            L_avg = np.mean(self.loss_history)
        episode_summary.value.add(simple_value=L_avg, tag='loss/episodic_loss')
        # if args.use_vp:
        #     episode_summary.value.add(simple_value=self.nn.loss_tf.summary.scalar('loss/batch_loss_v', tf.subtract(self.loss_end, self.loss)
        summ.add_summary(episode_summary, self.Episode)
        summ.flush()
    def get_action_feasibility(self, observe):
        temp = observe['state']
        feasibility = observe['feasibility']
        curr_time = observe['time']
        state = np.zeros([1, len(temp)])
        state[0] = temp
        state = np.expand_dims(state, axis=2) # to make 3-d vector of CQ-PG style
        prob = self.nn.get_probabilities(state)
        # self.sess.run(self.act_probs, feed_dict={self.inputs: state})[0]
        max_logit = -sys.maxsize
        action = 0
        for a in feasibility:
            if prob[a]>=max_logit:
                max_logit = prob[a]
                action = a
        observe['logits'] = prob
        observe['max_q'] = max_logit
        return observe, action, curr_time

    def get_action(self, observe):

        '''
        DRL agent recieves observations from a simulator or environments.
        :return: next observation, action vector, current scheduling time
        '''

        self.TimeStep += 1
        self.trigger_cnt += 1
        state = observe['state']
        feasible_action_index = observe['feasibility']
        curr_time = observe['time']
        if len(feasible_action_index) == 0:
            return observe, -1, curr_time

        if self.check_exploration():
            return observe, random.Random().choice(feasible_action_index), curr_time  # randrange(args.action_dim)

        if args.auxin_dim != 0:
            auxin = observe['auxin']
        else:
            auxin = []
        self.decision_cnt += 1
        if isinstance(self.nn, PolicyModel):
            return self.get_action_feasibility(observe)
        """For model of DQN-type, implement Greedy policy"""
        deterministric = True
        logits =  list(self.nn.critic_predict([state],[auxin], feasibility=feasible_action_index)[0]) # numpy array predictions to list
        # print(logits)
        if args.is_duel:
            q, adv, val = self.nn.critic_predict([state],[auxin])
            logits = list(q[0])
            if self.TimeStep % 10000 == 0: print('q', logits, 'adv', adv[0], 'val', val[0])
        if deterministric:
            if len(feasible_action_index) == args.action_dim:
                action = logits.index(max(logits))
            else:
                max_logit = -1000000
                max_index = -1
                for i in feasible_action_index:
                    now_logit = logits[i]
                    if max_logit < now_logit:
                        max_logit = now_logit
                        max_index = i
                action = max_index
        else:  # probabilistic action
            if sum(logits) != 1:
                logits[-1] += 1 - sum(logits)
            action = -1
            while action not in feasible_action_index:
                action = np.random.choice(len(logits), 1, p=logits)[0]

        for prod_idx in range(len(logits)): self.record.appendInfo('Qvalue_%03d' % prod_idx, logits[prod_idx])
        # if args.use_nost and action % 10 == action // 10: action = 0
        now_value = logits[action]
        self.cumQ += float(now_value)
        observe['logits'] = logits
        observe['max_q'] = now_value
        
        return observe, action, curr_time

    def check_exploration(self):
        if not args.is_train:
            return False
        if args.warmup > self.TimeStep:
            return True
        # epsilon-greedy policy
        epi = self.Episode
        if self.Epsilon_epoch != 0:
            eps = self.Epsilon * max(0, self.Epsilon_epoch - epi) / self.Epsilon_epoch# decaying eps
            self.eps=eps
        if np.random.rand() < self.eps:
            return True
        return False
    def getEps(self): return self.eps

    def remember_record(self, pre_observe, action, reward, terminalFlag):
        self.reward_history.append(reward)
        self.reward_total += reward
        state = pre_observe['state']
        for state_idx in range(len(state)):
            self.record.appendInfo('state_{:04d}'.format(state_idx), state[state_idx])
        self.record.appendInfo('action', np.where(action[0]==1)[0][0])
        self.record.appendInfo('reward', reward)
        self.record.appendInfo('terminal', terminalFlag)
        self.record.saveInfo()
    def remember(self, pre_observe, action, observe, reward, terminalFlag):
        feasible_action_index_list = observe['feasibility']
        curr_time = observe['time']
        self.reward_history.append(reward)
        self.reward_total += reward
        if args.auxin_dim != 0:
            auxin = pre_observe['auxin']
            next_auxin = observe['auxin']
        else:
            auxin = []
            next_auxin = []
        state = pre_observe['state']
        next_state= observe['state']
        if 'upm' in args.oopt:
            self.train_step([np.array(state)],  np.reshape(action, (args.action_dim,)), reward,
                        np.array(feasible_action_index_list), [np.array(next_state)], terminalFlag)
            return
        for state_idx in range(len(state)):
            self.record.appendInfo('state_{:04d}'.format(state_idx), state[state_idx])
        self.record.appendInfo('action', np.where(action[0]==1)[0][0])
        self.record.appendInfo('reward', reward)
        self.record.appendInfo('terminal', terminalFlag)
        self.dat.add(np.reshape(state, (args.state_dim)),
                     auxin,
                     np.reshape(action, (args.action_dim,)),
                     reward,
                     terminalFlag,
                     np.reshape(next_state, (args.state_dim)),
                     next_auxin,
                     np.array(feasible_action_index_list),)

        if self.TimeStep > args.warmup and (self.TimeStep - args.warmup) % args.freq_on == 0 and args.is_train:
            self.train_network(terminalFlag)
        if args.sampling == 'pretrain':
            if self.TimeStep <= args.warmup:
                if self.temp_buffer is None: self.temp_buffer = DataBuffer()
                self.temp_buffer.add(np.reshape(state, (args.state_dim,)),
                 np.reshape(action, (args.action_dim,)),
                 reward, terminalFlag,
                 np.reshape(next_state, (args.state_dim,)),
                 np.array(feasible_action_index_list))
            if self.TimeStep == args.warmup and terminalFlag:
                self.pretrain()

        self.record.saveInfo()
        if args.freq_tar==0: #for tmu=0.01 in TPDQN
            if self.TimeStep % 100 == 0 and args.is_train:
                self.update_target_network()
        elif terminalFlag and self.targetupFlag:
            if self.Episode % args.freq_tar == 0 and args.is_train:
                self.update_target_network()
                self.targetupFlag= False

    def pretrain(self):
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.temp_buffer.get()
        temp = 0
        found_flag=False
        real_cq = 0
        y_i = []
        for i in range(self.temp_buffer.count):
            reverse_idx = -(i + 1)
            if t_batch[reverse_idx]:
                found_flag=True
                temp=0
                # print(reverse_idx)
            if found_flag is False:
                continue
            temp = temp * args.GAMMA + r_batch[reverse_idx]
            real_cq += temp
            y_i.insert(0, temp)
        # print(y_i, prob_next)
        self.cumV_real = real_cq
        epoch_size = len(y_i)
        y_i = np.reshape(y_i, (epoch_size, 1))
        w_i = np.ones(epoch_size)
        epoch = 0
        # loss = 100
        while epoch < 2000:
            loss, train_op, predicted_Q, target_Q, action = self.critic_train(
                np.reshape(w_i, (epoch_size, 1)), s_batch[:epoch_size], a_batch[:epoch_size],
                np.reshape(y_i, (epoch_size, 1)), self.TimeStep)
            epoch += 1
            print(loss, epoch_size, epoch)

    def train_step(self, s, a, r, feas, s_, t):
        target_q = self.nn.critic_predict(s_, [], feas)

        if t:
            y = r
        else:
            if len(feas) == args.action_dim:
                y = r + args.GAMMA * np.amax(target_q)
            else:
                max_index = -1
                max_value = -1000000.0
                for act in feas:
                    if target_q[act] > max_value:
                        max_index = act
                        max_value = target_q[act]
                y = r + args.GAMMA * target_q[max_index]

        w = np.reshape(np.ones(1), (1,1))
        loss, train_op, predicted_Q, target_Q, action = self.nn.critic_train(
            [[1]], s, [], [a], [[y]], self.TimeStep)
        self.loss_history = np.roll(self.loss_history, 1)
        self.loss_history[0] = loss


    def train_network(self, terminal=False):
        if isinstance(self.dat, ReplayBuffer) or isinstance(self.dat, FlatteningReplayBuffer):
            s_batch, xs_batch, a_batch, r_batch, t_batch, s2_batch, xs2_batch, feasible_action = self.dat.sample_batch(args.batchsize)

        if isinstance(self.nn, BaseNetwork):
            target_q = self.nn.critic_target_predict(s2_batch, xs2_batch, feasible_action)
            if args.is_double: origin_q = self.nn.critic_predict(s2_batch, xs2_batch, feasible_action)
            y_i = []
            dat_size = min(args.batchsize, len(s_batch))
            # w_i = np.array(np.ones(self.MINIBATCH_SIZE))
            w_i = np.ones(dat_size)
            for k in range(dat_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    if args.is_double:
                        max_index = -1
                        max_value = -1000000.0
                        for act in range(args.action_dim):
                            if origin_q[k][act] > max_value:
                                max_index = act
                                max_value = origin_q[k][act]
                        y_i.append(r_batch[k] + args.GAMMA * target_q[k][max_index])
                    else:
                        if len(feasible_action) == args.action_dim:
                            y_i.append(r_batch[k] + args.GAMMA * np.amax(target_q[k]))
                        else:
                            max_index = -1
                            max_value = -1000000.0
                            for act in feasible_action[k]:
                                # if args.use_nost and act % 10 == act // 10: act = 0
                                if target_q[k][act] > max_value:
                                    max_index = act
                                    max_value = target_q[k][act]
                            y_i.append(r_batch[k] + args.GAMMA * target_q[k][max_index])


            loss, train_op, predicted_Q, target_Q, action = self.nn.critic_train(
                np.reshape(w_i, (dat_size, 1)), s_batch, xs_batch, a_batch,
                np.reshape(y_i, (dat_size, 1)), self.TimeStep)
            # print(loss)

            self.loss_history = np.roll(self.loss_history, 1)
            self.loss_history[0] = loss
        elif isinstance(self.nn, BaseNetwork):
            pass
        # elif isinstance(self.nn, PolicyModel):
        #     s_batch, xs_batch, a_batch, r_batch, t_batch, s2_batch, xs2_batch, feasible_action = self.dat.sample_sequence(recent=False, seq_size=None)
        #     """sample_sequence append experiences sequentially until it reaches terminal state
        #     if recent == True, only last episode is selected. if seq_size is specified, those numbers become maximum length of the buffer 
        #     """
        #     self.nn.learn(s_batch, a_batch, r_batch)

    def update_target_network(self):
        if isinstance(self.nn, PDQN):
            self.nn.update_critic()

    def toString(self, learning_instances, denominator):
        msg = ''
        info_col = ['state', 'action', 'reward', 't', 'next state']
        for k in range(len(learning_instances[0])):
            for i in range(len(info_col)):
                msg += info_col[i] + ':['
                if type(learning_instances[i][k]) == np.ndarray:
                    for l in range(len(learning_instances[i][k])):
                        msg += str(learning_instances[i][k][l]) + ','
                else:
                    msg += str(learning_instances[i][k])
                msg += ']' + denominator
            msg += '\n'

        return msg
