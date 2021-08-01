import tensorflow as tf
import os
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import initializers
from config import args
is_train = tf.constant(args.is_train)

def dense_layer(x, output_dim, scope,
                weights_initializer=initializers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer,
                use_bias=True, activation_fn=None):
    """
    A convenient function for constructing fully connected layers
    """

    shape = x.get_shape().as_list()
    if len(shape) == 2:     # if the previous layer is fully connected, the shape of X is (N, D)
        D = shape[1]
    else:                   # if the previous layer is convolutional, the shape of X is (N, H, W, C)
        N, H, W, C = shape
        D = H * W * C
        x = tf.reshape(x, (-1, D))

    with tf.variable_scope(scope):
        w = tf.get_variable("weights", shape=(D, output_dim), initializer=weights_initializer)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        # calculate
        x = tf.matmul(x, w)

        if use_bias:
            b = tf.get_variable("biases", shape=output_dim, initializer=biases_initializer)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
            x = tf.nn.bias_add(x, b)
        # x = batch_norm(x, is_train=is_train, scope='bn' )
    if activation_fn != None: return activation_fn(x)
    else: return x


def conv2d(x, filter_size, stride, output_size, initializer, scope, use_bias, padding="VALID"):
    """
    A convenient function for constructing convolutional layer
    """

    # input x should be (N, H, W, C)
    N, H, W, C = x.get_shape().as_list()
    stride = (1, stride, stride, 1)

    with tf.variable_scope(scope):
        w = tf.get_variable("W", shape=(filter_size, filter_size, C, output_size), initializer=initializer,
                            dtype=tf.float32)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w)
        x = tf.nn.conv2d(x, w, strides=stride, padding=padding)

        if use_bias:
            b = tf.get_variable("b", shape=output_size, initializer=tf.constant_initializer(0.01), dtype=tf.float32)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b)
            x = tf.nn.bias_add(x, b)

    return x


def batch_norm(x, is_train, scope):
    """
    A wrapper for batch normalization layer
    """
    train_time = tf.contrib.layers.batch_norm(x, decay=0.9, scope="%s/bn" % scope, center=True, scale=False,
                                              updates_collections=None, is_training=True, reuse=None)
    test_time = tf.contrib.layers.batch_norm(x, decay=0.9, scope="%s/bn" % scope, center=True, scale=False,
                                             updates_collections=None, is_training=False, reuse=True)

    x = tf.cond(is_train, lambda: train_time, lambda: test_time)
    return x
def noisy_dense(x, isTrain, size, scope, bias=True, activation_fn=tf.identity):

    # the function used in eq.7,8
    def f(x):
        return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
    # Initializer of \mu and \sigma
    mu_init = tf.random_uniform_initializer(minval=-1*1/np.power(x.get_shape().as_list()[1], 0.5),
                                                maxval=1*1/np.power(x.get_shape().as_list()[1], 0.5))
    sigma_init = tf.constant_initializer(0.4/np.power(x.get_shape().as_list()[1], 0.5))
    # Sample noise from gaussian
    p = sample_noise([x.get_shape().as_list()[1], 1])
    q = sample_noise([1, size])
    f_p = f(p); f_q = f(q)
    w_epsilon = f_p*f_q; b_epsilon = tf.squeeze(f_q)

    # w = w_mu + w_sigma*w_epsilon
    with tf.variable_scope(scope):
        w_mu = tf.get_variable("w_mu", [x.get_shape()[1], size], initializer=mu_init)
        w_sigma = tf.get_variable("w_sigma", [x.get_shape()[1], size], initializer=sigma_init)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w_mu)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, w_sigma)

        if isTrain:
            w = w_mu + tf.multiply(w_sigma, w_epsilon)
        else:
            w = w_mu

        ret = tf.matmul(x, w)
        if bias:
            # b = b_mu + b_sigma*b_epsilon
            b_mu = tf.get_variable("b_mu", [size], initializer=mu_init)
            b_sigma = tf.get_variable("b_sigma", [size], initializer=sigma_init)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b_mu)
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, b_sigma)

            if isTrain:
                b = b_mu + tf.multiply(b_sigma, b_epsilon)
            else:
                b = b_mu

            return activation_fn(ret + b)
        else:
            return activation_fn(ret)

def sample_noise(shape):
    noise = tf.random_normal(shape)
    return noise

def save(sess, save_dir, saver):
    """
    Save all model parameters and replay memory to self.save_dir folder.
    The save_path should be models/env_name/name_of_agent.
    """
    # path to the checkpoint name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, "AkC")
    print("Saving the model to path %s" % path)
    # self.memory.save(self.save_dir)
    print(saver.save(sess, path))
    print("Done saving!")


def restore(sess, save_dir, saver):
    """
    Restore model parameters and replay memory from self.save_dir folder.
    The name of the folder should be models/env_name
    """
    # TODO: Need to find a better way to store memory data. Storing all states is not efficient.
    ckpts = tf.train.get_checkpoint_state(save_dir)
    if ckpts and ckpts.model_checkpoint_path:
        ckpt = ckpts.model_checkpoint_path
        saver.restore(sess, ckpt)
        # self.memory.restore(save_dir)
        print("Successfully load the model %s" % ckpt)
        # print("Memory size is:")
        # self.memory.size()
    else:
        print("Model Restore Failed %s" % save_dir)
