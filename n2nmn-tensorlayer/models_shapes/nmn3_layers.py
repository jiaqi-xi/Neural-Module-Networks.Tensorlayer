from __future__ import absolute_import, division, print_function
import sys

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow import convert_to_tensor as to_T

sess = tf.Session()
tl.layers.initialize_global_variables(sess)

def conv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
               bias_term=True, weights_initializer=None, biases_initializer=None, reuse=None):
    # input has shape [batch, in_height, in_width, in_channels]
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, in_channels, out_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, input_dim, output_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)
        if not reuse:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.nn.l2_loss(weights))
    
    conv = tf.nn.conv2d(bottom, filter=weights,
          strides=[1, stride, stride, 1], padding=padding)
    if bias_term:
        conv = tf.nn.bias_add(conv, biases)

    return conv

def conv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                    bias_term=True, weights_initializer=None, biases_initializer=None, reuse=None):
    
     # input has shape [batch, in_height, in_width, in_channels]
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, in_channels, out_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, input_dim, output_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)
        if not reuse:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.nn.l2_loss(weights))
    
    conv = tf.nn.conv2d(bottom, filter=weights,
          strides=[1, stride, stride, 1], padding=padding)
    if bias_term:
        conv = tf.nn.bias_add(conv, biases)

    relu = tf.nn.relu(conv)
    return relu

def deconv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                 bias_term=True, weights_initializer=None, biases_initializer=None):
    # input_shape is [batch, in_height, in_width, in_channels]
    input_shape = bottom.get_shape().as_list()
    batch_size, input_height, input_width, input_dim = input_shape
    output_shape = [batch_size, input_height*stride, input_width*stride, output_dim]

    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer_conv2d()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, out_channels, in_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, output_dim, input_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)
        if not reuse:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.nn.l2_loss(weights))
    net = tl.layers.InputLayer(inputs=bottom, name=name+'input')

    deconv = tl.layers.DeConv2dLayer(net, act=tf.identity, shape=[kernel_size, kernel_size, output_dim, input_dim],
                               output_shape=output_shape, strides=[1, stride, stride, 1],
                               padding=padding, W_init=weights_initializer, b_init=biases_initializer,
                               name=name+'deconv2d')
    return deconv.outputs

def deconv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                      bias_term=True, weights_initializer=None, biases_initializer=None, reuse=None):
    deconv = deconv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                          bias_term, weights_initializer, biases_initializer, reuse=reuse)
    # relu = tl.layers.PReluLayer(deconv)
    relu = tf.nn.relu(deconv)
    return relu

def pooling_layer(name, bottom, kernel_size, stride):
    #pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
    #    strides=[1, stride, stride, 1], padding='SAME', name=name)
    net = tl.layers.InputLayer(inputs=bottom, name=name+'input')
    pool = tl.layers.PoolLayer(net, ksize=[1, kernel_size, kernel_size, 1],
           strides=[1, stride, stride, 1], padding='SAME', pool=tf.nn.max_pool, name=name+'pool')
    return pool.outputs

def fc_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None, reuse=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    # flat_bottom = tf.reshape(bottom, [-1, input_dim])
    net = tl.layers.InputLayer(inputs=bottom, name=name+'input')
    flat_bottom = tl.layers.ReshapeLayer(net, [-1, input_dim], name=name+'reshape').outputs
    
    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.contrib.layers.xavier_initializer()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # weights has shape [input_dim, output_dim]
        weights = tf.get_variable("weights", [input_dim, output_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)
        if not reuse:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 tf.nn.l2_loss(weights))

    if bias_term:
        fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    else:
        fc = tf.matmul(flat_bottom, weights)
    return fc

def fc_relu_layer(name, bottom, output_dim, bias_term=True,
                  weights_initializer=None, biases_initializer=None, reuse=None):
    fc = fc_layer(name, bottom, output_dim, bias_term, weights_initializer,
                  biases_initializer, reuse=reuse)
    relu = tf.nn.relu(fc)
    return relu

# convnet built for shapes dataset
def shapes_convnet(input_batch, hidden_dim=64, output_dim=64,
    scope='shapes_convnet', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        conv_1 = conv_relu_layer('conv_1', input_batch, kernel_size=10, stride=10,
            output_dim=hidden_dim, padding='VALID')
        conv_2 = conv_relu_layer('conv_2', conv_1, kernel_size=1, stride=1,
            output_dim=output_dim)

    return conv_2

# following convnet are safe even for empty data
def empty_safe_1x1_conv(name, bottom, output_dim, reuse=None):
    # use this for 1x1 convolution in modules to avoid the crash.
    bottom_shape = tf.shape(bottom)
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name, reuse=reuse):
        # initialize the variables
        weights_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.constant_initializer(0.)
        weights = tf.get_variable('weights', [input_dim, output_dim],
            initializer=weights_initializer)
        biases = tf.get_variable('biases', output_dim,
            initializer=biases_initializer)

        conv_flat = tf.matmul(tf.reshape(bottom, [-1, input_dim]), weights) + biases
        conv = tf.reshape(conv_flat, to_T([bottom_shape[0], bottom_shape[1], bottom_shape[2], output_dim]))

    return conv

#  use this for arbitrary convolution in modules to avoid the crash.
def empty_safe_conv(name, bottom, kernel_size, stride, output_dim, padding='SAME',
          bias_term=True, weights_initializer=None,
          biases_initializer=None, reuse=None):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Conv2D': 'Conv2D_handle_empty_batch'}):
        return conv_layer(name, bottom, kernel_size, stride, output_dim,
                    padding, bias_term, weights_initializer,
                    biases_initializer, reuse=reuse)

@tf.RegisterGradient('Conv2D_handle_empty_batch')
def _Conv2DGrad(op, grad):
    with tf.device('/cpu:0'):
        filter_grad = tf.nn.conv2d_backprop_input(  # compute gradient_filter
            tf.shape(op.inputs[0]), op.inputs[1], grad, op.get_attr('strides'),
            op.get_attr('padding'), op.get_attr('use_cudnn_on_gpu'), op.get_attr('data_format'))
        input_grad = tf.nn.conv2d_backprop_filter(  # compute gradient_input
            op.inputs[0],  tf.shape(op.inputs[1]), grad, op.get_attr('strides'),
            op.get_attr('padding'),op.get_attr('use_cudnn_on_gpu'), op.get_attr('data_format'))
        return [filter_grad, input_grad]
