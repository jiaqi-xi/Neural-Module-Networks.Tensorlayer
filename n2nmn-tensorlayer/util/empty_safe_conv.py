from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from tensorflow import convert_to_tensor as to_T

from util.cnn import conv_layer as conv

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
        return conv(name, bottom, kernel_size, stride, output_dim,
                    padding, bias_term, weights_initializer,
                    biases_initializer, reuse=reuse)

@tf.RegisterGradient('Conv2D_handle_empty_batch')
def _Conv2DGrad(op, grad):
    with tf.device('/cpu:0'):
        filter_grad = tf.nn.conv2d_backprop_input(  # 计算卷积相对于过滤器的梯度
            tf.shape(op.inputs[0]), op.inputs[1], grad, op.get_attr('strides'),
            op.get_attr('padding'), op.get_attr('use_cudnn_on_gpu'), op.get_attr('data_format'))
        input_grad = tf.nn.conv2d_backprop_filter(  # 计算卷积相对于输入的梯度
            op.inputs[0],  tf.shape(op.inputs[1]), grad, op.get_attr('strides'),
            op.get_attr('padding'),op.get_attr('use_cudnn_on_gpu'), op.get_attr('data_format'))
        return [filter_grad, input_grad]
