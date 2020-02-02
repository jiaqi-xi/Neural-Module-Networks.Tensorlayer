from __future__ import absolute_import, division, print_function
import sys

print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")

import tensorflow as tf
import tensorlayer as tl

print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")

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
    print(str(bias_term) + '\n')
    print(bottom)
    print('\n')
    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")

    '''
    net_l = tl.layers.InputLayer(inputs=bottom, name=name+'input')
    
    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")

    conv_l = tl.layers.Conv2dLayer(net_l, act=tf.identity,
                                shape=[kernel_size, kernel_size, input_dim, output_dim],
                                strides=[1, stride, stride, 1],
                                padding=padding,
                                W_init=weights_initializer, b_init=biases_initializer,
                                name=name+'convlayer'
                                )
    # 1.4.1 不同于 1.5.0 没有 use_cudnn_on_gpu 和 data_format 两个参数

    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")
    print(conv_l.outputs)
    print("------------------------------")
    '''
    print(conv)
    return conv

def conv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                    bias_term=True, weights_initializer=None, biases_initializer=None, reuse=None):
    conv = conv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                      bias_term, weights_initializer, biases_initializer, reuse=reuse)
    # print(type(conv))
    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")

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
    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")
    net = tl.layers.InputLayer(inputs=bottom, name=name+'input')

    print("------------------------------" + " cnn.py " + str(sys._getframe().f_lineno) + "------------------------------")
    
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
