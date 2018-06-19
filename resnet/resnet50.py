# -*- coding:utf-8 -*-
"""
本模块的作用的构建Resnet50，提供v1和v2版本
"""
import cv2
import tensorflow as tf
import os.path as osp
import sys
import numpy as np
from tensorflow.contrib import slim
# BN的滑动平均参数
_BATCH_NORM_DECAY = 0.997
# BN的epsilon
_BATCH_NORM_EPSILON = 1e-5
# 默认v2版本
DEFAULT_VERSION = 1
CHAR_NAME = "abcdefghijklmnopqrstuvwxyz"
# 根据论文deep Residual Learning for Image Recognition表1
weight_decay = 1e-5


def l2_regularizer(decay=0.0005, scope=None):
    def regularizer(tensor):
        with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
            l2_weight = tf.convert_to_tensor(decay, dtype=tensor.dtype.base_dtype, name='weight_decay')
            # tf.nn.l2_loss(t)的返回值是output = sum(t ** 2) / 2
            return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

    return regularizer




def batch_norm(inputs, training, name=None):
    """
    批归一化
    :param inputs: [批，高，宽，通道数]
    :param training: bool型，是否在训练
    :param name:
    :return:
    """
    return slim.batch_norm(inputs=inputs, decay=_BATCH_NORM_DECAY, center=True, scale=True,
                           epsilon=_BATCH_NORM_EPSILON,is_training=training,fused=True,scope=name,
                           updates_collections=tf.GraphKeys.UPDATE_OPS)
    # return tf.layers.batch_normalization(inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
    #                                      epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
    #                                      training=training, fused=True, name=name)

def dense(inputs, out_dimension, use_biase=False,name=None,trainable=True):
    """
    rank = 2
    :param inputs:
    :param out_dimension:
    :param use_biase:
    :param name:
    :param trainable:
    :return:
    """
    in_channel = inputs.shape[-1]
    with tf.variable_scope(name):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        # [filter_height, filter_width, in_channels, out_channels]`
        kernel = tf.get_variable(name='kernel', shape=[in_channel, out_dimension],
                                 initializer=init_weights, trainable=trainable,
                                 regularizer=l2_regularizer(weight_decay))
        if use_biase:
            biase = tf.get_variable(name='bias', shape=[out_dimension], initializer=init_biases, trainable=trainable)
            out=tf.nn.xw_plus_b(inputs,kernel,biase)
        else:
            out = tf.matmul(inputs,kernel)

        return out

    #
    # return slim.fully_connected(inputs=inputs,num_outputs=out_dimension,
    #                             activation_fn=None,scope=name,weights_regularizer=slim.l2_regularizer(weight_decay),
    #                             weights_initializer=slim.variance_scaling_initializer())
    # return tf.layers.dense(
    #     inputs=inputs, units=out_dimension, use_bias=use_biase,
    #     name=name,kernel_initializer=tf.variance_scaling_initializer())


def conv2d(inputs, out_channal, kernel_size, strides, use_bias=False, trainable=True, name=None):
    """
    卷积
    :param inputs: [批，高，宽，通道数]
    :param out_channal: 输出通道数
    :param kernel_size: 整数，卷积核大小
    :param strides: 步长， int
    :param name:
    :param use_bias:
    :return:
    """
    padding = 'SAME'
    if strides > 1:  # 如果步长大于1，需要填充
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], constant_values=0)
        padding = 'VALID'
    in_channel = inputs.shape[-1]
    with tf.variable_scope(name):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        # [filter_height, filter_width, in_channels, out_channels]`
        kernel = tf.get_variable(name='kernel', shape=[kernel_size, kernel_size, in_channel, out_channal],
                                 initializer=init_weights,trainable=trainable,
                                 regularizer=l2_regularizer(weight_decay))
        conv = tf.nn.conv2d(inputs,kernel,[1,strides,strides,1],padding=padding)
        if use_bias:
            biase = tf.get_variable(name='bias',shape=[out_channal], initializer=init_biases, trainable=trainable)
            conv = tf.nn.bias_add(conv, biase)

        return conv

    #
    # return slim.conv2d(inputs=inputs, num_outputs=out_channal,
    #                    kernel_size=kernel_size, padding=padding,activation_fn=None, scope=name,
    #                    stride=strides,weights_regularizer=slim.l2_regularizer(weight_decay),
    #                    weights_initializer=slim.variance_scaling_initializer())
    # return tf.layers.conv2d(inputs=inputs, filters=out_channal,kernel_size=kernel_size,
    #                         strides=strides,padding=padding, use_bias=use_bias,
    #                         kernel_initializer=tf.variance_scaling_initializer(),name=name)

def conv2d_transpose(inputs, out_channal, kernel_size, strides, name=None, trainable=True, use_bias = False):
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    # don't know why cannot write in_channel = tf.shape(inputs)[3] danteng...
    in_channel = inputs.shape[3]
    with tf.variable_scope(name):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
        init_biases = tf.constant_initializer(0.0)
        # [height, width, output_channels, in_channels]
        kernel = tf.get_variable(name='kernel', shape=[kernel_size, kernel_size, out_channal, in_channel],
                                 initializer=init_weights, trainable=trainable,
                                 regularizer=l2_regularizer(weight_decay))
        conv = tf.nn.conv2d_transpose(value=inputs, filter=kernel,
                                      output_shape=[batch_size, strides*height, strides*width, out_channal],
                                      strides = [1,strides,strides,1])
        if use_bias:
            biase = tf.get_variable(name='bias', shape=[out_channal], initializer=init_biases, trainable=trainable)
            conv = tf.nn.bias_add(conv, biase)

        return conv
    #
    # return slim.conv2d_transpose(inputs=inputs,num_outputs=out_channal,scope=name,
    #                              kernel_size=kernel_size, stride=strides,padding='same',activation_fn=None,
    #                              weights_regularizer=slim.l2_regularizer(weight_decay),
    #                              weights_initializer=slim.variance_scaling_initializer())

    # return tf.layers.conv2d_transpose(
    #     inputs=inputs, filters=out_channal,kernel_size=kernel_size,strides=strides,
    #     padding='same',use_bias=False, kernel_initializer=tf.variance_scaling_initializer())


def _bottleneck_block_v1(inputs, out_channal, training, strides, scope):
    """
    论文v1 图5 右边那个
    :param inputs: tensor [批，高，宽，通道数]
    :param out_channal: 输出通道数
    :param training: bool是否训练
    :param strides: 步长
    :param scope: str, 一般为'2a', '3b'等，表示第2、3个stage，第'a','b'个模块
    :return:
    """
    conv_name = 'res'+scope+"_branch"
    bn_name = 'bn'+scope+"_branch"

    in_channal = inputs.get_shape()[3]
    if in_channal == out_channal:
        shortcut = inputs
    # 通道数翻番，则分辨率减半
    elif in_channal * 2 == out_channal:
        assert strides == 2, " in {}, line {},when channal doubles, resolution must halve".\
            format(osp.dirname(__file__), sys._getframe().f_lineno)
        shortcut = conv2d(inputs=inputs, out_channal=out_channal,kernel_size=1,strides=strides, name=conv_name+"1")
        shortcut = batch_norm(inputs=shortcut,training=training,name=bn_name+"1")
    else: # 根据论文，conv1的输出通道是64， conv2的输出通道却是256，是四倍关系；而分辨率却只减半
        shortcut = conv2d(inputs=inputs, out_channal=out_channal,kernel_size=1,strides=strides, name=conv_name+"1")
        shortcut = batch_norm(inputs=shortcut,training=training, name=bn_name+"1")

    outputs = conv2d(inputs=inputs, out_channal=out_channal/4, kernel_size=1, strides=1, name=conv_name+"2a")
    outputs = batch_norm(outputs,training, name=bn_name+"2a")
    outputs = tf.nn.relu(outputs, name=conv_name+"2a_relu")

    # 可能的分辨率下降，发生在中间这层
    outputs = conv2d(inputs=outputs,out_channal=out_channal/4,kernel_size=3, strides=strides, name=conv_name+"2b")
    outputs = batch_norm(outputs, training, name=bn_name+"2b")
    outputs = tf.nn.relu(outputs, name=conv_name+"2b_relu")

    outputs = conv2d(inputs=outputs,out_channal=out_channal, kernel_size=1, strides=1, name=conv_name+"2c")
    outputs = batch_norm(outputs, training, name=bn_name+"2c")
    outputs = tf.nn.relu(outputs, name=conv_name+"2c_relu")
    outputs = tf.add_n([shortcut, outputs],name="res"+ scope)

    return tf.nn.relu(outputs, name="res"+scope+"_out")


def _bottleneck_block_v2(inputs, out_channal, training, strides, scope):
    """
    论文v2 图1 右边那个
    :param inputs: tensor [批，高，宽，通道数]
    :param out_channal: 输出通道数
    :param training: bool是否训练
    :param strides: 步长
    :return:
    """
    in_channal = inputs.get_shape()[3]

    if in_channal == out_channal:
        shortcut = inputs
    # 通道数翻番，则分辨率减半
    elif in_channal * 2 == out_channal:
        assert strides == 2, " in {}, line {},when channal doubles, resolution must halve". \
            format(osp.dirname(__file__), sys._getframe().f_lineno)
        shortcut = conv2d(inputs=inputs, out_channal=out_channal, kernel_size=1, strides=strides)
        shortcut = batch_norm(inputs=shortcut, training=training)

    else: # 根据论文，conv1的输出通道是64， conv2的输出通道却是256，是四倍关系；而分辨率却只减半
        shortcut = conv2d(inputs=inputs, out_channal=out_channal, kernel_size=1,strides=strides)
        shortcut = batch_norm(inputs=shortcut,training=training)

    outputs = batch_norm(inputs,training)
    outputs = tf.nn.relu(outputs)
    outputs = conv2d(inputs=outputs, out_channal=out_channal / 4, kernel_size=1, strides=1)

    outputs = batch_norm(outputs,training)
    outputs = tf.nn.relu(outputs)
    outputs = conv2d(inputs=outputs, out_channal=out_channal / 4, kernel_size=3, strides=strides)

    outputs = batch_norm(outputs,training)
    outputs = tf.nn.relu(outputs)
    outputs = conv2d(inputs=outputs, out_channal=out_channal, kernel_size=1, strides=1)

    return outputs + shortcut


def _block_layer(inputs, out_channal, version, num_block, training, stage):
    """
    构成一个block_layer层
    :param inputs: tensor [批，高，宽，通道数]
    :param out_channal: 输出通道数
    :param version: 整数，1或者2，可以选择_bottleneck_block_v1或_bottleneck_block_v2
    :param num_block: 该layer层的block个数
    :param training: 是否训练
    :param stage: 整数，表示该层属于第几个stage
    :return:
    """
    layer_name = str(stage)
    block_fn = _bottleneck_block_v1 if version==1 else _bottleneck_block_v2
    # 每个block_layer层的第一个block，分辨率都会减半
    inputs = block_fn(inputs, out_channal, training, strides=2, scope=layer_name+CHAR_NAME[0])
    for i in range(1,num_block):
        inputs = block_fn(inputs, out_channal, training, strides=1, scope=layer_name+CHAR_NAME[i])

    return inputs


class Model(object):
    def __init__(self, resnetlist, version=DEFAULT_VERSION, first_pool=True):
        """

        :param resnetlist: 网络列表，包含N个元素的列表，N表示block_layer的个数。每个元素是一个包含两个整数的列表，
        第一个整数代表该layer的block个数，第二个整数代表该layer的输出通道数
        :param version: 版本号，只能取1或者2
        :param first_pool: bool，v1论文表1的conv2最上面那层pool层还要不要
        """
        self.resnetlist = resnetlist
        self.version = version
        self.first_pool = first_pool
        if version not in (1, 2):
            raise ValueError('Resnet version should be 1 or 2..')

    def __call__(self, inputs, training):
        # 记录分辨率，它反映了最终的feature map缩小为原来的倍数
        resolution = 1
        inputs = conv2d(inputs=inputs, out_channal=64, kernel_size=7, strides=2,name='conv1')
        resolution *= 2


        inputs = batch_norm(inputs, training, name='bn_conv1')
        inputs = tf.nn.relu(inputs, name="conv1_relu")
        if self.first_pool:
            inputs = slim.max_pool2d(inputs=inputs, kernel_size=3, stride=2, padding="SAME",scope="pool1")
            # inputs = tf.nn.max_pool(value=inputs, ksize=[1,3,3,1], strides=[1,2, 2,1], padding="SAME")
            resolution *= 2

        layer_values = []
        scale = []
        for i, layer in enumerate(self.resnetlist):
            inputs = _block_layer(inputs, out_channal=layer[1], version=self.version,
                                  num_block=layer[0],training=training, stage=i+2)
            layer_values.append(inputs)
            resolution *= 2
            scale.append(resolution)

        # 返回最终的layer_values和分辨率
        return layer_values, scale

if __name__=='__main__':
    image = cv2.imread('E:\\DLnet\\ctpn\\dataset\\for_test\\TB1_hpqJFXXXXX2aFXXunYpLFXX.jpg')
    image = image.astype(np.float32)
    image -= [102.9801, 115.9465, 122.7717]
    image = cv2.resize(image, (1024, 1024))
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, 0)
    print(image.shape)
    # with tf.device('/gpu:0'):
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    resnet = Model([[3, 256], [4, 512], [6, 1024],  [3, 2048]])
    fp, resolution2= resnet(image, True)
    # f,  = sess.run([fp])

    print(fp[0].shape,fp[1].shape,fp[2].shape,fp[3].shape)
    print(resolution2)



