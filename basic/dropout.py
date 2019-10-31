#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 13:55
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
# LabelBinarizer  将非二值化特征二值化： ["yes", "no"] -> [1, 0]
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


digits = load_digits()
x_data = digits.data
y_data = digits.target
y_data = LabelBinarizer().fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3)


def add_layer(input, input_size, output_size, layer_name,
              activation_function=None):
    weights = tf.Variable(tf.random.normal([input_size, output_size]),
                          dtype=np.float32)
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    wx_plus_bias = tf.add(tf.matmul(input, weights), biases)
    wx_plus_bias = tf.nn.dropout(wx_plus_bias, keep_prob=0.1)
    if activation_function:
        output = activation_function(wx_plus_bias)
    else:
        output = wx_plus_bias
    return output


xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, 64, 50, "l1", activation_function=tf.nn.tanh)  # l1
prediction = add_layer(l1, 50, 10, "l2", activation_function=tf.nn.softmax)
# l2
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]),
                               )
# 因为cross_entropy 是一个标量 所以定义tf.summary.scalar

tf.summary.scalar("loss", cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 合并所有的summary
    merged = tf.summary.merge_all()
    # 得到summary 的FileWriter
    train_writer = tf.summary.FileWriter("logs/train/", sess.graph)
    test_writer = tf.summary.FileWriter("logs/test/", sess.graph)

    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_train,
                                        ys: y_train,
                                        keep_prob: 0.5})
        if i % 50 == 0:
            train_loss = sess.run(merged, feed_dict={xs: x_train,
                                                     ys: y_train,
                                                     keep_prob: 0.5})
            test_loss = sess.run(merged, feed_dict={xs: x_test,
                                                    ys: y_test,
                                                    keep_prob: 0.5})
            # train_writer.add_summary(train_loss, i)
            # mm = tf.compat.as_str(train_loss)
            print("train los: ", train_loss, " test los: ", test_loss)