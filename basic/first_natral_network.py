#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 16:34
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import tensorflow as tf
import numpy as np


def add_layer(inputs, inputs_size, output_size, activation_function=None):
    weights = tf.Variable(tf.random.normal([inputs_size, output_size]),
                          dtype=tf.float32)
    biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    out = tf.add(tf.matmul(inputs, weights), biases)
    if activation_function:
        out = activation_function(out)
    return out


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, size=x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)


# loss 函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

# 优化函数
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(optimizer, feed_dict={xs: x_data,
                                       ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data,
                                            ys: y_data}))
