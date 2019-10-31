#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 19:10
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0


import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def add_layer(inputs, in_size, out_size, activition_function=None):
    weights = tf.Variable(tf.random.normal([in_size, out_size]))
    biase = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    out = tf.add(tf.matmul(inputs, weights), biase)
    if activition_function:
        out = activition_function(out)
    return out


def compute_accuracy(v_xs, v_ys, sess):
    """
    计算acc
    :param v_xs:
    :param v_ys:
    :param sess:
    :return:
    """
    # 全局量 prediction
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 预测值每行是10列，tf.argmax(数据，axis），相等为1，不想等为0
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 计算平均值， 即计算准确率
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 运行计算acc这一步
    result = sess.run(acc, feed_dict={xs: v_xs, ys: v_ys})
    return result


xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activition_function=tf.nn.softmax)

loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                     reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={xs: batch_x,
                                       ys: batch_y})
        if i % 50 == 0:
            acc = compute_accuracy(mnist.test.images, mnist.test.labels, sess)
            print(acc)
            if acc > 0.87:
                print("end! ")
                saver.save(sess, "./model/ppp.mdel")
                break
