#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 15:59
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import tensorflow as tf
import numpy as np


# w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=np.float32, name="weights")
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name="biases")
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "model/save_net.ckpt")
#
W = tf.Variable(np.arange(6).reshape(2, 3), name='weights', dtype=tf.float32)


b = tf.Variable(np.arange(3).reshape(1, 3), name='biases', dtype=tf.float32)

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

