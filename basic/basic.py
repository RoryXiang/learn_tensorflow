#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/23 15:29
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0


import tensorflow as tf
import numpy as np


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 1

weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
print(weights)
weights = tf.Variable(tf.random.normal(shape=[1], dtype=np.float32))
print(weights)
biase = tf.Variable(tf.zeros(1))

y = weights * x_data + biase

# loss function
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

# 训练的目的是为了最小化loss
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 创建session 并初始化
sess = tf.Session()
sess.run(init)

# 保存对象
saver = tf.train.Saver()

# 开始训练
for step in range(201):
    sess.run(train)
    if step % 50 == 0:
        print(step, sess.run(weights), sess.run(biase))
        # 保存模型
        # saver.save(sess, f"./model/model_{step}.ckpt")


"""Tensorflow的Session,对话控制模块，可以用sesison.run来运行框架中的某一个
点的功能"""

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)
# pp = tf.multiply(matrix1, matrix2)
sess = tf.Session()
result = sess.run(product)
# rr = sess.run(pp)

print("matmul: ", result)
# print("maltiply: ", rr)

sess.close()


"""TF variable"""

state = tf.Variable(0, name="a")  # 使用tensorflow在默认的图中创建节点，这个节点是一个变量
one = tf.constant(1)  # 此处调用了tf的一个函数，用于创建常量
new_value = tf.add(state, one)  # 对常量与变量进行简单的加法操作，
# 这点需要说明的是： 在TensoorFlow中，所有的操作op，变量都视为节点，tf.add()
# 的意思就是在tf的默认图中添加一个op，这个op是用来做加法操作的。

update = tf.assign(state, new_value)
# 这个操作是：赋值操作。将new_value的值赋值给state变量,update只是一个用于sess
# 的变量

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 对变量进行初始化，执行（run）init语句
    for i in range(3):
        sess.run(update)
        print(sess.run(state))


"""TF placeholder
placeholder 是 Tensorflow 中的占位符，暂时储存变量.
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以
这种形式传输数据 sess.run(***, feed_dict={input: **}).
"""
input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [3.], input2: [5]}))
