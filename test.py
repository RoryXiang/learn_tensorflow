#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019/10/24 11:25
# @Author  : RoryXiang (pingping19901121@gmail.com)
# @Link    : ""
# @Version : 1.0

import numpy as np
from sklearn import preprocessing
from sklearn import tree

a = np.array([1,2,3,4])[:, np.newaxis]
b = np.square(a)
print(b)

featureList=[[1,0],[1,1],[0,0],[0,1]]
# 标签矩阵
labelList=['yes', 'no', 'no', 'yes']
# 将标签矩阵二值化
lb = preprocessing.LabelBinarizer()
dummY=lb.fit_transform(labelList)
# print(dummY)
# 模型建立和训练
clf = tree.DecisionTreeClassifier()
clf = clf.fit(featureList, dummY)
p=clf.predict([[0,1]])
# print(p)#取消注释可以查看p的值

# 逆过程
yesORno=lb.inverse_transform(p)
print(yesORno)
