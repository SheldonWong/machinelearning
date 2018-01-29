# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


'''
1. 准备数据
2. 模型
3. k-fold交叉验证
4. 输出结果
'''

iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

#数据切分
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#smaller values specify stronger regularization.
logreg = linear_model.LogisticRegression(C=1e5)

#logreg.fit(X, y)

scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print(scores)

