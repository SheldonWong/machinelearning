# -*- coding: utf-8 -*-

'''
1. 载入数据
2. 切分数据
3. 初始化模型
4. 拟合模型，得到参数
5. 在testSet上预测
6. 评测指标
'''

from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split

#1. 载入数据
iris = datasets.load_iris()
# <class 'numpy.ndarray'>
# print(type(X))
X = iris.data[:, :2]  
y = iris.target


#2. 数据切分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#3. 初始化模型
#smaller values specify stronger regularization.
logreg = linear_model.LogisticRegression(C=1e5)

#4. 拟合模型
logreg.fit(X_train, y_train)

#5. 预测与评测
# http://d0evi1.com/sklearn/model_evaluation/ sklearn中的模型评估
# https://www.zhihu.com/question/20778853 Mixin
#from .metrics import accuracy_score
#return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
print(logreg.score(X_test,y_test))

