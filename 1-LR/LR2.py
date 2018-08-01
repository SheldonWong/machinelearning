# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random


class LogisticRegression():

    def __init__(self):
        pass

    '''
    @desc:
    @param:
    @return:
    '''
    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    '''
    @desc:训练
    @param:X，y，alpha，iter
    @return:weights_array，权重向量
    '''

    def fit(self,X,y,alpha=0.8,iter=1000):
        X_array = np.array(X)
        y_array = np.array(y)
        m,n = np.shape(X_array)

        weights_array = np.ones(n)

        for i in range(iter):
            randIndex = random.randint(0,m-1)
            
            h = self.sigmoid(np.dot(X_array[randIndex],weights_array))
 
            grad = X[randIndex] * (y_array[randIndex] - h)
            weights_array = weights_array + alpha * grad

        return weights_array

    


    '''
    @desc: 传进来一个特征矩阵，预测标签向量
    @param: X_array
    @return: y_array,
    '''

    def predict(self,X_array,weights):
        prob = self.sigmoid(np.dot(X_array,weights))
        return np.where(prob>=0.5,1,0)

    '''
    @desc:传进来真是的标签向量和预测的标签向量，返回预测准确率
    @param:标签向量 y_array
    @return:预测准确率 
    '''
    def score(self,y_array,y_hat):
        res = np.where(y_array == y_hat)
        return len(np.where(res==True))/(1.0*len(res))


    def plotBestFit(self,X_array,label_array,weights):
        #Return self as an ndarray object
        #便于取值
        weights = weights_array
        dataArr = X_array
        n = shape(dataMat)[0]
        #散点图
        xcord1 = []; ycord1 = []
        xcord2 = []; ycord2 = []
        for i in range(n):
            if int(labelMat[i][0]) == 1:
                xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
            else:
                xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        #决策面 sigmoid函数值为0.5是决策面，也就是线性变换那一部分值为0,解出x2和x1的关系
        # w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2
        x = arange(-4.0, 4.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]
        ax.plot(x, y)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.show()