# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
from LR2 import LogisticRegression

def loadDataset(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append([int(lineArr[2])])
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
批梯度下降，
更新一次参数，需要用到所有数据点
'''
def gradAscent(dataMatIn,classLabels):
    #100 * 3
    dataMatrix = mat(dataMatIn)
    #100 * 1
    labelMat = mat(classLabels)
    #100 3
    m,n = shape(dataMatrix)
    alpha = 0.001
    iter = 500
    # 3 * 1
    weights = ones((n,1))
    for k in range(iter):
        #计算所有数据的预测值，n*1
        h = sigmoid(dot(dataMatrix,weights))
        #计算所有数据的误差，n*1
        error = labelMat-h
        #根据所有数据的误差向量更新参数
        weights = weights + alpha * dot(dataMatrix.T,error)
    return weights

'''
决策面可视化
'''
def plotBestFit(dataMat,labelMat,weights):
    #Return self as an ndarray object
    #便于取值
    weights = array(weights)
    dataArr = array(dataMat)
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

#随机梯度下降，每次更新参数使用一条数据
def stocGradAscent(dataMat,classLabels,alpha = 0.01,iter=100):
    m,n = shape(dataMat)
    weights = ones(n)
    for k in range(iter):
        for i in range(m):
            #第i条数据的预测类别
            h = sigmoid(dot(dataMat[i],weights))
            #第i条数据的误差
            error = classLabels[i] - h
            #根据第i条数据更新权重
            weights = weights + alpha * error * dataMat[i]
        #print(k,weights, "*"*10 , dataMat[i], "*"*10 , error)
    return weights
        
#改进的梯度下降，alpha，随机选数据
def stocGradAscent1(dataMatrix,classLabels,numIter):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for k in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+k+i)+0.0001 
            randIndex = int(random.uniform(0,m))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
            
#决策函数,来一个新点或一系列新点
def predict(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: 
        return 1.0
    else: 
        return 0.0
    
#疝气病预测
def colicTest():
    frTrain = open('data/horseColicTraining.txt')
    frTest = open('data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    
    weights = stocGradAscent(array(trainingSet), trainingLabels,0.5,500)
    
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(predict(array(lineArr), weights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
     
def simpleTest():
    data,label = loadDataset('data/testSet.txt')
    #注意label是n行一列，是个二维列表,[[0]]
    #print(mat(label)[0])
    
    model = LogisticRegression()
    weights = model.fit(array(data),array(label))
    '''
    weights = stocGradAscent(data,label)
    '''
    #weights = stocGradAscent1(data,label,300)
    plotBestFit(data,label,weights)
    


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))) 
simpleTest()
#colicTest()
#multiTest()