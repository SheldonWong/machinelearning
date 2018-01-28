# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

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
        h = sigmoid(dot(dataMatrix,weights))
        error = labelMat-h
        weights = weights + alpha * dot(dataMatrix.T,error)
    return weights

data,label = loadDataset('testSet.txt')
print(gradAscent(data,label))