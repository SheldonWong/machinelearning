# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadDataset(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    m,n = shape(dataMatrix)
    alpha = 0.001
    iter = 500
    weights = ones((n,1))
    for k in range(iter):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

data,label = loadDataset('testSet.txt')
print(gradAscent(data,label))