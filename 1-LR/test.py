# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1. 载入数据
data = []
label = []
fr = open('testSet.txt')
for line in fr.readlines():
    lineArr = line.strip().split()
    data.append([1.0,float(lineArr[0]),float(lineArr[1])])
    label.append(lineArr[2])
    
data_df = pd.DataFrame(data)
data_df[3] = label
#选择列
#data_df[1]


#2. 可视化

#可视化散点图，两种思路，
#一种思路是每次在画布上画出同一个label的sample，
#另外一种是每次画一个，在画的时候，判断颜色
plt.figure()

for index,row in data_df.iterrows():
    x = row[1]
    y = row[2]
    c = row[3]
    
    if(c == '1'):
        color = 'r'
        marker = 'x'
    else:
        color = 'b'
        marker = 'o'

    plt.scatter(x,y,c=color,marker=marker)

plt.show()

#3. 初始化




#4. 拟合



