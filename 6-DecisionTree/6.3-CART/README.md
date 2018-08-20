## CART分类回归树的实现

### 参考
[参考1](https://zhuanlan.zhihu.com/p/32164933)
[决策树之CART算法原理及python实现](https://blog.csdn.net/LY_ysys629/article/details/72809129)
[How To Implement The Decision Tree Algorithm From Scratch In Python](https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/)
[unique统计](https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray-in-python)



### 流程


### 函数
- create_dataset()
- unique()
统计dataset的某一列各种不同取值的数目，输出形如{A:3,B:2,C:5}
- gini()
根据公式计算gini系数
- split_dataset()
根据col,value切分数据
- choose_best_feature()
遍历feature，寻找当前最佳的切分feature
- build_tree
递归的构建决策树
- predict()
遍历决策树，找到相应的叶子节点，给出sample的预测值
- score()
传入pred_y,test_y，计算误差
- get_badcase()
给出badcase的feature，test_y,pred_y
