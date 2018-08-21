## CART分类回归树的实现

### 参考
[CART实现参考](https://www.ibm.com/developerworks/cn/analytics/library/machine-learning-hands-on5-cart-tree/index.html)
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

### 关于过程的一些思考
```python
# 数据
        dataset = [[0, 0, 0, 0, 'N'], 
                   [0, 0, 0, 1, 'N'], 
                   [1, 0, 0, 0, 'Y'], 
                   [2, 1, 0, 0, 'Y'], 
                   [2, 2, 1, 0, 'Y'], 
                   [2, 2, 1, 1, 'N'], 
                   [1, 2, 1, 1, 'Y']]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
# 划分过程
best_feature_index=>0,rest_label_index=>[1, 2, 3]
[[0, 0, 0, 0, 'N'], [0, 0, 0, 1, 'N']] === [[1, 0, 0, 0, 'Y'], [2, 1, 0, 0, 'Y'], [2, 2, 1, 0, 'Y'], [2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']]

best_feature_index=>3,rest_label_index=>[1, 2]
[[1, 0, 0, 0, 'Y'], [2, 1, 0, 0, 'Y'], [2, 2, 1, 0, 'Y']] === [[2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']]

best_feature_index=>1,rest_label_index=>[2]
[[2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']] === []

best_feature_index=>2,rest_label_index=>[]
[[2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']] === []

{
    "outlook": {
        "<=0": "N",
        ">0": {
            "windy": {
                "<=0": "Y",
                ">0": {
                    "temperature": {
                        "<=2": {
                            "humidity": {
                                "<=1": "N",
                                ">1": null
                            }
                        },
                        ">2": null
                    }
                }
            }
        }
    }
}
```
- 问题：当数据划分到一定程度时，特征的取值完全一致，这种时候怎么处理，例如过程2到过程3？
划分数据集是依据特征的取值来划分的，所以当特征取值完全一样的时候，按照换分的标准，如果要进行划分，会把数据集划分成一个空集和一个全集，这样的划分是毫无意义的。所以当特征取值完全一样的时候，停止划分。也就是说当前的特征无法划分数据了已经。
