## 机器学习

[Python Cookbook](https://python3-cookbook.readthedocs.io/zh_CN/latest/)

### 1. LR
### 2. Linear
### 3. KNN
### 4. NaiveBayesian
### 5. SVM
### 6. DecisionTree
- ID3
- C4.5
- CART
- GBDT
- RandomForest

### 7. NN 


### 基于Python的机器学习框架设计与实现

- 需要考虑的点


- 一些约定：
	- 问题划分，主要是数据，数据包含两个方面，一个是特征维度，一个是数据量，n*m,一般用n表示样本数量，m表示维度
- 概念设计，样本
- 功能模块划分
- 数据结构设计，状态设计
- 变量设计

- 如何选择模型，pipeline-模型搜索，gridsearch-参数搜索

- 输出模型的训练过程，要知道它是怎样一步一步得到最终结果的，日志与动态图像 -监控训练的健康程度

- 评价模型的指标
- 结果的可视化

- badcase的获取，便于分析为何预测出现错误或偏差，从而改进模型


- 大规模数据场景下的机器学习性能优化，SparkMlib，ParameterServer等
