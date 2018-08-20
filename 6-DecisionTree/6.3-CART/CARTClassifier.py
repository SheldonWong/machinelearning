
from math import log
import operator


'''
文档快速生成注释的方法介绍,首先我们要用到__all__属性
在Py中使用为导出__all__中的所有类、函数、变量成员等
在模块使用__all__属性可避免相互引用时命名冲突
'''
__all__ = ['DecisionTree', 'unique']

class  DecisionTree(object):
    '''
    C4.5 决策树的实现
    '''
    
    def createDataSet2(self):
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing','flippers']
        return dataSet,labels

    def createDataSet(self):
        dataset = [[0, 0, 0, 0, 'N'], 
                   [0, 0, 0, 1, 'N'], 
                   [1, 0, 0, 0, 'Y'], 
                   [2, 1, 0, 0, 'Y'], 
                   [2, 2, 1, 0, 'Y'], 
                   [2, 2, 1, 1, 'N'], 
                   [1, 2, 1, 1, 'Y']]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
        return dataset, labels

    def unique(self,dataset):
        '''
        Desc:无论是最初未划分前的数据，还是划分后的数据，信息熵的计算都是根据最后一列计算的
        具体来说，依赖于最后一列的统计量，统计最后一列{取值：个数}
        Args:
        Returns:
        '''
        d = {}

        if(len(dataset) > 0 and isinstance(dataset[0],list)):
            column = [example[-1] for example in dataset]
        else:
            column = dataset
        for v in column:
            if(v not in d.keys()): d[v] = 0
            d[v] += 1
        return d

    # 根据类别统计计算gini指数，gini越大，不纯度越高，选择gini最小的作为切分点
    def gini(self,dataset):
        sample_num = len(dataset)
        result = 0.0
        for k,v in self.unique(dataset).items():
            prob = v / sample_num
            result += prob ** 2
        return 1 - result
    

    # 根据(feature_index，value)将数据集切分成两份
    def split_dataset(self,dataset, feature_index, value):
        list1 = []
        list2 = []
        if(isinstance(value,int) or isinstance(value,float)): #for int and float type
            for row in dataset:
                if (row[feature_index] >= value):list1.append(row)
                else:list2.append(row)
         
        return list1,list2


    # desc:
    # 维护一个未使用特征索引列表，
    # 对于未使用的特征，计算gini系数，选择最佳特征，加入到表示决策树的嵌套字典中
    # 参数1 要划分的数据集，参数2 未使用特征索引列表
    def choose_best_feature(self,dataset,rest_features_index):
        
        #当前节点数据集的信息熵（可能是根，也可能是叶子节点）
        base_gini = self.gini(dataset)
        
        # gini系数
        best_gini_gain = 0.0
        best_feature_index = -1# 初始化   
        
        rows_length = len(dataset)

        #对于未使用的特征，遍历计算信息增益率
        for feature_index in rest_features_index:
            #当前列对应的特征取值列表
            feature_value_list = [sample[feature_index] for sample in dataset]
            unique_val = set(feature_value_list)
            new_entropy = 0.0
            
            #计算条件熵，已知特征为feature_index
            for value in unique_val:
                #按照第feature_index的value列划分数据集
                left,right = self.split_dataset(dataset,feature_index,value)
                p = len(left) / rows_length
                new_gini = p * self.gini(left) + (1-p) * self.gini(right)
                new_gini_gain = base_gini - new_gini
            

            print('feature_index:{0},new_gini_gain:{1}'.format(feature_index,new_gini_gain))
            
            if(new_gini_gain > best_gini_gain):
                best_gini_gain = new_gini_gain
                best_feature_index = feature_index
                
            print('best_feature_index:{},best_gini_gain:{}'.format(best_feature_index,best_gini_gain))
        return best_feature_index

 
    #多数表决
    def majorityCnt(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    # 递归的构建决策树
    # args：dataset 数据，rest_labels 剩余的标签索引列表 labels 标签真实值，仅为了构建树使用
    def build_tree(self,dataset,rest_labels,labels):

        class_list = [sample[-1] for sample in dataset]

        # 为啥会有空的列表呢？
        if(len(class_list) == 0):
            return
        #终止条件
        # 类别完全相同则停止继续划分，返回类别
        if(class_list.count(class_list[0]) == len(class_list)):
            print('终止条件1') 
            return class_list[0]

        # 遍历完所有特征时返回出现次数最多的
        if len(rest_labels) == 0: 
            print('终止条件2') 
            return self.majorityCnt(class_list)
        
        print('rest_labels=>{0}'.format(rest_labels))
        
        ### 选择最佳特征，划分数据集
        best_feature_index = self.choose_best_feature(dataset,rest_labels)
        best_feature_label = labels[best_feature_index]
        myTree = {best_feature_label:{}}
        
        #在特征索引列表中，删除指定特征索引
        rest_labels.remove(best_feature_index)
        print('best_feature_index=>{0},rest_labels=>{1}'.format(best_feature_index,rest_labels))
        
        #递归调用
        featValues = [example[best_feature_index] for example in dataset]
        uniqueVals = set(featValues)
        for value in uniqueVals:        
            left,right = self.split_dataset(dataset, best_feature_index, value)    
            myTree[best_feature_label][value] = self.build_tree(left,rest_labels,labels)
            myTree[best_feature_label][value] = self.build_tree(right,rest_labels,labels)
        return myTree 

    # 预测单个样本的类别
    def predict(self,inputTree,featLabels,testVec):#根据已有的决策树，对给出的数据进行分类
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)#这里是将标签字符串转换成索引数字
        #处理某个特征取值不全的情况
        if(testVec[featIndex] not in secondDict.keys()):
            return 'e'
        for key in secondDict.keys(): 
            if(testVec[featIndex] == key):#如果key值等于给定的标签时
                if(type(secondDict[key]).__name__ == 'dict'):
                    classLabel = self.predict(secondDict[key],featLabels,testVec)#递归调用分类
                else: 
                    classLabel = secondDict[key]#此数据的分类结果
        return classLabel
                         
    # 预测样本列表的类别
    def predict_list(self,input_tree,feature_labels,test_vec_list):
        class_list = []
        for test_vec in test_vec_list:
            print(test_vec)
            c = self.predict(input_tree,feature_labels,test_vec)
            print(c)
            class_list.append(c)
        return class_list



    # 计算预测准确率          




dt = DecisionTree()
dataset,labels = dt.createDataSet2()
rest_labels = list(range(0,len(labels)))
myTree= dt.build_tree(dataset,rest_labels,labels)

print(myTree)
'''
import json
print(json.dumps(myTree,indent=4))


sample = [2, 2, 1, 0]
pred = dt.predict(myTree,labels,sample)
print("sample=>{},pred=>{}".format(sample,pred))


'''