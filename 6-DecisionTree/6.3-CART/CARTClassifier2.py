from collections import Counter
import operator
class CARTClassifier():

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

    def create_dataset(self):
        pass

    # @desc:计数统计，传入一个列表，统计不同取值出现的次数。
    # @param:data,列表类型
    # @return:返回一个字典，形如{0:2,1:3}，表示0出现2次，1出现3次
    def unique(self,sample_set,feature_index=-1):
        # 统计最后一列的值
        result = Counter([sample[feature_index] for sample in sample_set])
        return result

    # @desc: 计算gini不纯度,gini系数越大，数据不纯度越高，不确定性越大
    # @param: 样本集 sample_set
    # @return: gini
    def gini(self,sample_set):
        sample_num = len(sample_set)
        res = 0.0
        for k,v in self.unique(sample_set).items():
            prob = v / sample_num
            res += prob ** 2
        gini = 1 - res
        return gini

    #@desc: 根据选择的最佳特征，以及最佳切分点，切分数据集,
    #       注意按照这个切分条件，遍历切分值得时候，需要去掉最大值，因为小于等于max_value就是整个集合
    #@param: sample_set
    #@return: 切分后的数据集
    def split_dataset(self, sample_set, feature_index, value):
        left = [sample for sample in sample_set if sample[feature_index] <= value]
        right = [sample for sample in sample_set if sample[feature_index] > value]
        return left,right

    #@desc: 选择最佳划分特征，
    #       大概的计算过程是，遍历所有特征的所有切分点，选择Gini最小的那个作为最佳特征
    #@param:
    #@return:
    def choose_best_feature(self,sample_set,rest_feature_index):
        best_gini = 1.0
        best_value = 0.0
        best_feature_index = -1# 初始化
        sample_number = len(sample_set)
        # 遍历特征
        for feature_index in rest_feature_index:
            #当前列对应的特征取值列表
            feature_value_list = [sample[feature_index] for sample in sample_set]
            unique_val = set(feature_value_list)
            if(len(unique_val) > 1):
                unique_val.remove(max(unique_val))



            # 遍历特征的每个切分点
            for value in unique_val:
                #按照第feature_index的value列划分数据集
                left,right = self.split_dataset(sample_set,feature_index,value)
                p = len(left) / sample_number
                new_gini = p * self.gini(left) + (1-p) * self.gini(right)

                
                if(new_gini < best_gini):
                    best_gini = new_gini
                    best_feature_index = feature_index
                    best_value = value
                # 取值级别，与当前特征的取值个数一致
                print('feature_index=>{0},split value=>{1},new_gini=>{2},best_gini=>{3}'.format(feature_index,value,new_gini,best_gini))
            
            # 特征级别，与特征的size一致
            print('best_feature_index=>{},best_gini=>{}'.format(best_feature_index,best_gini))                
        return best_feature_index,best_value    

    #多数表决
    def majorityCnt(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    # 递归的构建决策树
    # @param:
    # @return: 
    def build_tree(self,sample_set,rest_label_index,labels):
        class_list = [sample[-1] for sample in sample_set]
        #终止条件
        # 特征值完全一样，也要返回
        # 类别完全相同则停止继续划分，返回类别
        if class_list.count(class_list[0]) == len(class_list):
            print('终止条件1') 
            return class_list[0]
        # 特征用完,返回出现次数最多的
        if len(rest_label_index) == 0: 
            print('终止条件2') 
            return self.majorityCnt(class_list)

        # 选择最佳特征，构建决策树
        best_feature_index,best_value = self.choose_best_feature(sample_set,rest_label_index)
        best_feature_label = labels[best_feature_index]
        myTree = {best_feature_label:{}}

        #在特征列表中，删除指定特征索引
        rest_label_index.remove(best_feature_index)
        print('best_feature_index=>{0},rest_label_index=>{1}'.format(best_feature_index,rest_label_index))

        #递归调用
        l_name = '<='+ str(best_value)
        r_name = '>' + str(best_value)

        left,right = self.split_dataset(sample_set, best_feature_index, best_value)
        print(left,'===',right)
        if(len(left) > 1):
            myTree[best_feature_label][l_name] = self.build_tree(left,rest_label_index,labels)
        else:
            myTree[best_feature_label][l_name] = None
        
        if(len(right) > 1):
            myTree[best_feature_label][r_name] = self.build_tree(right,rest_label_index,labels)
        else:
            myTree[best_feature_label][r_name] = None
        
        return myTree 

    def predict():
        pass

    def score():
        pass

    def get_badcase():
        pass

cart = CARTClassifier()
'''
# 0.5
print(cart.gini([1,1,2,2]))
# 0.67
print(cart.gini([1,1,2,2,3,3]))
# 0.75
print(cart.gini([1,1,2,2,3,3,4,4]))
'''
# print('gini:',cart.gini([[0, 0, 0, 0, 'N'], [0, 0, 0, 1, 'N']]))
data,labels = cart.createDataSet()
'''
left,right = cart.split_dataset(data,0,0)
print(left)
print(right)
'''



#data = [[2, 2, 1, 1, 'N'], [1, 2, 1, 1, 'Y']]
#print(cart.choose_best_feature(data,[1,2]))



rest_labels = list(range(0,len(labels)))
myTree= cart.build_tree(data,rest_labels,labels)

import json

print(json.dumps(myTree,indent=4))
