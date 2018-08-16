import pandas as pd 
from DecisionTree import DecisionTree


data = pd.read_csv('../dataset/mushrooms.csv')
labels = list(data.columns)[1:]
print(labels)

print(data[:10])

# 调整列顺序
df = data[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat','class']]

print(df[:10])
df.to_csv('data.csv',index=None,header=None)



dataset = df.values.tolist()

import math 
l = 0.8 * len(dataset)
l = math.ceil(l)
trainset = dataset[:l]
testset = dataset[l:]

rest_labels = list(range(0,len(labels)))


dt = DecisionTree()
myTree= dt.build_tree(trainset,rest_labels,labels)

#sample = dataset[0]
#print(sample)

'''
# 测试单个样本，预测类别
# ['x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u', 'p']
sample = ['x', 's', 'n', 't', 'n', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u', 'p']
pred = dt.predict(myTree,labels,sample)
print(labels)
print("sample=>{},pred=>{}".format(sample,pred))
'''
import json
print(json.dumps(myTree,indent=4))

c = zip( list(range(len(labels))),labels) 
print(list(c))

'''
sample = ['x', 's', 'n', 'f', 'n', 
		  'a', 'c', 'b', 'n', 'e', 
		  '?', 's', 's', 'o', 'o', 
		  'p', 'o', 'o', 'p', 'b', 
		  'v', 'l', 'e']
pred = dt.predict(myTree,labels,sample)
print("sample=>{},pred=>{}".format(sample,pred))
'''


pred_list = dt.predict_list(myTree,labels,testset)

from sklearn.metrics import accuracy_score

true_list = [example[-1] for example in testset]

acc = accuracy_score(y_true=true_list, y_pred=pred_list)
print(acc)





