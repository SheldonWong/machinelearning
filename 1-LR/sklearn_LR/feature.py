# -*- coding: utf-8 -*-

import pandas as pd 

train_df = pd.read_csv('E:/workpace/jupyter/machine_learning/Kaggle/Titanic/input/train.csv')

#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Name Sex Cabin Embarked 是字符串型，Cabin有缺失

'''
# 处理缺失值
mean_age = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_age)

# 特征编码
sex_encoder = preprocessing.LabelBinarizer()
sex_encoder = sex_encoder.fit(train_df['Sex'])
train_df['Sex_encoded'] = sex_encoder.transform(train_df['Sex'])

# 特征选择
# features = train_df[['Age', 'Sex_encoded', 'Pclass']]
features = train_df[['Age', 'Sex_encoded', 'Pclass','Fare','SibSp','Parch']]
'''

labelList = ['male','female','sex']
lb = preprocessing.LabelBinarizer()
dummY=lb.fit_transform(labelList)
print(dummY)