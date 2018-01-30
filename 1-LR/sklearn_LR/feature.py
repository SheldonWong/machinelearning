# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
from patsy import dmatrices #根据离散变量生成哑变量

train_df = pd.read_csv('E:/workpace/jupyter/machine_learning/Kaggle/Titanic/input/train.csv')

'''
#dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived
y, X = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = train_df, return_type='dataframe')
y = np.ravel(y)
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Name Sex Cabin Embarked 是字符串型，Cabin有缺失
'''

#ndarray
category = pd.Categorical(train_df['Embarked'])
print(type(category.codes))
print(category.codes)


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
'''
labelList = ['male','female','sex']
lb = preprocessing.LabelBinarizer()
encoder = lb.fit(labelList)
dummY=encoder.fit_transform(labelList)
print(dummY)
'''