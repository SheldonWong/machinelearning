# -*- coding: utf-8 -*-


import pandas as pd 
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

train_df = pd.read_csv('E:/workpace/jupyter/machine_learning/Kaggle/Titanic/input/train.csv')
test_df = pd.read_csv('E:/workpace/jupyter/machine_learning/Kaggle/Titanic/input/test.csv')

#  Return is NOT a Numpy-matrix, rather, a Numpy-array
# 属性查看，树数据探测，可视化，统计值等
print(train_df.columns)
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
# Name Sex Cabin Embarked 是字符串型，Cabin有缺失

# 处理缺失值
mean_age = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_age)

# 特征编码
lb = preprocessing.LabelBinarizer()
sex_encoder = lb.fit(train_df['Sex'])
train_df['Sex_encoded'] = sex_encoder.transform(train_df['Sex'])



# 特征选择
# features = train_df[['Age', 'Sex_encoded', 'Pclass']]
category = pd.Categorical(train_df['Embarked'])
train_df['Embarked'] = category.codes
features = train_df[['Age', 'Sex_encoded', 'Pclass','SibSp','Parch','Embarked']]



#标签
labels = train_df['Survived']

logreg = linear_model.LogisticRegression(C=1e2)
model = logreg.fit(features, labels)
score = model.score(features, labels)
print(score)



#预测
mean_age = test_df['Age'].mean()
test_df['Age'] = train_df['Age'].fillna(mean_age)
test_df['Sex_encoded'] = sex_encoder.transform(test_df['Sex'])

test_category = pd.Categorical(test_df['Embarked'])
test_df['Embarked'] = test_category.codes
test_features = test_df[['Age', 'Sex_encoded', 'Pclass','SibSp','Parch','Embarked']]
test_df_results = model.predict(test_features)

test_df['prediction'] = test_df_results


expected_results_df = pd.read_csv('E:/workpace/jupyter/machine_learning/Kaggle/Titanic/input/gender_submission.csv')
test_df['Survived'] = expected_results_df['Survived']


#保存结果
result_df = pd.DataFrame()
result_df['PassengerId'] = test_df['PassengerId']
result_df['Survived'] = test_df['Survived']
result_df.to_csv('result1.csv',index=None)


'''
X = trainset_df.as_matrix()
y = testset_df.as_matrix()

logreg = linear_model.LogisticRegression(C=1e5)

scores = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print(scores)
'''



