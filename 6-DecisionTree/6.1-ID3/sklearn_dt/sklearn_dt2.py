import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

'''
https://blog.csdn.net/llh_1178/article/details/78516774
'''

data = pd.read_csv('../../dataset/mushrooms.csv')

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

y = data['class']
X = data.drop('class', axis=1)
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
'''
#columns = X_train.columns

import math
l = math.ceil(len(data))


X_train = X[:l]
X_test = X[l:]
y_train = y[:l]
y_test = y[l:]
print(X_train[:10])

'''
# 数据标准化
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
'''

from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)

y_prob = model_tree.predict_proba(X_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)
result = model_tree.score(X_test, y_pred)
print(result)
