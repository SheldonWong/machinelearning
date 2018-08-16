
import pandas as pd 
from sklearn import tree
from sklearn.model_selection import train_test_split


data = pd.read_csv('../../dataset/mushrooms.csv')

data_l = data.values.tolist()
X = [example[:-1] for example in data_l]
y = [example[-1] for example in data_l]

X_train, X_test, y_train, y_test = train_test_split(
										X, y, test_size=0.33, random_state=42)



clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
print(acc)
