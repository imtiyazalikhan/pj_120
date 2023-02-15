from cgi import test
from re import T
from tokenize import Special
import pandas as pd

col_names = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']

df = pd.read_csv("titanic.csv", names=col_names).iloc[1:]

print(df.head())

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']
X = df[features]
y = df.label

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from sklearn.external.six import stringIO
from IPython.display import Image
import pydotplus

dot_data = stringIO()

export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, Special_characcters=True, features_names=features, class_names=['0', '1'])

print(dot_data.grtvalue())