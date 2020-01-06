import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Read in Data
virginia = pd.read_csv('processed.filled.va.csv')
virginia.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
                    'fbs', 'restecg', 'thalach', 'exang',
                    'oldpeak', 'slope', 'num']

# Print data
# print(virginia)
# print(virginia.isnull().sum())

# Counting missing values --> not needed, missing data has been filled in
# virginia.isnull().sum()

# Remap for graphs
virginia['num'] = virginia.num.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})  # [1-4] = heart disease [0] = no heart disease
virginia['sex'] = virginia.sex.map({0: 'female', 1: 'male'})  # 1 = male, 0 = female

# Distribution of num vs age
sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=virginia, x='age', hue='num', order=virginia['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

# Distribution of num vs sex
sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=virginia, x='sex', hue='num', order=virginia['sex'].sort_values().unique())
plt.title('Variation of Sex for each target class')
plt.show()

# Barplot of age vs sex with hue = num
sns.catplot(kind='bar', data=virginia, y='age', x='sex', hue='num')
plt.title('Distribution of age vs sex with the target class')
plt.show()

# Count of how many people have heart disease
sns.set_context("paper", font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
a = sns.countplot(data=virginia, x='num', hue='num', order=virginia['num'].sort_values().unique())
plt.title('How many people have heart disease')
plt.show()

# ################################## data preprocessing & hyperparamters to tune for the models
# Before applying any prediction models to the data, we handled the missing
# values as described below.Then we split the data into training and test data
# using the train_ test_ split package and standardizedit with the StandardScaler
# package. We also created lists of hyperparameters to use for each package
# so that we could get the best possible results. The list of hyperparameters
# we created and what wenamed them are in the hyperparameter section.

# Remap for analysis
virginia['sex'] = virginia.sex.map({'female': 0, 'male': 1})  # 1 = male, 0 = female
X = virginia.iloc[:, :-1].values  # index x axis
y = virginia.iloc[:, -1].values  # index y axis

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale data
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# creating hyper-parameters to tune for the models
criterion = ['gini', 'entropy']
c = [0.1, 0.5, 1, 5, 10]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
gamma = ['auto', 0.0001, 0.001, 0.01, 1]
degree = [1, 2, 3, 4, 5]
coef0 = [0, 0.0001, 0.001, 0.01, 1]
penalty = ['l1', 'l2']
learning_rate = [0.05, 0.01, 0.3, 0.5]
max_depth = [3, 5, 7, 10]

# #########################################   SVM   #############################################################
print("-------------------- SVM --------------------")
parameters = {'C':c, 'kernel': kernel, 'degree': degree, 'coef0': coef0, 'gamma': gamma}

svr = SVC()
clf = GridSearchCV(svr, {})

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm_test = confusion_matrix(y_test, y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)
print()

# #########################################   Logistic Regression  #############################################################
print("-------------------- Logistic Regression --------------------")

parameters = {'C': c, 'penalty': penalty}

lr = LogisticRegression()
clf = GridSearchCV(lr, parameters)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm_test = confusion_matrix(y_test, y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)
print()

# #########################################   Decision Tree  #############################################################
print("-------------------- Decision Tree --------------------")

parameters = {'criterion': criterion}

dc = DecisionTreeClassifier()
clf = GridSearchCV(dc, {})
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)
print()

# #########################################   XGBoost #############################################################
print("-------------------- XGBoost --------------------")

xg = XGBClassifier()

parameters = {'learning_rate': learning_rate, 'max_depth': max_depth}

clf = GridSearchCV(xg, {})
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)
