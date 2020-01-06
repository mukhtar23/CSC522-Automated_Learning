import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

from sklearn import metrics
from sklearn.model_selection import GridSearchCV

cleveland = pd.read_csv('processed.filled.cleveland.csv')

# cleveland.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
#               'fbs', 'restecg', 'thalach', 'exang', 
#               'oldpeak', 'slope', 'ca', 'thal', 'target']

### 1 = male, 0 = female
# print(cleveland)
# print(cleveland.isnull().sum())

cleveland['num'] = cleveland.num.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
cleveland['sex'] = cleveland.sex.map({0: 'female', 1: 'male'})


#distribution of target vs age 
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = cleveland, x = 'age', hue = 'num', order = cleveland['age'].sort_values().unique())
plt.title('Variation of Age for each target class')
plt.show()

# distribution of target vs sex 
sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
sns.catplot(kind = 'count', data = cleveland, x = 'sex', hue = 'num', order = cleveland['sex'].sort_values().unique())
plt.title('Variation of Sex for each target class')
plt.show()

# # barplot of age vs sex with hue = target
# sns.catplot(kind = 'bar', data = cleveland, y = 'age', x = 'sex', hue = 'num')
# plt.title('Distribution of age vs sex with the target class')
# plt.show()

# # count of how many people have heart disease
# sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
# a = sns.countplot(data = cleveland, x = 'num', hue = 'num', order = cleveland['num'].sort_values().unique())
# plt.title('How many people have heart disease')
# plt.show()


cleveland['sex'] = cleveland.sex.map({'female': 0, 'male': 1})


################################## data preprocessing
X = cleveland.iloc[:, :-1].values
y = cleveland.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# creating hyperparameters to tune for the models
criterion = ['gini', 'entropy']
c = [ 0.1, 0.5, 1, 5, 10 ]
kernel = ['linear', 'poly', 'rbf','sigmoid' ]
gamma = ['auto', 0.0001, 0.001, 0.01, 1 ] 
degree = [ 1, 2, 3, 4, 5 ] 
coef0 = [ 0, 0.0001, 0.001, 0.01, 1 ]
penalty = ['l1', 'l2']
learning_rate = [ 0.05, 0.01, 0.3, 0.5 ]
max_depth = [ 3, 5, 7, 10 ]

#########################################   SVM   #############################################################
print("SVM")


# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

parameters = {'C':c, 'kernel': kernel, 'degree': degree,'coef0': coef0, 'gamma': gamma}

svr = SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# cm_test = confusion_matrix(y_test, y_pred)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)

#########################################   Logistic Regression  #############################################################
print("Logistic Regression")
# X = cleveland.iloc[:, :-1].values
# y = cleveland.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)

# Predicting the Test set results
# y_pred = classifier.predict(X_test)

parameters = {'C':c, 'penalty': penalty}

lr = LogisticRegression()
clf = GridSearchCV(lr, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm_test = confusion_matrix(y_test, y_pred)

# y_pred_train = classifier.predict(X_train)
# cm_train = confusion_matrix(y_pred_train, y_train)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)


#########################################   Decision Tree  #############################################################
print("Decision Tree")
# X = cleveland.iloc[:, :-1].values
# y = cleveland.iloc[:, -1].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# classifier = DecisionTreeClassifier()
# classifier.fit(X_train, y_train)

# Predicting the Test set results
# y_pred = classifier.predict(X_test)


parameters = {'criterion':criterion}

dc = DecisionTreeClassifier()
clf = GridSearchCV(dc, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)

# y_pred_train = classifier.predict(X_train)
# cm_train = confusion_matrix(y_pred_train, y_train)

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)

############################################################################### XGBoost #########################################################
print("XGBoost")
# applying XGBoost

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

xg = XGBClassifier()
# xg.fit(X_train, y_train)
# y_pred = xg.predict(X_test)

parameters = {'learning_rate':learning_rate, 'max_depth': max_depth}

clf = GridSearchCV(xg, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)

# y_pred_train = xg.predict(X_train)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 - Measure:", metrics.f1_score(y_test, y_pred))
print(clf.best_params_)
