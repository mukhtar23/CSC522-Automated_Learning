import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import   StandardScaler as standardize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import argparse
np.random.seed(30)


def plot_summary_statistics(data_frame):
	sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
	sns.catplot(kind = 'count', data = data_frame, x = 'age', hue = 'num', order = data_frame['age'].sort_values().unique())
	plt.title('Variation of Age for each target class')
	plt.legend()
	plt.show()

	sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
	sns.catplot(kind = 'count', data = data_frame, x = 'sex', hue = 'num', order = data_frame['sex'].sort_values().unique())
	plt.title('Variation of Sex for each target class')
	plt.show()

	sns.catplot(kind = 'bar', data = data_frame, y = 'age', x = 'sex', hue = 'num')
	plt.title('Distribution of age vs sex with the target class')
	plt.show()

	# count of how many people have heart disease
	sns.set_context("paper", font_scale = 2, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20}) 
	a = sns.countplot(data = data_frame, x = 'num', hue = 'num', order = data_frame['num'].sort_values().unique())
	plt.title('How many people have heart disease')
	plt.show()
	# File paths
def main(plot):
	hungary_csv = 'processed.filled.hungarian.csv'
	swiss_csv = 'processed.filled.switzerland.csv'

	df_hungary = pd.read_csv(hungary_csv)
	df_hungary.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'num']

	df_swiss = pd.read_csv(swiss_csv)
	df_swiss.columns = ['age', 'sex', 'cp',  'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'num']

	X_hungary = df_hungary.iloc[:, :-1].values
	Y_hungary = df_hungary.iloc[:, -1].values

	X_swiss = df_swiss.iloc[:, :-1].values
	Y_swiss = df_swiss.iloc[:, -1].values

	# Map swiss dataset labels to binary
	df_swiss['num'] = df_swiss.num.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
	df_swiss['sex'] = df_swiss.sex.map({0: 'female', 1: 'male'})

	df_hungary['sex'] = df_hungary.sex.map({0:'female', 1:'male'})

	## split into training and testing
	### Create plots for summary statistics of the dataset
	if(plot):
		plot_summary_statistics(df_hungary)
		plot_summary_statistics(df_swiss)

	x_train_hung, x_test_hung, y_train_hung, y_test_hung = train_test_split(X_hungary, Y_hungary, test_size=0.2, random_state=0, stratify=Y_hungary)
	x_train_swiss, x_test_swiss, y_train_swiss, y_test_swiss = train_test_split(X_swiss, Y_swiss, test_size=0.2, random_state=0, stratify=Y_swiss)

	## Standardize data
	standardize_scaler = standardize()
	x_train_hung = standardize_scaler.fit_transform(x_train_hung)
	x_test_hung = standardize_scaler.transform(x_test_hung)

	x_train_swiss = standardize_scaler.fit_transform(x_train_swiss)
	x_test_swiss = standardize_scaler.transform(x_test_swiss)

	criterion = ['gini', 'entropy']
	c = [ 0.1, 0.5, 1, 5, 10 ]
	kernel = ('linear', 'poly', 'rbf','sigmoid' )
	gamma = ('auto', 0.0001, 0.001, 0.01, 1, 'scale')
	degree = ( 1, 2, 3, 4, 5 )
	coef0 = ( 0, 0.0001, 0.001, 0.01, 1 )
	penalty = ['l1', 'l2']
	learning_rate = [ 0.05, 0.01, 0.3, 0.5 ]
	max_depth = [ 3, 5, 7, 10 ]

	print("Performing gridsearch for optimal SVM hyperparameters...")

	parameters = {'C':c, 'kernel': kernel, 'degree': degree,'coef0': coef0, 'gamma': gamma}


	## Train and Predict with SVM

	init_svm = SVC()
	hung_svm_clf = GridSearchCV(init_svm, parameters, cv=3, iid=True)
	hung_svm_clf = hung_svm_clf.fit(x_train_hung, y_train_hung)
	hung_svm_test_pred = hung_svm_clf.predict(x_test_hung)
	print(hung_svm_clf.best_estimator_)
	print("Testing accuracy of SVM over Hungary data: " + str(accuracy_score(y_test_hung, hung_svm_test_pred)))
	print("Testing precision of SVM over Hungary data: " + str(metrics.precision_score(y_test_hung, hung_svm_test_pred)))
	print("Testing recall of SVM over Hungary data: " + str(metrics.recall_score(y_test_hung, hung_svm_test_pred)))
	print("Testing F-measure of SVM over Hungary data: " + str(metrics.f1_score(y_test_hung, hung_svm_test_pred)))
	print()
	swiss_svm = SVC()
	swiss_svm_clf = GridSearchCV(swiss_svm, parameters, cv=3, iid=True)
	swiss_svm_clf = swiss_svm_clf.fit(x_train_swiss, y_train_swiss)
	swiss_svm_predictions = swiss_svm_clf.predict(x_test_swiss)
	print("Testing accuracy of SVM over Swiss data: " + str(accuracy_score(y_test_swiss, swiss_svm_predictions)))
	print("Testing precision of SVM over Swiss data: " + str(metrics.precision_score(y_test_swiss, swiss_svm_predictions)))
	print("Testing recall of SVM over Swiss data: " + str(metrics.recall_score(y_test_swiss, swiss_svm_predictions)))
	print("Testing F-measure of SVM over Swiss data: " + str(metrics.f1_score(y_test_swiss, swiss_svm_predictions)))
	print()
	# Train and predict with Logistic Regressin
	print("Performing gridsearch for optimal Logistic Regression hyperparameters...")

	parameters = {'C': c, 'penalty': penalty}
	lr_hung = LogisticRegression(solver='liblinear')
	lr_swiss = LogisticRegression(solver='liblinear')
	lr_hung_clf = GridSearchCV(lr_hung, parameters, cv=3, iid=True)
	# Fit hungary model
	lr_hung_clf.fit(x_train_hung, y_train_hung)
	predictions_lr_hung = lr_hung_clf.predict(x_test_hung)
	print(lr_hung_clf.best_estimator_)

	print("Testing accuracy of Logistic Regression over Hungary data: " + str(accuracy_score(y_test_hung, predictions_lr_hung)))
	print("Testing precision of LR over Hungary data: " + str(metrics.precision_score(y_test_hung, predictions_lr_hung)))
	print("Testing recall of LR over Hungary data: " + str(metrics.recall_score(y_test_hung, predictions_lr_hung)))
	print("Testing F-measure of LR over Hungary data: " + str(metrics.f1_score(y_test_hung, predictions_lr_hung)))
	print()
	lr_swiss_clf = GridSearchCV(lr_swiss, parameters, cv=3, iid=True)
	#Fit swiss model
	lr_swiss_clf.fit(x_train_swiss, y_train_swiss)
	predictions_lr_swiss = lr_swiss_clf.predict(x_test_swiss)
	print("Testing accuracy of Logistic Regression over Swiss data: " + str(accuracy_score(y_test_swiss, predictions_lr_swiss)))
	print("Testing precision of Logistic Regression over Swiss data: " + str(metrics.precision_score(y_test_swiss, predictions_lr_swiss)))
	print("Testing recall  of Logistic Regression over Swiss data: " + str(metrics.recall_score(y_test_swiss, predictions_lr_swiss)))
	print("Testing F-measure  of Logistic Regression over Swiss data: " + str(metrics.f1_score(y_test_swiss, predictions_lr_swiss)))
	print()
	print("Performing gridsearch for optimal Decision Tree hyperparameters...")
	# Train and predict with Decision Tree
	parameters = {'criterion': criterion}
	hung_dt = DecisionTreeClassifier()
	hung_dt_clf = GridSearchCV(hung_dt, parameters, cv=3, iid=True)
	# Fit model
	hung_dt_clf.fit(x_train_hung, y_train_hung)
	prediction_hung_dt = hung_dt_clf.predict(x_test_hung)
	print(hung_dt_clf.best_estimator_)

	print("Testing accuracy of Decision Tree over Hungary data: " + str(accuracy_score(y_test_hung, prediction_hung_dt)))
	print("Testing precision of Decision Tree over Hungary data: " + str(metrics.precision_score(y_test_hung, prediction_hung_dt)))
	print("Testing recall of Decision Tree over Hungary data: " + str(metrics.recall_score(y_test_hung, prediction_hung_dt)))
	print("Testing F-measure of Decision Tree over Hungary data: " + str(metrics.f1_score(y_test_hung, prediction_hung_dt)))
	print()
	swiss_dt = DecisionTreeClassifier()
	swiss_dt_clf = GridSearchCV(swiss_dt, parameters, cv=3, iid=True)
	#Fit Model
	swiss_dt_clf.fit(x_train_swiss, y_train_swiss)
	prediction_swiss_dt = swiss_dt_clf.predict(x_test_swiss)
	print("Testing accuracy of Decision Tree over Swiss data: " + str(accuracy_score(y_test_swiss, prediction_swiss_dt)))
	print("Testing precision of Decision Tree over Swiss data: " + str(metrics.precision_score(y_test_swiss, prediction_swiss_dt)))
	print("Testing recall of Decision Tree over Swiss data: " + str(metrics.recall_score(y_test_swiss, prediction_swiss_dt)))
	print("Testing F-measure of Decision Tree over Swiss data: " + str(metrics.f1_score(y_test_swiss, prediction_swiss_dt)))
	print()
	print("Performing gridsearch for optimal XGBoost hyperparameters...")
	## Train and predict with XGBoost
	parameters = {'learning_rate': learning_rate, 'max_depth': max_depth}

	hung_xg = XGBClassifier()

	hung_xg_clf = GridSearchCV(hung_xg, parameters, cv=3, iid=True)
	hung_xg_clf.fit(x_train_hung, y_train_hung)
	predictions_hung_xg = hung_xg_clf.predict(x_test_hung)
	print(hung_xg_clf.best_estimator_)

	print("Testing accuracy of XGBoost over Hungary Data: " + str(accuracy_score(y_test_hung, predictions_hung_xg)))
	print("Testing precision of XGBoost over Hungary Data: " + str(metrics.precision_score(y_test_hung, predictions_hung_xg)))
	print("Testing recall of XGBoost over Hungary Data: " + str(metrics.recall_score(y_test_hung, predictions_hung_xg)))
	print("Testing F-measure of XGBoost over Hungary Data: " + str(metrics.f1_score(y_test_hung, predictions_hung_xg)))
	print()
	swiss_xg = XGBClassifier()
	swiss_xg_clf = GridSearchCV(swiss_xg, parameters, cv=3, iid=True)
	swiss_xg_clf.fit(x_train_swiss, y_train_swiss)
	predictions_swiss_xg = swiss_xg_clf.predict(x_test_swiss)
	print("Testing accuracy of XGBoost over Swiss Data: " + str(accuracy_score(y_test_swiss, predictions_swiss_xg)))
	print("Testing precision of XGBoost over Swiss Data: " + str(metrics.precision_score(y_test_swiss, predictions_swiss_xg)))
	print("Testing recall of XGBoost over Swiss Data: " + str(metrics.recall_score(y_test_swiss, predictions_swiss_xg)))
	print("Testing F-measure of XGBoost over Swiss Data: " + str(metrics.f1_score(y_test_swiss, predictions_swiss_xg)))


def combined_data():
	return


if __name__ == "__main__":
	plot = False
	parser = argparse.ArgumentParser()
	parser.add_argument("--plot")
	parser.add_argument("--combine")
	args = parser.parse_args()
	if args.plot:
		print("Generating Plots")
		plot = True
		main(plot)
	# elif args.combine:
	# 	print("Performing ML exploration over Hungary + Swiss Set")
	# 	combined_data()
	else:
		main(plot)
