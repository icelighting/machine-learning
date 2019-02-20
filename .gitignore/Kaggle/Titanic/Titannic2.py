import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import (AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier,VotingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LogisticRegression, PassiveAggressiveClassifier,RidgeClassifierCV
from sklearn.metrics import accuracy_score,auc,classification_report,confusion_matrix,mean_squared_error, precision_score, recall_score,roc_curve
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate,train_test_split,GridSearchCV,KFold,learning_curve,RandomizedSearchCV,StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv(r'D:\data analysis\Kaggle\Titanic\output\Data_to_train.csv')
print(data.head())
test = pd.read_csv(r'D:\data analysis\Kaggle\Titanic\output\test.csv')
print(test.head())
df_train = pd.read_csv(r'D:\data analysis\Kaggle\Titanic\input\train.csv')
df_test = pd.read_csv(r'D:\data analysis\Kaggle\Titanic\input\test.csv')

LGBM = GradientBoostingClassifier()

X_train,X_test,y_train,y_test = train_test_split(data,df_train['Survived'],test_size=0.3,random_state=21, stratify=df_train['Survived'])
LGBM.fit(X_train,y_train)
Submission = pd.DataFrame()
Submission['PassengerId'] = df_test['PassengerId']
Submission['Survived'] = LGBM.predict(test)
Submission.to_csv(r'D:\data analysis\Kaggle\Titanic\output\LGBM.csv')
