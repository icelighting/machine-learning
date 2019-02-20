import numpy as np
import pandas as pd
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


print(os.listdir("../Titanic/input"))

###读取数据
df_train = pd.read_csv('../Titanic/input/train.csv',sep=',')
df_test = pd.read_csv('../Titanic/input/test.csv',sep=',')
df_data = df_train.append(df_test) ##Entire Data

##对于乘客的ID进行整合
PassengerId = df_data["PassengerId"]
SubMission = pd.DataFrame()
SubMission['PassengerId'] = df_test['PassengerId']

###确定数据形状
a,b = np.shape(df_train)
c,d = np.shape(df_test)
print((a,b),(c,d))
##确定数据的列，对应特征
print(df_train.columns)
print(df_test.columns)

###确定数据的缺失信息
df_data.info()
print(pd.isnull(df_data).sum())

##表格化，statistical overview of the Data
print(df_train.describe())

print(df_test.describe())

'''
###数据的可视化


#Survival by Age ,Number of siblings(兄弟姐妹) and gender(性别)
grid = sns.FacetGrid(df_train,col="SibSp",row="Sex",hue="Survived",palette='seismic')#palette 调色板，颜色
grid = grid.map(plt.scatter,"PassengerId","Age")
#grid.add_legend()
#grid

#Parch 和性别 关于 生存下来的关系图
grid2 = sns.FacetGrid(df_train,col="Parch",row="Sex",hue="Survived",palette='seismic')
grid2 = grid2.map(plt.scatter,"PassengerId","Age")
#grid2.add_legend()
#grid2

#Survival by Age,Class and gender
grid3 = sns.FacetGrid(df_train,col="Pclass",row="Sex",hue="Survived",palette='seismic')
grid3 = grid3.map(plt.scatter,"PassengerId","Age")
#grid3.add_legend()
#grid3

#Survival by Age Emabarked and gender
grid4 = sns.FacetGrid(df_train,col="Embarked",row="Sex",hue="Survived",palette='seismic')
grid4 = grid4.map(plt.scatter,"PassengerId","Age")

plt.show()
'''
#g = sns.pairplot(df_train[[u'Survived',u'Pclass',u'Sex',u'Age',u'Parch',u'Fare',u'Embarked']],hue='Survived',palette='seismic',size=4,diag_kind='kde',diag_kws=dict(shade=True),plot_kws=dict(s=50))
#g.set(xticklabels = [])


Numeric_Columns = ['Pclass','Age','SibSp','Parch','Fare']

#create test and training data
data_to_train = df_train[Numeric_Columns].fillna(-1000)##fillna是pd的函数，可以将Nan值替换为括号里的数
y = df_train["Survived"]
x = data_to_train
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=21,stratify=y)

from sklearn.svm import LinearSVC

clf = SVC()
clf.fit(X_train,y_train)
linear_svc = LinearSVC()

print("Accuracy:{}".format(clf.score(X_test,y_test)))

#Create initial prediction
test = df_test[Numeric_Columns].fillna(-1000)
SubMission['Survived'] = clf.predict(test)

#Make first Submission
SubMission.set_index("PassengerId",inplace=True)
SubMission.to_csv('myFirstSubmission.csv',sep=',')

fig,(ax1,ax2) = plt.subplots(ncols=2,sharey=True,figsize=(12,6))
sns.boxplot(data=df_train,x="Pclass",y="Fare",ax=ax1)
plt.figure(1)
sns.boxplot(data=df_train,x="Embarked",y ="Fare",ax=ax2)
plt.show()

embarked = ['S','C','Q']
for port in embarked:
    fare_to_impute = df_data.groupby('Embarked')['Fare'].median()[embarked.index(port)]
    df_data.loc[(df_data['Fare'].isnull()) & (df_data['Embarked'] == port),'Fare'] = fare_to_impute

#Fare in df_train an df_test
df_train["Fare"] = df_data["Fare"][:891]
df_test["Fare"] = df_data["Fare"][891:]
print("Missing Fares Estimated")

##fill in missing Fare value in training set based on mean fare for that class
for x in range(len(df_train['Fare'])):
    if pd.isnull(df_train["Fare"][x]):
        pclass = df_train["Pclass"][x]
        df_train["Fare"][x] = round(df_train[df_train["Pclass"] == pclass]["Fare"].mean(),4)

    ##fill in missing Fare value in test set based on mean fare for that Pclass
for x in range(len(df_test["Fare"])):
    if pd.isnull(df_test["Fare"][x]):
        pclass = df_test["Pclass"][x]
        df_test["Fare"][x] = round(df_test[df_test["Pclass"] == pclass]["Fare"].mean(),4)###round 四舍五入取整函数 round(a,b)，a保留0后面b位，进行四舍五入取整


##map Fare values into groups of numerical values
df_data["FareBand"] = pd.qcut(df_data['Fare'],4,labels=[1,2,3,4]).astype('int')
df_train["FareBand"] = pd.qcut(df_train['Fare'],4,labels=[1,2,3,4]).astype('int')
df_test["Fareband"] = pd.qcut(df_test['Fare'],4,labels=[1,2,3,4]).astype('int')
df_train[["FareBand","Survived"]].groupby(["FareBand"], as_index=False).mean()
print('FareBand feature created')


#map each Embarked value to a numerical value
embarked_mapping = {"S":1,"C":2,"Q":3}
df_data["Embarked"] = df_data["Embarked"].map(embarked_mapping)
#split Embarked into df_train and df_test
df_train["Embarked"] = df_data["Embarked"][:891]
df_test["Embarkde"] = df_data["Embarked"][891:]
print("Embarked feature created")
df_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean()
print(df_data[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean())

#Estimate missing Embarkation Data
##Fill the na values in Embarked based on fareband data
fareband = [1,2,3,4]
for fare in fareband:
    embark_to_impute = df_data.groupby('FareBand')['Embarked'].median()[fare]
    df_data.loc[(df_data['Embarked'].isnull()) & (df_data['FareBand'] == fare),'Embarked'] = embark_to_impute

df_train["Embarked"] = df_data["Embarked"][:891]
df_test["Embarkde"] = df_data["Embarked"][891:]
print("Missing Embarkdation Estimated")







