# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

path="C:/Users/25012/Documents/ByteJump/algo_challenge_2_1/train/"
train_ground_truth = path+"train_ground_truth.csv"
train_action= path+"train_actions_info.csv"
action = pd.read_csv(train_action)
truth = pd.read_csv(train_ground_truth)

action.sort_values(by='user_id')
left=pd.DataFrame(action)
right=pd.DataFrame(truth)

result=pd.merge(left,right,on='user_id')
features=['action_id','offset_hour','pv']
x=result[features]
y=result['label']
from sklearn.model_selection import train_test_split  

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=17000000)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
import joblib
joblib.dump(clf,"action_tree_model.m")
y_pred=clf.predict(x_test)

from sklearn import metrics
from sklearn.model_selection import cross_val_score
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test   accuary: ", accuracy)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print("precison:%.2f, recall:%.2f, f1:%.2f" %( precision, recall, f1))
y_pred=clf.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Train    accuary: ", accuracy)
precision = metrics.precision_score(y_train, y_pred)
recall = metrics.recall_score(y_train, y_pred)
f1 = metrics.f1_score(y_train, y_pred)
print("precison:%.2f, recall:%.2f, f1:%.2f" %( precision, recall, f1))