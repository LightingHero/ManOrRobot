# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

path1="C:/Users/25012/Documents/ByteJump/__MACOSX/"
model_path=path1+"action_tree_second_model.m"
ans_path=path1+'ans.csv'
path="C:/Users/25012/Documents/ByteJump/algo_challenge_2_1/train/"
train_ground_truth = path+"train_ground_truth.csv"
train_action= path+"train_actions_info.csv"
action = pd.read_csv(ans_path)
action=pd.DataFrame(action)
features=['man_action','machine_action']
x=action[features]
y=action['label']
from sklearn.model_selection import train_test_split  

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1000)
print(x_train.shape)
print(x_test.shape)

 
from sklearn.tree import DecisionTreeClassifier
treeclf = DecisionTreeClassifier()
treeclf.fit(x_train, y_train)
y_pred=treeclf.predict(x_test)

import joblib
joblib.dump(treeclf,"action_tree_secondmodel.m")

from sklearn import metrics
from sklearn.model_selection import cross_val_score
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Test   accuary: ", accuracy)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print("precison:%.2f, recall:%.2f, f1:%.2f" %( precision, recall, f1))
y_pred=treeclf.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred)
print("Train    accuary: ", accuracy)
precision = metrics.precision_score(y_train, y_pred)
recall = metrics.recall_score(y_train, y_pred)
f1 = metrics.f1_score(y_train, y_pred)
print("precison:%.2f, recall:%.2f, f1:%.2f" %( precision, recall, f1))