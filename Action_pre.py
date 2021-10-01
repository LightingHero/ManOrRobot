# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

path1="C:/Users/25012/Documents/ByteJump/__MACOSX/"
model_path=path1+"action_tree_model.m"
ans_path=path1+'vali.csv'
path="C:/Users/25012/Documents/ByteJump/algo_challenge_2_1/train/"
train_ground_truth = path+"train_ground_truth.csv"
train_action= path+"train_actions_info.csv"
action = pd.read_csv(train_action)
truth = pd.read_csv(train_ground_truth)

action.sort_values(by='user_id')
result=pd.DataFrame(action)
features=['action_id','offset_hour','pv']
x=result[features]

import joblib
clf=joblib.load(model_path)
predict=clf.predict(x)
right=pd.DataFrame(predict)
right.columns=['label']
left=result['user_id']
ans=pd.merge(left,right,left_index=True, right_index=True)
ans.to_csv(ans_path)
print(ans)

num=pd.get_dummies(ans.label,prefix=None,columns=None,drop_first=False)
action=pd.concat([ans,num],axis=1)
action.columns=['user_id','label','man_action','machine_action']
action.drop('label',axis=1,inplace=True)

action=pd.DataFrame(action.groupby(action.user_id).sum())
action=action.T.drop_duplicates().T
action=pd.merge(action,truth,on='user_id')
action.to_csv(ans_path)
print(action)
