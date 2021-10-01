# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

path1="C:/Users/25012/Documents/ByteJump/__MACOSX/"
ans_path=path1+'vali.csv'
path1="C:/Users/25012/Documents/ByteJump/__MACOSX/"
model_one_path=path1+"action_tree_model.m"
path="C:/Users/25012/Documents/ByteJump/algo_challenge_2_1/validation/"
model_path="C:/Users/25012/Documents/ByteJump/action_tree_secondmodel.m"
validation_action= path+"validation_actions_info.csv"
action = pd.read_csv(validation_action)

action.sort_values(by='user_id')
left=action['user_id']
result=pd.DataFrame(action)
features=['action_id','offset_hour','pv']
x=result[features]

import joblib
clf=joblib.load(model_one_path)
predict=clf.predict(x)

right=pd.DataFrame(predict)
right.columns=['label']
left=result['user_id']
ans=pd.merge(left,right,left_index=True, right_index=True)

num=pd.get_dummies(ans.label,prefix=None,columns=None,drop_first=False)
action=pd.concat([ans,num],axis=1)
action.columns=['user_id','label','man_action','machine_action']
action.drop('label',axis=1,inplace=True)

action=pd.DataFrame(action.groupby(action.user_id,as_index=False).sum())
action=action.T.drop_duplicates().T
print(action)

result=pd.DataFrame(action)
features=['man_action','machine_action']
x=result[features]
#结果
clf=joblib.load(model_path)
predict=clf.predict(x)
right=pd.DataFrame(predict)
right.columns=['label']
left=result['user_id']
ans=pd.merge(left,right,left_index=True, right_index=True)

ans.to_csv(ans_path)
print(ans)