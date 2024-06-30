
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

credit_risk = pd.read_csv("/content/drive/MyDrive/credit_risk_dataset.csv")
credit_risk.head()

from google.colab import drive
drive.mount('/content/drive')

credit_risk.describe()

credit_risk_copy = credit_risk.copy()

credit_risk.pivot_table(index='person_age',columns='loan_status',values='person_income',aggfunc='count').reset_index().sort_values(by='person_age',ascending=False)

cr_age_ = credit_risk[credit_risk['person_age']<=70]
cr_age_.reset_index(drop=True, inplace=True)
cr_age_.head()

cr_age_.head()

cr_age_.pivot_table(index='person_emp_length',columns='loan_status',values='person_income',aggfunc='count').reset_index().sort_values(by='person_emp_length',ascending=False)

person_emp_ = cr_age_[cr_age_['person_emp_length']<=47]
person_emp_.reset_index(drop=True, inplace=True)
person_emp_.head()

person_emp_.describe()

person_emp_.isnull().sum()

cr_data = person_emp_.copy()

cr_data.fillna({'loan_int_rate':cr_data['loan_int_rate'].median()},inplace=True)

cr_data.isnull().sum()

cr_data.describe()

cr_data.groupby('loan_status').count()['person_age']

6825/(6825+24846)

cr_data.head()

cr_data.groupby('person_home_ownership').count()['loan_intent']

cr_data.groupby('loan_intent').count()['person_home_ownership']

cr_data.head()

cr_data_copy=cr_data.drop('loan_grade',axis=1)

display(cr_data_copy.shape)
display(cr_data_copy.head())

cr_data_cat_treated = cr_data_copy.copy()

person_home_ownership = pd.get_dummies(cr_data_cat_treated['person_home_ownership'],drop_first=True).astype(int)
loan_intent = pd.get_dummies(cr_data_cat_treated['loan_intent'],drop_first=True).astype(int)
cr_data_cat_treated['cb_person_default_on_file_binary'] = np.where(cr_data_cat_treated['cb_person_default_on_file']=='Y',1,0)
cr_data_cat_treated.head()

person_home_ownership.head()

data_to_scale = cr_data_cat_treated.drop(['person_home_ownership','loan_intent','loan_status','cb_person_default_on_file','cb_person_default_on_file_binary'],axis=1)

data_to_scale.head()

scaler = StandardScaler()

data_to_scale.columns

scaled_data = scaler.fit_transform(data_to_scale)
scaled_df = pd.DataFrame(scaled_data,columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'])
scaled_df.head()

scaled_df.shape

scaled_data_combined = pd.concat([scaled_df,person_home_ownership,loan_intent],axis=1)
scaled_data_combined['cb_person_default_on_file'] = cr_data_cat_treated['cb_person_default_on_file_binary']
scaled_data_combined['loan_status'] = cr_data_cat_treated['loan_status']
scaled_data_combined.head()

scaled_data_combined.groupby('loan_status').count()['EDUCATION']

target = scaled_data_combined['loan_status']
features = scaled_data_combined.drop('loan_status',axis=1)
features.head()

## Data Balancing

from imblearn.over_sampling import SMOTE

smote = SMOTE()

balanced_features, balanced_target = smote.fit_resample(features,target)

balanced_target.shape

balanced_target_df = pd.DataFrame({'target':balanced_target})
balanced_target_df.groupby('target').size()

## Models Training

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(balanced_features, balanced_target,test_size=0.20,random_state=42)

logit = LogisticRegression()

logit.fit(x_train,y_train)

import pickle

with open('logisticPDmodel.pkl','wb') as file:
    pickle.dump(logit,file)

logit.score(x_train,y_train)

logit_prediction = logit.predict(x_test)

print(classification_report(y_test,logit_prediction))

print(logit.coef_[0])

features_imp_logit = pd.DataFrame({'features':balanced_features.columns,'logit_imp':logit.coef_[0]})

## RandomForest

rf = RandomForestClassifier()

rf.fit(x_train,y_train)

with open('RandomForesPDmodel.pkl','wb') as file:
    pickle.dump(rf,file)

rf.score(x_train,y_train)

rf_prediction = rf.predict(x_test)
rf_prediction

print(classification_report(y_test,rf_prediction))

rf.feature_importances_

features_imp_rf = pd.DataFrame({'features':balanced_features.columns,'rf_imp':rf.feature_importances_})

## XgBoost Model

xgb_model = XGBClassifier(tree_method = 'exact')

# model.fit(x,y.values.ravel())
xgb_model.fit(x_train,y_train.values.ravel())

with open('XGBpdModel.pkl','wb') as file:
    pickle.dump(xgb_model,file)

xgb_model.score(x_train,y_train.values.ravel())

xgb_prediction = xgb_model.predict(x_test)
xgb_prediction

print(classification_report(y_test,xgb_prediction))

features_imp_rf = pd.DataFrame({'features':balanced_features.columns,'rf_imp':rf.feature_importances_})
features_imp_xgb = pd.DataFrame({'features':balanced_features.columns,'xgb_imp':xgb_model.feature_importances_})
features_imp_xgb.sort_values(by='xgb_imp',ascending=False)

features_imp = pd.concat([features_imp_logit,features_imp_rf,features_imp_xgb],axis=1)
features_imp

xgb_prediction_df = pd.DataFrame({'test_indices_xgb':x_test.index,'xgb_pred':xgb_prediction})
rf_prediction_df = pd.DataFrame({'test_indices_rf':x_test.index,'rf_pred':rf_prediction})
logit_prediction_df = pd.DataFrame({'test_indices_logit':x_test.index,'logit_pred':logit_prediction})

xgb_prediction_df.head()

logit_prediction_df


merged_with_orig = cr_data_copy.merge(xgb_prediction_df,left_index=True,right_on='test_indices_xgb',how='left')
merged_with_orig.head()

merged_with_rf = merged_with_orig.merge(rf_prediction_df,left_index=True,right_on='test_indices_rf',how='left')
merged_with_rf.head()

merged_with_final = merged_with_rf.merge(logit_prediction_df,left_index=True,right_on='test_indices_logit',how='left')
merged_with_final.head()

merged_with_final.shape

merged_with_final.dropna(inplace=True)

final_data_with_pred = merged_with_final.drop(['test_indices_xgb','test_indices_rf','test_indices_logit'],axis=1)
final_data_with_pred.head()
