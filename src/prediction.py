import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from datetime import date, datetime
import statistics

df = pd.read_csv("new_inputs/lc_testset.csv")

required_features = ['term','int_rate','installment','grade','emp_length',
                     'annual_inc','issue_d','dti','earliest_cr_line',
                     'open_acc','revol_util','mort_acc','address','loan_amnt'

]

prediction_features = ['term', 'grade', 'emp_length', 
                       'int_rate', 'revol_util', 'mort_acc',
                       'loan_amnt', 'ID_delta', 'dti',
                       'annual_inc', 'installment', 'ECL_delta',
                       'open_acc','00813',
                       '05113', '11650', '22690', '29597',
                       '30723', '48052', '70466', '86630', '93700']

#Validate whether required features are found

errors = []

for features in required_features:
    if features not in df.columns.tolist():
        errors.append(f'{features} not found in request.')
        
if errors:
    print(errors)
    raise SystemExit("Missing required features, terminating script.")


#If no errors, proceed to process features
    
    # Start with 'emp_length'

    #Convert all missing value in emp_length to < 1 year. Based on data description, null/0 values = < 1 year

df['emp_length'] = df['emp_length'].fillna("0")
df['emp_length'] = df['emp_length'].str.replace('< 1 year','0')
df['emp_length'] = df['emp_length'].str.replace('10\+ years','10')
df['emp_length'] = df['emp_length'].str.replace(' years','').str.replace(' year','')


    #Moving on to getting ID_delta and ECL_delta

#Start with issue date (issue_d)
df[['ID_Month','ID_Year']] = df['issue_d'].str.split('-', expand =True)

#Drop issue_d column now that we no longer need it
df = df.drop('issue_d', axis = 1)

#Followed by earliest cr_line
df[['ECL_Month','ECL_Year']] = df['earliest_cr_line'].str.split('-', expand =True)

#Drop earliest_cr_line column now that we no longer need it
df = df.drop('earliest_cr_line', axis = 1)

#Convert months into numerical values
df['ECL_Month'] = df['ECL_Month'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}) 
df['ID_Month'] = df['ID_Month'].replace({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}) 

#Convert ECL/ID month and year into date
df['ECL_Date'] = pd.to_datetime(df['ECL_Month']+ "-" + df['ECL_Year'], format = '%m-%Y')
df['ID_Date'] = pd.to_datetime(df['ID_Month']+ "-" + df['ID_Year'], format = '%m-%Y')

#Get Date Delta between ECL/ID and today to use the delta as a feature
df['ECL_delta'] = (pd.Timestamp('today').normalize() - df['ECL_Date']).dt.days
df['ID_delta'] = (pd.Timestamp('today').normalize() - df['ID_Date']).dt.days


    #Moving on to getting zipcodes
df['zipcode'] = df['address'].str.split().str[-1]
df = df.drop(columns=["address"])

#Convert zipcode Dtype to float64
df['zipcode'] = df['zipcode'].astype('object')

#Label Encode Categorical Features

label_encoder = LabelEncoder()

#Label Encode term, grade
df['term'] = label_encoder.fit_transform(df['term']) #label encode 60 months = 1, 36 months = 0
df['grade'] = label_encoder.fit_transform(df['grade']) # A = 0, B = 1, etc. 
df['emp_length'] = df['emp_length'].astype(int)

#One-Hot encode zipcode
zipcode_OH_encoding = pd.get_dummies(df['zipcode'])
df = pd.concat([df,zipcode_OH_encoding], axis = 1)


