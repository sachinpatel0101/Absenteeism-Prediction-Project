#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer

# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
    
    def __init__(self, model_file, scaler_file):
        # read the 'model' and 'scaler' files which were saved
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    # take a data file (*.csv) and preprocess it
    def load_and_clean_data(self, data_file):
        
        # import the data
        df = pd.read_csv(data_file,delimiter=',')
        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        # drop the 'ID' column
        df.drop(columns=['ID'], inplace= True)
        # to preserve the code we've created in the previous section, we will add a column with 'NaN' strings
        df['Absenteeism Time in Hours'] = 'NaN'
        # Transforming Reason for Absence column
        One_hot_encode_reason = pd.get_dummies(df['Reason for Absence'], drop_first= True)
        
        reason_type_1 = One_hot_encode_reason.loc[:,1:14].max(axis= 1)
        reason_type_2 = One_hot_encode_reason.loc[:,15:17].max(axis= 1)
        reason_type_3 = One_hot_encode_reason.loc[:,18:21].max(axis= 1)
        reason_type_4 = One_hot_encode_reason.loc[:,22:].max(axis= 1)
        
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis= 1)
        
        df.columns = ['Reason for Absence', 'Date', 'Transportation Expense',
                       'Distance to Work', 'Age', 'Daily Work Load Average',
                       'Body Mass Index', 'Education', 'Children', 'Pets',
                       'Absenteeism Time in Hours', 'reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4']
        
        df.drop(columns=['Reason for Absence'], inplace= True)
        
        df.columns = ['Date', 'Transportation Expense',
                       'Distance to Work', 'Age', 'Daily Work Load Average',
                       'Body Mass Index', 'Education', 'Children', 'Pets',
                       'Absenteeism Time in Hours', 'reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4']
        
        reordered_col_names = ['reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4','Date', 'Transportation Expense',
                               'Distance to Work', 'Age', 'Daily Work Load Average',
                               'Body Mass Index', 'Education', 'Children', 'Pets',
                               'Absenteeism Time in Hours']
        df_reason_mod = df[reordered_col_names]
        
        df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format= '%d/%m/%Y')
        
        df_reason_mod['month_value'] = df_reason_mod['Date'].dt.month
        df_reason_mod['weekday'] = df_reason_mod['Date'].dt.weekday
        
        reset_col_order = ['reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4',
                           'Date','month_value', 'weekday', 'Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education',
                           'Children', 'Pets', 'Absenteeism Time in Hours']
        df_reason_mod = df_reason_mod[reset_col_order]
        
        df_reason_mod.drop(columns=['Date'], inplace= True)
        
        df_reason_mod['Education'] = np.where(df_reason_mod['Education'] == 1,0,1)
        
        df = df_reason_mod
        # replace the NaN values
        df = df.fillna(value=0)
        
        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        
        # drop the variables we decide we don't need
        df = df.drop(['weekday','Distance to Work','Daily Work Load Average'],axis=1)
        
        self.preprocessed_data = df.copy()
        
        # columns_to_scale = [4,5,6,7,9,10]
        
        self.data = self.scaler.transform(df)
        
        columns_ORDER_after_scale = ['month_value', 'Transportation Expense','Age','Body Mass Index','Children','Pets', 
                                     'reason_type_1', 'reason_type_2', 'reason_type_3', 'reason_type_4', 'Education']
        
    # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
        
    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
    # predict the outputs and the probabilities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data


# In[ ]:




