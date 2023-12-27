#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 16:10:22 2023

@author: s190387
"""
import pandas as pd
import os
import numpy as np
import sklearn

path_data = '/Users/s190387/Desktop/golf/data/'
os.chdir(path_data)
pga_data = pd.read_csv('pgaTourData 2.csv')

## DATA PREPROCESSING 
pga_data_filled = pga_data.copy()

# Drop any datapoints with at least 5 columns of missing data
pga_data_filled = pga_data_filled.dropna(thresh=5)
n_rows_before = pga_data.shape[0]
n_rows_after = pga_data_filled.shape[0]
print('Number of rows before dropping rows with missing values: ', n_rows_before)
print('Number of rows after dropping rows with missing values: ', n_rows_after)

# Unique golfers in cleaned dataset
#unique_golfers = pga_data_filled['Player Name'].nunique()
#years_golfer = pga_data_filled.groupby('Player Name')['Year'].count().sort_values(ascending=False)
#summary_years = years_golfer.value_counts().sort_index(ascending=False)
#summary_df= pd.DataFrame(summary_years).reset_index()
#summary_df = ['Years of Data', 'Number of Golfers']
#summary_df= summary_df.sort_values(by='Years of Data', ascending=False)
#unique_golfers, summary_df
#print(unique_golfers)
#print(summary_df)


# Handle Missing Values with 0
pga_data_filled['Wins'] = pga_data_filled['Wins'].fillna(0)
pga_data_filled['Top 10'] = pga_data_filled['Top 10'].fillna(0)


# Convert column Points to numerical formats
pga_data_filled['Points'] = pd.to_numeric(pga_data_filled['Points'].str.replace(',', ''), errors='coerce')

# Clean up Money column
pga_data_filled['Money'] = pga_data_filled['Money'].astype(str)
pga_data_filled['Money'] = pga_data_filled['Money'].replace('[\$,]', '', regex=True)
pga_data_filled['Money'] = pga_data_filled['Money'].astype(float)
pga_data_filled['Money'] = pga_data_filled['Money'].fillna(0)

# Collinearity matrix between features
pga_data_corr = pga_data_filled.drop(columns=['Player Name', 'Year'])
corr_matrix = pga_data_corr.corr()

# Heatmap of collinearity matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True)
# plt.show()

# Features correlation matrix
pga_data_corr_features = pga_data_filled[['Fairway Percentage', 'Rounds', 'Avg Distance', 'gir', 'Average Putts', 'Average Scrambling', 'Average SG Putts', 'Average SG Total', 'SG:OTT', 'SG:APR', 'SG:ARG']]
corr_matrix_features = pga_data_corr_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_features, annot=True)
# plt.show()

# Due to high collinearity, drop Average SG Total feature
pga_data_filled = pga_data_filled.drop(columns=['Average SG Total'])

# Outcomes correlation matrix
pga_data_corr_outcomes = pga_data_filled[['Average Score','Points', 'Wins', 'Top 10', 'Money']]
corr_matrix_outcomes = pga_data_corr_outcomes.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_outcomes, annot=True)
# plt.show()

# print money column
#print(pga_data_filled['Money'])

# Outcome Varaibles to remove
outcome_variables = ['Points', 'Top 10', 'Wins','Money']
pga_data_filled = pga_data_filled.drop(columns=outcome_variables)


# Splitting data into Train and Test dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

X = pga_data_filled.drop(columns = ['Year', 'Average Score', 'Player Name'])
y = pga_data_filled['Average Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop empty rows in all 4 datasets
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train.dropna()
y_test = y_test.dropna()

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Download X_train_scaled and X_test_scaled as csv
#X_train_scaled = pd.DataFrame(X_train_scaled)
#X_test_scaled = pd.DataFrame(X_test_scaled)
#X_train_scaled.to_csv('X_train_scaled.csv')
#X_test_scaled.to_csv('X_test_scaled.csv')

#shapes of the training and testing sets
print('X_train: ', X_train_scaled.shape)    
print('X_test: ', X_test_scaled.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)
