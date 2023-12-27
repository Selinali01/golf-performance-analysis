# Read in csv file from data folder
import pandas as pd
import os
import numpy as np

path_data = '/Users/s190387/Desktop/golf/data/'
os.chdir(path_data)
lpga_data = pd.read_csv('lpga_kpmg.csv')
lpga_sg = pd.read_csv('lpga_shotsgained.csv')
lpga_driving = pd.read_csv('lpga_driving.csv')
lpga_approach = pd.read_csv('lpga_approach.csv')
lpga_shortgame = pd.read_csv('lpga_shortgame.csv')


def roundsplayed (df, rounds_col, name_col):
    df[rounds_col] = df[name_col].str.extract(r'Rounds played (\d+)$')
    df[name_col] = df[name_col].str.extract(r'^(.*?)Rounds played')
    return df

roundsplayed(lpga_data, 'Rounds Played', 'PLAYER')
roundsplayed(lpga_sg, 'Rounds Played', 'PLAYER')
roundsplayed(lpga_driving, 'Rounds Played', 'PLAYER')
roundsplayed(lpga_approach, 'Rounds Played', 'PLAYER')
roundsplayed(lpga_shortgame, 'Rounds Played', 'PLAYER')

# Drop rows in lpga_data where Scoring average is -
lpga_data = lpga_data[lpga_data['SCORING AVERAGE'] != '-']

# Merge lpga_data and lpga_sg on PLAYER, only several columns from lpga_sg
lpga_sg = lpga_sg[['PLAYER','SG:TEE TO GREEN', 'SG:OFF THE TEE', 'SG:APPROACH', 'SG:AROUND THE GREEN', 'SG:PUTTING']]
lpga_data = pd.merge(lpga_data, lpga_sg, on='PLAYER', how='left')
 # Merge relevant info from lpga_distance onto lpga_data
lpga_driving = lpga_driving[['PLAYER','FAIRWAYS HIT','DRIVING DISTANCE']]
# Remove 'yds' from DRIVING DISTANCE column
lpga_driving['DRIVING DISTANCE'] = lpga_driving['DRIVING DISTANCE'].str.replace('yds', '')
lpga_data = pd.merge(lpga_data, lpga_driving, on='PLAYER', how='left')
 # Merge relevant info from lpga_approach onto lpga_data
lpga_approach = lpga_approach[['PLAYER','GIR','AVG PROXIMITY (>50 YARDS)', 'GUR',]]
lpga_approach['AVG PROXIMITY (>50 YARDS)'] = lpga_approach['AVG PROXIMITY (>50 YARDS)'].str.replace('ft', '')

lpga_data = pd.merge(lpga_data, lpga_approach, on='PLAYER', how='left')
# Merge relevant info from lpga_shortgame onto lpga_data
lpga_shortgame = lpga_shortgame[['PLAYER', 'SCRAMBLING','PUTTS PER GIR']]
lpga_data = pd.merge(lpga_data, lpga_shortgame, on='PLAYER', how='left')

# Remove '#' from ROLEXRANK Column
lpga_data['ROLEXRANK'] = lpga_data['ROLEXRANK'].str.replace('#', '')
# Remove '%' from FAIRWAYS HIT, GIR, SCRAMBLING
lpga_data['FAIRWAYS HIT'] = lpga_data['FAIRWAYS HIT'].str.replace('%', '')
lpga_data['GIR'] = lpga_data['GIR'].str.replace('%', '')
lpga_data['SCRAMBLING'] = lpga_data['SCRAMBLING'].str.replace('%', '')
lpga_data['GUR'] = lpga_data['GUR'].str.replace('%', '')



# Replace '-' with Nan in all rows
lpga_data = lpga_data.replace('-', np.nan)
missing_data_rows = lpga_data[lpga_data.isnull().sum(axis=1) >= 5]

# Drop rows that are in missing_data_rows
lpga_data_filled = lpga_data.copy()
lpga_data_filled = lpga_data_filled.drop(missing_data_rows.index)

# Drop any datapoints with at least 5 columns of missing data
n_rows_before = lpga_data.shape[0]
n_rows_after = lpga_data_filled.shape[0]
#print('Number of rows before dropping rows with missing values: ', n_rows_before)
#print('Number of rows after dropping rows with missing values: ', n_rows_after)

# Change all columns except for PLAYER to numeric at once
for column in lpga_data_filled.columns:
    if column != 'PLAYER':
        lpga_data_filled[column] = lpga_data_filled[column].astype(float)

# Save lpga_data to csv file in data folder
lpga_data_filled.to_csv('lpga_data_filled.csv', index=False)

# Collinearity matrix between features
# lpga_data_corr = lpga_data_filled.drop(columns=['PLAYER'])
# corr_matrix = lpga_data_corr.corr()
# # Heatmap of collinearity matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True)
#plt.show()

# Features correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
lpga_data_corr_features = lpga_data_filled[['SG:TOTAL', 'Rounds Played', 'SG:TEE TO GREEN', 'SG:OFF THE TEE', 'SG:APPROACH', 'SG:AROUND THE GREEN', 'SG:PUTTING',
                                             'FAIRWAYS HIT', 'DRIVING DISTANCE', 'GIR', 'AVG PROXIMITY (>50 YARDS)', 'GUR', 'SCRAMBLING', 'PUTTS PER GIR' ]]
corr_matrix_features = lpga_data_corr_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_features, annot=True)
#plt.show()

# Drop SG:TOTAL column
lpga_data_filled = lpga_data_filled.drop(columns=['SG:TOTAL'])


# Outcomes correlation matrix
lpga_data_corr_outcomes = lpga_data_filled[['SCORING AVERAGE', 'ROLEXRANK', 'BIRDIES OR BETTER', 'PAR 3’s','PAR 4’s', 'PAR 5’s']]
corr_matrix_outcomes = lpga_data_corr_outcomes.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_outcomes, annot=True)
#plt.show()

# Outcome Varaibles to remove
outcome_variables = ['ROLEXRANK', 'BIRDIES OR BETTER', 'PAR 3’s','PAR 4’s', 'PAR 5’s']
lpga_data_filled = lpga_data_filled.drop(columns=outcome_variables)
lpga_data_filled = lpga_data_filled.dropna()

# Splitting data into Train and Test dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

X = lpga_data_filled.drop(columns = ['PLAYER', 'SCORING AVERAGE'])
y = lpga_data_filled['SCORING AVERAGE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#shapes of the training and testing sets
print('X_train: ', X_train_scaled.shape)    
print('X_test: ', X_test_scaled.shape)
print('y_train: ', y_train.shape)
print('y_test: ', y_test.shape)