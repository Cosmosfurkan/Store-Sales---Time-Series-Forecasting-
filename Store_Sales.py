# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from google.colab import drive
from sklearn.linear_model import LinearRegression



# Load the dataset
data_train = pd.read_csv("/content/train.csv").head(90000)
data_test = pd.read_csv("/content/test.csv").head(90000)
print(data_train.head())
print(data_test.head())
data_train.drop('id', axis=1, inplace=True)
data_test.drop(['id'], axis=1, inplace=True)

# Data Preprocessing
def data_preprocess_train(data_train):
    data_train['date'] = pd.to_datetime(data_train['date'])
    data_train['month'] = data_train['date'].dt.month
    data_train['day'] = data_train['date'].dt.day
    data_train['day_of_week'] = data_train['date'].dt.dayofweek
    data_train['weekend'] = data_train['day_of_week'].isin([5, 6]).astype(int)
    data_train['day_of_year'] = data_train['date'].dt.dayofyear
    data_train['quarter'] = data_train['month'].apply(lambda x: (x - 1) // 3 + 1)
    data_train['family'] = la.fit_transform(data_train['family'])
    return data_train

# Data Preprocessing
def data_preprocess_test(data_test):
    data_test['date'] = pd.to_datetime(data_test['date'])
    data_test['month'] = data_test['date'].dt.month
    data_test['day'] = data_test['date'].dt.day
    data_test['day_of_week'] = data_test['date'].dt.dayofweek
    data_test['weekend'] = data_test['day_of_week'].isin([5, 6]).astype(int)
    data_test['day_of_year'] = data_test['date'].dt.dayofyear
    data_test['quarter'] = data_test['month'].apply(lambda x: (x - 1) // 3 + 1)
    data_test['family'] = la.fit_transform(data_test['family'])
    return data_test

# Feature Engineering
def feature_engineering(X):
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['day_of_week'] = X['date'].dt.dayofweek
    X['weekend'] = X['day_of_week'].isin([5, 6]).astype(int)
    X['day_of_year'] = X['date'].dt.dayofyear
    X['quarter'] = X['month'].apply(lambda x: (x - 1) // 3 + 1)
    X.drop('date', axis=1, inplace=True)
    return X

# Define LabelEncoder
la = LabelEncoder()

data_train = data_preprocess_train(data_train)
data_test = data_preprocess_test(data_test)

la.fit_transform(data_train['family'])

# Split Data to Train and Test
y_train = data_train["sales"]
X_train = feature_engineering(data_train.drop('sales', axis=1))
X_test = feature_engineering(data_test)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model Building
linear_model = LinearRegression()
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
hard_voting_regressor = VotingRegressor(estimators=[('linear', linear_model), ('tree', random_forest)])


hard_voting_regressor.fit(X_train, y_train)
prediction = hard_voting_regressor.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, prediction)))
print("R2 score:", r2_score(y_test, prediction))
