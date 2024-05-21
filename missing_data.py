# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Data set 
Dataset = pd.read_csv('data.csv')
x  = Dataset.iloc[:, : -1].values
y = Dataset.iloc[:, -1].values
#print(x)
#print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer =  SimpleImputer(missing_values = np.nan, strategy = 'mean')
# exclude the string columns
imputer.fit(x[:, 1:3])
x[:, 1:3]= imputer.transform(x[:, 1:3])
#print(x)
#encoding categories data
#encoding the independent variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x= np.array(ct.fit_transform(x))
#print(x)
#encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#print(y)
#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_train)
print(x_test)

