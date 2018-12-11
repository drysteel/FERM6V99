# -----------------------
# General package import:
# -----------------------
import numpy as np                                       # Scientific computing and arrays
import pandas as pd                                      # Data structure ana analysis tools
from sklearn.ensemble import RandomForestRegressor       # Import random forest from Scikit
from sklearn.metrics import mean_squared_error as mse    # Import MSE from scikit
from sklearn.model_selection import train_test_split     # For random train and test splits
from sklearn.preprocessing import StandardScaler         # For feature standardization
from keras.models import Sequential                      # Configure the model for training
from keras.layers import Dense                           # Neural network layer

# ---------------------------------
# Retreive pre-downloaded datasets:
# ---------------------------------
# Import dataset for CDS
cds = pd.read_stata("C:/Users/drysteel/Downloads/cds_spread5y_2001_2016.dta")

# Import dataset for csv information (CRSP)
csv = pd.read_csv("C:/Users/drysteel/Downloads/Quarterly Merged CRSP-Compustat.csv",
                  low_memory = False)

# -----------------------------------------
# Classify CDS date/time to month and year:
# -----------------------------------------
# Separate date from CDS
cds['Date'] = pd.to_datetime(cds['mdate'])

# Classify months
cds['Month'] = cds['Date'].dt.month

# Classify years
cds['Year'] = cds['Date'].dt.year

# ------------------------------------------------
# Group months to quarters to match CDS with CRSP:
# ------------------------------------------------
# Set and divide based on number of months
cds['Quarter'] = '4'

# Above month 9 falls to Q4
cds['Quarter'][cds['Month'] > 9] = '4'

# Month 7 to 9 is Q3
cds['Quarter'][(cds['Month'] > 6) & (cds['Month'] < 9)] = '3'

# Month 4 to 6 is Q2
cds['Quarter'][(cds['Month'] > 3) & (cds['Month'] < 6)] = '2'

# Below month 4 falls to Q1
cds['Quarter'][cds['Month'] < 3] = '1'

# -------------------------------------------
# Minor column conversion and naming changes:
# -------------------------------------------
# Convert selected columns to float to merge with CRSP
cds['gvkey'] = cds['gvkey'].astype(float)
cds['Quarter'] = cds['Quarter'].astype(float)
cds['Year'] = cds['Year'].astype(float)

# Change selected column names to match with CDS
csv = csv.rename(columns = {'datadate':'mdate'})
csv = csv.rename(columns = {'GVKEY':'gvkey'})

# -----------------------------------------
# Classify CDS date/time to month and year:
# -----------------------------------------
# Separate date from CRSP
csv['Date'] = pd.to_datetime(csv['mdate'])

# Classify months
csv['Month'] = csv['Date'].dt.month

# Classify years
csv['Year'] = csv['Date'].dt.year

# ------------------------------------------------
# Group months to quarters to match CRSP with CDS:
# ------------------------------------------------
# Set and divide based on number of months
csv['Quarter'] = '4'

# Above month 9 falls to Q4
csv['Quarter'][csv['Month'] > 9] = '4'

# Month 7 to 9 is Q3
csv['Quarter'][(csv['Month'] > 6) & (csv['Month'] < 9)] = '3'

# Month 4 to 6 is Q2
csv['Quarter'][(csv['Month'] > 3) & (csv['Month'] < 6)] = '2'

# Below month 4 falls to Q1
csv['Quarter'][csv['Month'] < 3] = '1'

# -------------------------------------------
# Minor column conversion and naming changes:
# -------------------------------------------
# Convert selected columns to float to merge with CDS
csv['gvkey'] = csv['gvkey'].astype(float)
csv['Quarter'] = csv['Quarter'].astype(float)
csv['Year'] = csv['Year'].astype(float)

# ------------------------------------------
# Merge selected columns from both datasets:
# ------------------------------------------ 
merdat = pd.merge(csv, cds, on = ['gvkey', 'Quarter', 'Year'])

# --------------
# Data cleaning:
# --------------
# Numerical columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# Remove non-numeric variables
merdat = merdat.select_dtypes(include = numerics)

# Impute missing values as median
merdat = merdat.fillna(merdat.median())

# Remove still missing variables
merdat = merdat.dropna(axis = 1, how = 'any')

# -----------
# Split data:
# -----------
X = merdat.drop('spread5y', axis = 1)
y = merdat['spread5y']

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train

# ------------------
# Feature selection:
# ------------------
# Remove features test dataset
X_test = X_test.drop('Month_x', axis = 1)
X_test = X_test.drop('Month_y', axis = 1)
X_test = X_test.drop('Quarter', axis = 1)
X_test = X_test.drop('Year',    axis = 1)
X_test = X_test.drop('gvkey',   axis = 1)

# Remove features train dataset
X_train= X_train.drop('Month_x', axis = 1)
X_train= X_train.drop('Month_y', axis = 1)
X_train= X_train.drop('Quarter', axis = 1)
X_train= X_train.drop('Year',    axis = 1)
X_train= X_train.drop('gvkey',   axis = 1)

# -----------------------
# Standardizing features:
# -----------------------
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy = True, with_mean = True, with_std = True)

X_train_1 = X_train
X_test_1 = X_test

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Random forest classifier:
# -------------------------
# Number of trees = 10
rf = RandomForestRegressor(n_estimators = 10) 
rf.fit(X_train, y_train) 

# Score the model
rf.score(X_test, y_test)
rf10 = rf.predict(X_test)

# Segragate top 40 features, in order of importance
featimp = rf.feature_importances_

featimp = pd.DataFrame(rf.feature_importances_, index = X_train.columns, 
                      columns = ['importance']).sort_values('importance',
                                ascending = False)
top40 = featimp.iloc[:40, :]
top40 = top40.index.tolist()

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mse(y_test, rf10)
mape(y_test, rf10)

# Filter top 50 features
Fil_X_train = X_train[top40]
Fil_X_test = X_test[top40]

# Filter the fitted standardized training data
scaler.fit(Fil_X_train)
StandardScaler(copy = True, with_mean = True, with_std = True)

Fil_X_train_1 = scaler.transform(Fil_X_train)
Fil_X_test_1 = scaler.transform(Fil_X_test)

# ---------------
# Neural Network:
# ---------------
# Generate fix random seed
np.random.seed(7)

# Create multilayer perceptron (MLP) model
model = Sequential()
model.add(Dense(32, input_dim = 50, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

# Compile MLP
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

# Fit MLP
model.fit(Fil_X_train_1, y_train, epochs = 100, batch_size = 10)

NN1 = model.predict(Fil_X_test_1)
mapeNN1 = mape(y_test, NN1)
mapeNN1

# Create second model
model = Sequential()
model.add(Dense(32, input_dim = 50, activation = 'sigmoid'))
model.add(Dense(8, activation = 'sigmoid'))
model.add(Dense(1, activation = 'relu'))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(Fil_X_train_1, y_train, epochs = 30, batch_size = 10)

NN2 = model.predict(Fil_X_test_1)
mapeNN2 = mape(y_test, NN2)
mapeNN2

# Create second model
model = Sequential()
model.add(Dense(32, input_dim = 50, activation = 'hard_sigmoid'))
model.add(Dense(8, activation = 'elu'))
model.add(Dense(1, activation = 'relu'))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(Fil_X_train_1, y_train, epochs = 30, batch_size = 10)

NN3 = model.predict(Fil_X_test_1)
mapeNN3 = mape(y_test, NN3)
mapeNN3

print("Both MODEL 2 AND 3 are better models than MODEL 1")