# -----------------------------------------------
# Runned on Python 2.7
# Predicting prices of credit default swaps (CDS)
# -----------------------------------------------

# -----------------------
# General package import:
# -----------------------
import numpy as np                                     # Scientific computing and arrays
import pandas as pd                                    # Data structure ana analysis tools
import xgboost as xgb                                  # Generate XGBoost
import pylab as pl                                     # Generate plot
from sklearn.ensemble import RandomForestRegressor     # Import random forest from Scikit
from sklearn.metrics import mse as mse                 # Import MSE from scikit
from sklearn.ensemble import GradientBoostingRegressor # Import gradient boosting

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
# Count null values
merdat.isnull().sum()
merdat.describe()

# Impute missing values
merdat.fillna(merdat.median())

# Remove non-numeric variables
merdata = merdat.select_dtypes([np.number])
merdata

# Check missing values
merdata.isnull().sum()

# Remove still missing variables
merdata2 = merdata.dropna(axis = 1, how = 'any')
merdata2.head()

# ----------------------------------
# Split dataset into train and test:
# ----------------------------------
# Create and split train data
train = merdata2[(merdata2['Year'] < 2016)]
X_train = train.drop('spread5y', axis = 1)
y_train = train['spread5y']

# Remove features from train data
X_train= X_train.drop('Month_x', axis = 1)
X_train= X_train.drop('Month_y', axis = 1)
X_train= X_train.drop('Quarter', axis = 1)
X_train= X_train.drop('Year', axis = 1)
X_train= X_train.drop('gvkey', axis = 1)

# Create and split test data
test = merdata2[(merdata2['Year'] >= 2016) & (merdata2['Year'] <= 2018)]
X_test = test.drop('spread5y', axis = 1)
y_test = test['spread5y']

# Remove features from test data
X_test = X_test.drop('Month_x', axis = 1)
X_test = X_test.drop('Month_y', axis = 1)
X_test = X_test.drop('Quarter', axis = 1)
X_test = X_test.drop('Year', axis = 1)
X_test = X_test.drop('gvkey', axis = 1)

# -------------------------
# Random forest classifier:
# -------------------------
# Number of trees = 50
rf = RandomForestRegressor(n_estimators = 50) 
rf.fit(X_train, y_train) 

# Score the model
rf.score(X_test, y_test)
rf50 = rf.predict(X_test)

# Segragate first 50 features, in order of importance
featimp = rf.feature_importances_

feaimp = pd.DataFrame(rf.feature_importances_, index = X_train.columns, 
                      columns = ['importance']).sort_values('importance',
                                ascending = False)
top50 = featimp.iloc[:50, :]
top50 = top50.index.tolist()

def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mse(y_test, rf50)
mape(y_test, rf50)

# Filter top 50 features
Filtered_X_train = X_train[top50]
Filtered_X_test = X_test[top50]

# -----------------------
# Generate random forest:
# -----------------------
# Number of trees = 100
rf100 = RandomForestRegressor(n_estimators = 100) 
rf100.fit(Filtered_X_train, y_train)
rf100_pred = rf100.predict(Filtered_X_test)

mape100 = mape(y_test, rf100_pred)
mape100
mse100 = mse(y_test, rf100_pred)
mse100

# Number of trees = 200
rf200 = RandomForestRegressor(n_estimators = 200) 
rf200.fit(Filtered_X_train, y_train)
rf200_pred = rf200.predict(Filtered_X_test)

mape200 = mape(y_test, rf200_pred)
mape200
mse200 = mse(y_test, rf200_pred)
mse200

# Number of trees = 500
rf500 = RandomForestRegressor(n_estimators = 500) 
rf500.fit(Filtered_X_train, y_train)
rf500_pred = rf500.predict(Filtered_X_test)

mape500 = mape(y_test, rf500_pred)
mape500
mse500 = mse(y_test, rf500_pred)
mse500

# Number of trees = 1000
rf1000 = RandomForestRegressor(n_estimators = 1000) 
rf1000.fit(Filtered_X_train, y_train)
rf1000_pred = rf1000.predict(Filtered_X_test)

mape1000 = mape(y_test, rf1000_pred)
mape1000
mse1000 = mse(y_test, rf1000_pred)
mse1000

# --------------------------
# Execute gradient boosting:
# --------------------------
# Number of trees = 100
gb100 = GradientBoostingRegressor(n_estimators = 100) 
gb100.fit(Filtered_X_train, y_train)
gb100_pred = gb100.predict(Filtered_X_test)

mapegb100 = mape(y_test, gb100_pred)
mapegb100
msegb100 = mse(y_test, gb100_pred)
msegb100

# Number of trees = 200
gb200 = GradientBoostingRegressor(n_estimators = 200) 
gb200.fit(Filtered_X_train, y_train)
gb200_pred = gb200.predict(Filtered_X_test)

mapegb200 = mape(y_test, gb200_pred)
mapegb200
msegb200 = mse(y_test, gb200_pred)
msegb200

# Number of trees = 500
gb500 = GradientBoostingRegressor(n_estimators = 500) 
gb500.fit(Filtered_X_train, y_train)
gb500_pred = gb500.predict(Filtered_X_test)

mapegb500 = mape(y_test, gb500_pred)
mapegb500
msegb500 = mse(y_test, gb500_pred)
msegb500

# Number of trees = 1000
gb1000 = GradientBoostingRegressor(n_estimators = 1000) 
gb1000.fit(Filtered_X_train, y_train)
gb1000_pred = gb1000.predict(Filtered_X_test)

mapegb1000 = mape(y_test, gb1000_pred)
mapegb1000
msegb1000 = mse(y_test, gb1000_pred)
msegb1000

# -----------------
# Generate XGBoost:
# -----------------
# Declare number of trees for XGBoost
xgb100 = xgb.XGBRegressor(objective = 'reg:linear', n_estimators = 100)
xgb200 = xgb.XGBRegressor(objective = 'reg:linear', n_estimators = 200)
xgb500 = xgb.XGBRegressor(objective = 'reg:linear', n_estimators = 500)
xgb1000 = xgb.XGBRegressor(objective = 'reg:linear', n_estimators = 1000)

xgbf100 = xgb100.fit(Filtered_X_train, y_train)
xgbf200 = xgb200.fit(Filtered_X_train, y_train)
xgbf500 = xgb500.fit(Filtered_X_train, y_train)
xgbf1000 = xgb1000.fit(Filtered_X_train, y_train)

xgb100_pred = xgbf100.predict(Filtered_X_test)  
xgb200_pred = xgbf200.predict(Filtered_X_test)
xgb500_pred = xgbf500.predict(Filtered_X_test)  
xgb1000_pred = xgbf1000.predict(Filtered_X_test)

mapexgb100 = mape(y_test, xgb100_pred)
mapexgb100
msexgb100 = mse(y_test, xgb100_pred)
msexgb100

mapexgb200 = mape(y_test, xgb200_pred)
mapexgb200
msexgb200 = mse(y_test, xgb200_pred)
msexgb200

mapexgb500 = mape(y_test, xgb500_pred)
mapexgb500
msexgb500 = mse(y_test, xgb500_pred)
msexgb500

mapexgb1000 = mape(y_test, xgb1000_pred)
mapexgb1000
msexgb1000 = mse(y_test, xgb1000_pred)
msexgb1000

# ----------------------------
# Print the regression models:
# ----------------------------
# MSE random forest
print 'MSE for RandomForest with 100 trees: ', mse100
print 'MSE for RandomForest with 200 trees: ', mse200
print 'MSE for RandomForest with 500 trees: ', mse500
print 'MSE for RandomForest with 1000 trees: ', mse1000

mselist = []
mselist.append(mse100)
mselist.append(mse200)
mselist.append(mse500)
mselist.append(mse1000)

# MSE gradient boosting
msegb1000 = mse(y_test, gb1000_pred)
msegb1000

print 'MSE for GradientBoosting with 100 trees: ', msegb100
print 'MSE for GradientBoosting with 200 trees: ', msegb200
print 'MSE for GradientBoosting with 500 trees: ', msegb500
print 'MSE for GradientBoosting with 1000 trees: ', msegb1000

msegblist = []
msegblist.append(msegb100)
msegblist.append(msegb200)
msegblist.append(msegb500)
msegblist.append(msegb1000)

# MSE XGBoost
print 'MSE for Xgboost with 100 trees: ', msexgb100
print 'MSE for Xgboost with 200 trees: ', msexgb200
print 'MSE for Xgboost with 500 trees: ', msexgb500
print 'MSE for Xgboost with 1000 trees: ', msexgb1000

msexgblist=[]
msexgblist.append(msexgb100)
msexgblist.append(msexgb200)
msexgblist.append(msexgb500)
msexgblist.append(msexgb1000)

# ------------------------------------
# Generate plot for regression models:
# ------------------------------------
tree = [100, 200, 500, 1000]
pl.plot(tree, mselist, label = 'Random Forest')
pl.plot(tree,msegblist, label = 'Gradient Boosting' )
pl.plot(tree,msexgblist,label = 'XGBoost')
pl.legend(loc = 'upper right')
pl.ylabel('MSE')
pl.xlabel('Number of Trees')
pl.show()