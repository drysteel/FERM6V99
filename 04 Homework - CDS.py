# -----------------------------------------------
# Runned on Python 2.7
# Predicting prices of credit default swaps (CDS)
# -----------------------------------------------

# -----------------------
# General package import:
# -----------------------
import pandas as pd              # Data structure ana analysis tools 

# ---------------------------------
# Retreive pre-downloaded datasets:
# ---------------------------------
# Import dataset for CDS
cds = pd.read_stata("C:/Users/drysteel/Downloads/cds_spread5y_2001_2016.dta")

# Import dataset for csv information (CRSP)
csv = pd.read_csv("C:/Users/drysteel/Downloads/Quarterly Merged CRSP-Compustat.csv", low_memory = False)

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