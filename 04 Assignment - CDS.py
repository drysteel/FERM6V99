# -----------------------
# General package import:
# -----------------------
import numpy as np                                       # Scientific computing and arrays
import pandas as pd                                      # Data structures and analysis tools

# ------------------------------------
# Read and extract data from dta file:
# ------------------------------------
# Extract data
mydata = pd.read_stata("C:/Users/drysteel/Downloads/cds_spread5y_2001_2016.dta", low_memory = False)

# Show dimension of dataset
mydata.shape

# Show types of objects of dataset
mydata.dtypes

# Show first five observations of dataset
mydata.head()

# ------------------------------------
# Read and extract data from CSV file:
# ------------------------------------
# Extract data
myqmc = pd.read_csv("C:/Users/Abdramane/Documents/Quarterly Merged CRSP-Compustat.csv", low_memory = False)

# Show dimension of dataset
myqmc.shape

# Show types of objects of dataset
myqmc.dtypes

# Show first five observations of dataset
myqmc.head()

# Merge dta file and csv file
data = pd.merge(mydata, myqmc, on = "gvkey")