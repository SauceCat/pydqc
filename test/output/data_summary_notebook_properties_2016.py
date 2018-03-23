
"""
## import useful packages

"""

import pandas as pd
import numpy as np

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

%matplotlib inline

from pydqc.data_summary import distribution_summary_pretty


"""
## assign values

"""

# the data table (pandas DataFrame)
table =
print("table size: " + str(table.shape))

# global values
VER_LINE = "#4BACC6"
TEXT_LIGHT = "#DAEEF3"
DIS_LINE = "#F79646"

# get date of today
snapshot_date_now = str(datetime.datetime.now().date())
print("date of today: " + snapshot_date_now)
"""
## excluded columns

decktypeid, hashottuborspa, poolcnt, pooltypeid10, pooltypeid2, pooltypeid7, storytypeid, fireplaceflag, taxdelinquencyflag
"""


"""
## parcelid (type: key)

"""

col = "parcelid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## airconditioningtypeid (type: str)

"""

col = "airconditioningtypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## architecturalstyletypeid (type: str)

"""

col = "architecturalstyletypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## basementsqft (type: numeric)

"""

col = "basementsqft"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## bathroomcnt (type: numeric)

"""

col = "bathroomcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## bedroomcnt (type: numeric)

"""

col = "bedroomcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## buildingclasstypeid (type: str)

"""

col = "buildingclasstypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## buildingqualitytypeid (type: str)

"""

col = "buildingqualitytypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## calculatedbathnbr (type: numeric)

"""

col = "calculatedbathnbr"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedfloor1squarefeet (type: numeric)

"""

col = "finishedfloor1squarefeet"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## calculatedfinishedsquarefeet (type: numeric)

"""

col = "calculatedfinishedsquarefeet"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet12 (type: numeric)

"""

col = "finishedsquarefeet12"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet13 (type: numeric)

"""

col = "finishedsquarefeet13"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet15 (type: numeric)

"""

col = "finishedsquarefeet15"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet50 (type: numeric)

"""

col = "finishedsquarefeet50"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet6 (type: numeric)

"""

col = "finishedsquarefeet6"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## fips (type: str)

"""

col = "fips"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## fireplacecnt (type: numeric)

"""

col = "fireplacecnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## fullbathcnt (type: numeric)

"""

col = "fullbathcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## garagecarcnt (type: numeric)

"""

col = "garagecarcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## garagetotalsqft (type: numeric)

"""

col = "garagetotalsqft"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## heatingorsystemtypeid (type: str)

"""

col = "heatingorsystemtypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## latitude (type: numeric)

"""

col = "latitude"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## longitude (type: numeric)

"""

col = "longitude"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## lotsizesquarefeet (type: numeric)

"""

col = "lotsizesquarefeet"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## poolsizesum (type: numeric)

"""

col = "poolsizesum"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## propertycountylandusecode (type: str)

"""

col = "propertycountylandusecode"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## propertylandusetypeid (type: str)

"""

col = "propertylandusetypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## propertyzoningdesc (type: str)

"""

col = "propertyzoningdesc"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## rawcensustractandblock (type: key)

"""

col = "rawcensustractandblock"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## regionidcity (type: str)

"""

col = "regionidcity"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## regionidcounty (type: str)

"""

col = "regionidcounty"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## regionidneighborhood (type: str)

"""

col = "regionidneighborhood"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## regionidzip (type: str)

"""

col = "regionidzip"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## roomcnt (type: numeric)

"""

col = "roomcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## threequarterbathnbr (type: numeric)

"""

col = "threequarterbathnbr"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## typeconstructiontypeid (type: str)

"""

col = "typeconstructiontypeid"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## unitcnt (type: numeric)

"""

col = "unitcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## yardbuildingsqft17 (type: numeric)

"""

col = "yardbuildingsqft17"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## yardbuildingsqft26 (type: numeric)

"""

col = "yardbuildingsqft26"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## yearbuilt (type: str)

"""

col = "yearbuilt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## numberofstories (type: numeric)

"""

col = "numberofstories"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## structuretaxvaluedollarcnt (type: numeric)

"""

col = "structuretaxvaluedollarcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## taxvaluedollarcnt (type: numeric)

"""

col = "taxvaluedollarcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## assessmentyear (type: str)

"""

col = "assessmentyear"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)
"""
## landtaxvaluedollarcnt (type: numeric)

"""

col = "landtaxvaluedollarcnt"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## taxamount (type: numeric)

"""

col = "taxamount"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## taxdelinquencyyear (type: numeric)

"""

col = "taxdelinquencyyear"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check basic stats

"""

value_min=value_df[col].min()
value_mean=value_df[col].mean()
value_median=value_df[col].median()
value_max=value_df[col].max()

print("min: " + str(value_min))
print("mean: " + str(value_mean))
print("median: " + str(value_median))
print("max: " + str(value_max))

"""
#### check distribution

"""

value_dropna = value_df[col].dropna().values
plt.figure(figsize=(10, 5))
plt.title(col)
sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)

"""
"""

#you can also use the build-in draw function
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
"""
## censustractandblock (type: key)

"""

col = "censustractandblock"

value_df = table[[col]].copy()
nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]
num_uni = value_df[col].dropna().nunique()

print("nan_rate: " + str(nan_rate))
print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))

"""
#### check value counts

"""

value_df[col].value_counts().head(10)