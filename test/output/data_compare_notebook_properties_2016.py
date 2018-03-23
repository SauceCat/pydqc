
"""
## import useful packages

"""

import pandas as pd
import numpy as np

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from matplotlib_venn import venn2

%matplotlib inline

from pydqc.data_compare import distribution_compare_pretty

"""
## assign values

"""

# the data table (pandas DataFrame)
table1 =
table2 =
print("table1 size: " + str(table1.shape))
print("table2 size: " + str(table2.shape))

# global values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"
TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"

# get date of today
snapshot_date_now = str(datetime.datetime.now().date())
print("date of today: " + snapshot_date_now)
"""
## error columns

**decktypeid:** exclude<br>**hashottuborspa:** exclude<br>**poolcnt:** exclude<br>**pooltypeid10:** exclude<br>**pooltypeid2:** exclude<br>**pooltypeid7:** exclude<br>**storytypeid:** exclude<br>**fireplaceflag:** exclude<br>**taxdelinquencyflag:** exclude<br>"""


"""
## parcelid (type: key)

"""

col = "parcelid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"],set_colors=("#4BACC6", "#F79646"), alpha=0.8)
"""
## airconditioningtypeid (type: str)

"""

col = "airconditioningtypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## architecturalstyletypeid (type: str)

"""

col = "architecturalstyletypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## basementsqft (type: numeric)

"""

col = "basementsqft"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## bathroomcnt (type: numeric)

"""

col = "bathroomcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## bedroomcnt (type: numeric)

"""

col = "bedroomcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## buildingclasstypeid (type: str)

"""

col = "buildingclasstypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## buildingqualitytypeid (type: str)

"""

col = "buildingqualitytypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## calculatedbathnbr (type: numeric)

"""

col = "calculatedbathnbr"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedfloor1squarefeet (type: numeric)

"""

col = "finishedfloor1squarefeet"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## calculatedfinishedsquarefeet (type: numeric)

"""

col = "calculatedfinishedsquarefeet"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet12 (type: numeric)

"""

col = "finishedsquarefeet12"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet13 (type: numeric)

"""

col = "finishedsquarefeet13"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet15 (type: numeric)

"""

col = "finishedsquarefeet15"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet50 (type: numeric)

"""

col = "finishedsquarefeet50"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## finishedsquarefeet6 (type: numeric)

"""

col = "finishedsquarefeet6"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## fips (type: str)

"""

col = "fips"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## fireplacecnt (type: numeric)

"""

col = "fireplacecnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## fullbathcnt (type: numeric)

"""

col = "fullbathcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## garagecarcnt (type: numeric)

"""

col = "garagecarcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## garagetotalsqft (type: numeric)

"""

col = "garagetotalsqft"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## heatingorsystemtypeid (type: str)

"""

col = "heatingorsystemtypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## latitude (type: numeric)

"""

col = "latitude"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## longitude (type: numeric)

"""

col = "longitude"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## lotsizesquarefeet (type: numeric)

"""

col = "lotsizesquarefeet"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## poolsizesum (type: numeric)

"""

col = "poolsizesum"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## propertycountylandusecode (type: str)

"""

col = "propertycountylandusecode"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## propertylandusetypeid (type: str)

"""

col = "propertylandusetypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## propertyzoningdesc (type: str)

"""

col = "propertyzoningdesc"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## rawcensustractandblock (type: key)

"""

col = "rawcensustractandblock"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"],set_colors=("#4BACC6", "#F79646"), alpha=0.8)
"""
## regionidcity (type: str)

"""

col = "regionidcity"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## regionidcounty (type: str)

"""

col = "regionidcounty"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## regionidneighborhood (type: str)

"""

col = "regionidneighborhood"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## regionidzip (type: str)

"""

col = "regionidzip"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## roomcnt (type: numeric)

"""

col = "roomcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## threequarterbathnbr (type: numeric)

"""

col = "threequarterbathnbr"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## typeconstructiontypeid (type: str)

"""

col = "typeconstructiontypeid"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## unitcnt (type: numeric)

"""

col = "unitcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## yardbuildingsqft17 (type: numeric)

"""

col = "yardbuildingsqft17"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## yardbuildingsqft26 (type: numeric)

"""

col = "yardbuildingsqft26"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## yearbuilt (type: str)

"""

col = "yearbuilt"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## numberofstories (type: numeric)

"""

col = "numberofstories"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## structuretaxvaluedollarcnt (type: numeric)

"""

col = "structuretaxvaluedollarcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## taxvaluedollarcnt (type: numeric)

"""

col = "taxvaluedollarcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## assessmentyear (type: str)

"""

col = "assessmentyear"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### check value counts

"""

value_counts_df1 = pd.DataFrame(df1[col].value_counts())
value_counts_df1.columns = ["count_1"]
value_counts_df1[col] = value_counts_df1.index.values
value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]
value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)

value_counts_df2 = pd.DataFrame(df2[col].value_counts())
value_counts_df2.columns = ["count_2"]
value_counts_df2[col] = value_counts_df2.index.values
value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]
value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)

value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)

"""
"""

value_counts_df
"""
## landtaxvaluedollarcnt (type: numeric)

"""

col = "landtaxvaluedollarcnt"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## taxamount (type: numeric)

"""

col = "taxamount"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## taxdelinquencyyear (type: numeric)

"""

col = "taxdelinquencyyear"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
#### check basic stats

"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))

# min value
value_min1 = df1[col].min()
value_min2 = df2[col].min()
print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))

# mean value
value_mean1 = df1[col].mean()
value_mean2 = df2[col].mean()
print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))

# median value
value_median1 = df1[col].median()
value_median2 = df2[col].median()
print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))

# max value
value_max1 = df1[col].max()
value_max2 = df2[col].max()
print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))

"""
#### check distribution

"""

value_dropna_df1 = df1[col].dropna().values
value_dropna_df2 = df2[col].dropna().values
plt.figure(figsize=(10, 5))
sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")
sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")

"""
"""

# you can also use the build-in draw function
distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
"""
## censustractandblock (type: key)

"""

col = "censustractandblock"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### basic comparison

"""

# sample values

"""
"""

df1.sample(5)

"""
"""

df2.sample(5)

"""
"""

# nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]

print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

"""
"""

# num_uni
num_uni1 = df1[col].dropna().nunique()
num_uni2 = df2[col].dropna().nunique()

print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))
print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))


"""
#### compare intersection

"""

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"],set_colors=("#4BACC6", "#F79646"), alpha=0.8)