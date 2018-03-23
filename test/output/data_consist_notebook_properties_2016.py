
"""
## import useful packages

"""

import pandas as pd
import numpy as np

import datetime

from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
from matplotlib_venn import venn2

%matplotlib inline

from pydqc.data_consist import numeric_consist_pretty

"""
## assign values

"""

#the data table (pandas DataFrame)
table1 =
table2 =
print("table1 size: " + str(table1.shape))
print("table2 size: " + str(table2.shape))

key1 =
key2 =

#global values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"
TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"

#get date of today
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
#### compare intersection

"""

#nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]
print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"], set_colors=("#4BACC6", "#F79646"), alpha=0.8)
"""
## airconditioningtypeid (type: str)

"""

col = "airconditioningtypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## architecturalstyletypeid (type: str)

"""

col = "architecturalstyletypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## basementsqft (type: numeric)

"""

col = "basementsqft"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## bathroomcnt (type: numeric)

"""

col = "bathroomcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## bedroomcnt (type: numeric)

"""

col = "bedroomcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## buildingclasstypeid (type: str)

"""

col = "buildingclasstypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## buildingqualitytypeid (type: str)

"""

col = "buildingqualitytypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## calculatedbathnbr (type: numeric)

"""

col = "calculatedbathnbr"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedfloor1squarefeet (type: numeric)

"""

col = "finishedfloor1squarefeet"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## calculatedfinishedsquarefeet (type: numeric)

"""

col = "calculatedfinishedsquarefeet"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedsquarefeet12 (type: numeric)

"""

col = "finishedsquarefeet12"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedsquarefeet13 (type: numeric)

"""

col = "finishedsquarefeet13"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedsquarefeet15 (type: numeric)

"""

col = "finishedsquarefeet15"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedsquarefeet50 (type: numeric)

"""

col = "finishedsquarefeet50"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## finishedsquarefeet6 (type: numeric)

"""

col = "finishedsquarefeet6"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## fips (type: str)

"""

col = "fips"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## fireplacecnt (type: numeric)

"""

col = "fireplacecnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## fullbathcnt (type: numeric)

"""

col = "fullbathcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## garagecarcnt (type: numeric)

"""

col = "garagecarcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## garagetotalsqft (type: numeric)

"""

col = "garagetotalsqft"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## heatingorsystemtypeid (type: str)

"""

col = "heatingorsystemtypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## latitude (type: numeric)

"""

col = "latitude"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## longitude (type: numeric)

"""

col = "longitude"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## lotsizesquarefeet (type: numeric)

"""

col = "lotsizesquarefeet"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## poolsizesum (type: numeric)

"""

col = "poolsizesum"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## propertycountylandusecode (type: str)

"""

col = "propertycountylandusecode"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## propertylandusetypeid (type: str)

"""

col = "propertylandusetypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## propertyzoningdesc (type: str)

"""

col = "propertyzoningdesc"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## rawcensustractandblock (type: key)

"""

col = "rawcensustractandblock"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### compare intersection

"""

#nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]
print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"], set_colors=("#4BACC6", "#F79646"), alpha=0.8)
"""
## regionidcity (type: str)

"""

col = "regionidcity"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## regionidcounty (type: str)

"""

col = "regionidcounty"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## regionidneighborhood (type: str)

"""

col = "regionidneighborhood"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## regionidzip (type: str)

"""

col = "regionidzip"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## roomcnt (type: numeric)

"""

col = "roomcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## threequarterbathnbr (type: numeric)

"""

col = "threequarterbathnbr"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## typeconstructiontypeid (type: str)

"""

col = "typeconstructiontypeid"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## unitcnt (type: numeric)

"""

col = "unitcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## yardbuildingsqft17 (type: numeric)

"""

col = "yardbuildingsqft17"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## yardbuildingsqft26 (type: numeric)

"""

col = "yardbuildingsqft26"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## yearbuilt (type: str)

"""

col = "yearbuilt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## numberofstories (type: numeric)

"""

col = "numberofstories"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## structuretaxvaluedollarcnt (type: numeric)

"""

col = "structuretaxvaluedollarcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## taxvaluedollarcnt (type: numeric)

"""

col = "taxvaluedollarcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## assessmentyear (type: str)

"""

col = "assessmentyear"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")

# calculate consistency
df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan' and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]
print("consistency rate: " + str(corr))
"""
## landtaxvaluedollarcnt (type: numeric)

"""

col = "landtaxvaluedollarcnt"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## taxamount (type: numeric)

"""

col = "taxamount"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## taxdelinquencyyear (type: numeric)

"""

col = "taxdelinquencyyear"

df1 = table1[[key1, col]].copy()
df2 = table2[[key2, col]].copy()

"""
#### check pairwise consistency

"""

# merge 2 tables
df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")
df = df.dropna(how='any', subset=[col + "_x", col + "_y"]).reset_index(drop=True)

corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)
print("consistency rate: " + str(corr))

"""
#### draw consistency graph

"""

# prepare data
df["diff_temp"] = df[col + "_y"] - df[col + "_x"]
draw_values = df["diff_temp"].dropna().values

both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])
both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])

# draw
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)
plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")
plt.xlim(both_min, both_max)
plt.ylim(both_min, both_max)
plt.title("corr: %.3f" %(corr))

plt.subplot(122)
sns.distplot(draw_values, color=TABLE2_DARK)
plt.title("Distribution of differences")

"""
"""

#you can also use the build-in draw function
numeric_consist_pretty(df1, df2, key1, key2, col)
"""
## censustractandblock (type: key)

"""

col = "censustractandblock"

df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

"""
#### compare intersection

"""

#nan_rate
nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]
print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))

set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()
set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()
col_overlap = len(set_df1_col.intersection(set_df2_col))
col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))

"""
#### draw venn graph

"""

plt.figure(figsize=(10, 5))
venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"], set_colors=("#4BACC6", "#F79646"), alpha=0.8)