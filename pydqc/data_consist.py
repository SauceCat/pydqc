import pandas as pd
import numpy as np
import os
import shutil

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

from sklearn.externals.joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('white')

import datetime
from scipy.stats import pearsonr

import sankey

import warnings
warnings.filterwarnings('ignore')
from matplotlib_venn import venn2

from dqc_utils import _style_range
from data_compare import _compare_key, _insert_compare_results


# global color values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"

TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"


"""
function: draw pretty consist graph for numeric columns
parameters:
_df1: pandas DataFrame
	slice of table1 containing enough information to check
_df2: pandas DataFrame
	slice of table2 containing enough information to check
_key1: string
	key for table1
_key2: string
	key for table2
col: string
	name of column to check
figsize: tuple, default=None
	figure size
date_flag: bool, default=False
	whether it is checking date features
"""
def numeric_consist_pretty(_df1, _df2, _key1, _key2, col, figsize=None, date_flag=False):

	# check _df1
	if type(_df1) != pd.core.frame.DataFrame:
		raise ValueError('_df1: only accept pandas DataFrame')

	# check _df2
	if type(_df2) != pd.core.frame.DataFrame:
		raise ValueError('_df2: only accept pandas DataFrame')

	# check whether keys are valid
	if not _key1 in _df1.columns.values:
		raise ValueError('_key1: does not exist in df1')
	if not _key2 in _df2.columns.values:
		raise ValueError('_key2: does not exist in df2')

	# check whether two tables are unique in key level
	if (_df1[_key1].nunique() != _df1.shape[0]):
		raise ValueError('_df1: should be unique in %s level' %(_key1))
	if (_df2[_key2].nunique() != _df2.shape[0]):
		raise ValueError('_df2: should be unique in %s level' %(_key2))

	# check col
	if type(col) != str:
		raise ValueError('col: only accept string')
	if col not in _df1.columns.values:
		raise ValueError('col: column not in df1')
	if col not in _df2.columns.values:
		raise ValueError('col: column not in df2')

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

	# check date_flag
	if type(date_flag) != bool:
		raise ValueError('date_flag: only accept boolean values')

	# color values for graph
	TABLE1_DARK = "#4BACC6"
	TABLE1_LIGHT = "#DAEEF3"

	TABLE2_DARK = "#F79646"
	TABLE2_LIGHT = "#FDE9D9"

	df1, df2 = _df1.copy(), _df2.copy()
	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	if ((df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0)):
		print("All values are nan in one of the 2 tables.")

	corr = round(pearsonr(df['%s_x' %(col)].values, df['%s_y' %(col)].values)[0], 3)

	df['diff_temp'] = df['%s_x' %(col)] - df['%s_y' %(col)]
	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	both_min = np.min([df['%s_x' %(col)].min(), df['%s_y' %(col)].min()])
	both_max = np.max([df['%s_x' %(col)].max(), df['%s_y' %(col)].max()])

	# get distribution
	scale_flg = 0
	if np.max(abs(draw_values)) >= 1000000:
		scale_flg = 1
		signs = np.sign(draw_values)

		draw_values = signs * np.log10(abs(draw_values) + 1)
		draw_value_4_signs = [np.sign(dv) for dv in origin_value_4]
		draw_value_4_scale = [np.log10(abs(dv) + 1) for dv in origin_value_4]
		draw_value_4 = [draw_value_4_signs[i] * draw_value_4_scale[i] for i in range(4)]
	else:
		draw_value_4 = origin_value_4

	# draw the scatter plot
	plt.figure(figsize=(12, 6))

	plt.subplot(121)
	plt.scatter(df['%s_x' %(col)].values, df['%s_y' %(col)].values, c=TABLE1_DARK, s=15)
	plt.plot([both_min, both_max], [both_min, both_max], '--', c='#bbbbbb')

	plt.xlim(both_min, both_max)
	plt.ylim(both_min, both_max)

	plt.title('corr: %.3f' %(corr))

	ax2 = plt.subplot(122)
	sns.distplot(draw_values, color=TABLE2_DARK)
	plt.axvline(x=draw_value_4[0], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[1], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[2], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[3], color=TABLE1_DARK, linestyle='--', linewidth=1.5)

	y_low, y_up = ax2.get_ylim()

	plt.text(draw_value_4[0], y_low + (y_up - y_low) * 0.2, 'min:' + str(round(origin_value_4[0], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[1], y_low + (y_up - y_low) * 0.4, 'mean:' + str(round(origin_value_4[1], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[2], y_low + (y_up - y_low) * 0.6, 'median:' + str(round(origin_value_4[2], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[3], y_low + (y_up - y_low) * 0.8, 'max:' + str(round(origin_value_4[3], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif date_flag:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	plt.show()


"""
function: check consistency for numeric features in both tables
parameters:
col: string
	column name of the numeric feature
_df1: pandas DataFrame
	slice of table1 containing enough information to compare
_df2: pandas DataFrame
	slice of table2 containing enough information to compare
_key1: string
	key for table1
_key2: string
	key for table2
img_dir: string
	directory for the generated images
date_flag: bool, default=False
	whether it is comparing date features
"""
def _consist_numeric(col, _df1, _df2, _key1, _key2, img_dir, date_flag=False):

	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	if ((df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0)):
		return col

	corr = round(pearsonr(df['%s_x' %(col)].values, df['%s_y' %(col)].values)[0], 3)

	df['diff_temp'] = df['%s_x' %(col)] - df['%s_y' %(col)]
	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	both_min = np.min([df['%s_x' %(col)].min(), df['%s_y' %(col)].min()])
	both_max = np.max([df['%s_x' %(col)].max(), df['%s_y' %(col)].max()])

	# get distribution
	scale_flg = 0
	if np.max(abs(draw_values)) >= 1000000:
		scale_flg = 1
		signs = np.sign(draw_values)

		draw_values = signs * np.log10(abs(draw_values) + 1)
		draw_value_4_signs = [np.sign(dv) for dv in origin_value_4]
		draw_value_4_scale = [np.log10(abs(dv) + 1) for dv in origin_value_4]
		draw_value_4 = [draw_value_4_signs[i] * draw_value_4_scale[i] for i in range(4)]
	else:
		draw_value_4 = origin_value_4

	# draw the scatter plot
	plt.figure(figsize=(12, 6))

	plt.subplot(121)
	plt.scatter(df['%s_x' %(col)].values, df['%s_y' %(col)].values, c=TABLE1_DARK, s=15)
	plt.plot([both_min, both_max], [both_min, both_max], '--', c='#bbbbbb')

	plt.xlim(both_min, both_max)
	plt.ylim(both_min, both_max)

	plt.title('corr: %.3f' %(corr))

	ax2 = plt.subplot(122)
	sns.distplot(draw_values, color=TABLE2_DARK)
	plt.axvline(x=draw_value_4[0], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[1], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[2], color=TABLE1_DARK, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[3], color=TABLE1_DARK, linestyle='--', linewidth=1.5)

	y_low, y_up = ax2.get_ylim()

	plt.text(draw_value_4[0], y_low + (y_up - y_low) * 0.2, 'min:' + str(round(origin_value_4[0], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[1], y_low + (y_up - y_low) * 0.4, 'mean:' + str(round(origin_value_4[1], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[2], y_low + (y_up - y_low) * 0.6, 'median:' + str(round(origin_value_4[2], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	plt.text(draw_value_4[3], y_low + (y_up - y_low) * 0.8, 'max:' + str(round(origin_value_4[3], 3)),
		ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif date_flag:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	# save the graphs
	plt.savefig(os.path.join(img_dir, col + '.png'), transparent=True)


"""
function: check consistency for string features in both tables
parameters:
col: string
	column name of the string feature
_df1: pandas DataFrame
	slice of table1 containing enough information to compare
_df2: pandas DataFrame
	slice of table2 containing enough information to compare
_key1: string
	key for table1
_key2: string
	key for table2
img_dir: string
	directory for the generated images
"""
def _consist_string(col, _df1, _df2, _key1, _key2, img_dir):

	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	if ((df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0)):
		return col

	df['diff_temp'] = df[['%s_x' %(col), '%s_y' %(col)]].apply(lambda x: "Same" 
		if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)

	corr = df[df['diff_temp'] == "Same"].shape[0] * 1.0 / df.shape[0]

	pie_values = [df[df['diff_temp'] == 'Same'].shape[0], df[df['diff_temp'] == 'Diff'].shape[0]]
	pie_labels = ['Same: %.3f' %(pie_values[0] * 1.0 / np.sum(pie_values)), 
				  'Diff: %.3f' %(pie_values[1] * 1.0 / np.sum(pie_values))]
	pie_colors = [TABLE1_LIGHT, TABLE2_LIGHT]

	plt.figure(figsize=(12, 6))
	plt.subplot(121)
	plt.pie(x=pie_values, labels=pie_labels, colors=pie_colors, radius=0.6)
	plt.title('consist rate: %.3f' %(corr))
	
	if np.max([df['%s_x' %(col)].nunique(), df['%s_y' %(col)].nunique()]) <= 20:
		plt.subplot(122)
		sankey.sankey(left=df['%s_x' %(col)], right=df['%s_y' %(col)], fontsize=10,
						leftLabels=sorted(df['%s_x' %(col)].unique()), 
						rightLabels=sorted(df['%s_y' %(col)].unique()), aspect=2)

		plt.title('corr: %.3f' %(corr))

	# save the graphs
	plt.savefig(os.path.join(img_dir, col + '.png'), transparent=True)


"""
function: write results to worksheet
parameters:
features: list
	all checked features
error_features: list
	all features with errors
ws: excel worksheet
	worksheet to write on
col_height: integer
	height of column for this worksheet
img_dir: string
	directory for the generated images
"""
def _insert_consist_results(features, error_features, ws, col_height, img_dir):
	# construct the thick border
	thick = Side(border_style="thick", color="000000")
	border = Border(top=thick, left=thick, right=thick, bottom=thick)

	# loop and output the results
	for feat in features:
		ws.append([feat])
		ws["A%d" %(ws.max_row)].style = 'Accent5'
		ws.row_dimensions[ws.max_row].height = 30

		if feat in error_features:
			ws.append(['column error: values are all nan in one of the tables.'])
		else:
			img = openpyxl.drawing.image.Image(os.path.join(img_dir, '%s.png' %(feat)))
			ws.add_image(img, 'A%d' %(ws.max_row+1))
			ws.row_dimensions[ws.max_row+1].height = 320

		# draw the thick outline border
		_style_range(ws, 'A%d:A%d'%(ws.max_row, ws.max_row + 1), border=border)
		
		# add gap
		ws.append([''])
		ws.append([''])

	ws.column_dimensions['A'].width = 125


"""
function: check consistency for same columns between two tables
parameters:
_table1: pandas DataFrame
	one of the two tables to compare
_table2: pandas DataFrame
	one of the two tables to compare
_key1: string
	key for table1
_key2: string
	key for table2
_schema1: pandas DataFrame
	data schema (contains column names and corresponding data types) for _table1
_schema2: pandas DataFrame
	data schema (contains column names and corresponding data types) for _table2
fname: string
	the output file name
sample_size: integer or float(<=1.0), default=1.0
	int: number of sample rows to do the comparison (useful for large tables)
	float: sample size in percentage
feature_colname1: string, default='column'
	name of the column for feature of _table1
feature_colname2: string, default='column'
	name of the column for feature of _table2
dtype_colname1: string, default='type'
	name of the column for data type of _table1
dtype_colname2: string, default='type'
	name of the column for data type of _table2
output_root: string, default=''
	the root directory for the output file
n_jobs: int, default=1
	the number of jobs to run in parallel
"""
def data_consist(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, sample_size=1.0, feature_colname1='column', 
	feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root='', n_jobs=1):

	# check _table1 and _table2
	if type(_table1) != pd.core.frame.DataFrame:
		raise ValueError('_table1: only accept pandas DataFrame')
	if type(_table2) != pd.core.frame.DataFrame:
		raise ValueError('_table2: only accept pandas DataFrame')

	# check whether keys are valid
	if not _key1 in _table1.columns.values:
		raise ValueError('_key1: does not exist in table1')
	if not _key2 in _table2.columns.values:
		raise ValueError('_key2: does not exist in table2')

	# check whether two tables are unique in key level
	if (_table1[_key1].nunique() != _table1.shape[0]):
		raise ValueError('_table1: should be unique in %s level' %(_key1))
	if (_table2[_key2].nunique() != _table2.shape[0]):
		raise ValueError('_table2: should be unique in %s level' %(_key2))

	# check _schema1 and _schema2
	if type(_schema1) != pd.core.frame.DataFrame:
		raise ValueError('_schema1: only accept pandas DataFrame')
	if type(_schema2) != pd.core.frame.DataFrame:
		raise ValueError('_schema2: only accept pandas DataFrame')

	schema1_dtypes = np.unique(_schema1[dtype_colname1].values)
	if not set(schema1_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema1: data types should be one of ['key', 'date', 'str', 'numeric']")
	schema2_dtypes = np.unique(_schema2[dtype_colname2].values)
	if not set(schema2_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema2: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check sample_size
	if (type(sample_size) != int) and (type(sample_size) != float):
		raise ValueError('sample_size: only accept integer or float value')
	if sample_size > 1:
		if int(sample_size) != sample_size:
			raise ValueError('sample_size: only accept integer when it is > 1.0')
		if (sample_size > _table1.shape[0]) or (sample_size > _table2.shape[0]):
			print('sample_size: %d is smaller than %d or %d...' %(sample_size, _table1.shape[0], _table2.shape[0]))
	else:
		if sample_size <= 0:
			raise ValueError('sample_size: should be larger than 0')

	# check fname
	if type(fname) != str:
		raise ValueError('fname: only accept string')

	# check feature_colname1 and feature_colname2
	if type(feature_colname1) != str:
		raise ValueError('feature_colname1: only accept string value')
	if not feature_colname1 in _schema1.columns.values:
		raise ValueError('feature_colname1: column not in _schema1')

	if type(feature_colname2) != str:
		raise ValueError('feature_colname2: only accept string value')
	if not feature_colname2 in _schema2.columns.values:
		raise ValueError('feature_colname2: column not in _schema2')

	# check dtype_colname1 and dtype_colname2
	if type(dtype_colname1) != str:
		raise ValueError('dtype_colname1: only accept string value')
	if not dtype_colname1 in _schema1.columns.values:
		raise ValueError('dtype_colname1: column not in _schema1')

	if type(dtype_colname2) != str:
		raise ValueError('dtype_colname2: only accept string value')
	if not dtype_colname2 in _schema2.columns.values:
		raise ValueError('dtype_colname2: column not in _schema2')

	# check output_root
	if output_root != '':
		if type(output_root) != str:
			raise ValueError('output_root: only accept string')
		if not os.path.isdir(output_root):
			raise ValueError('output_root: root not exists')

	# check n_jobs
	if type(n_jobs) != int:
		raise ValueError('n_jobs: only accept integer value') 

	# start to compare with correct schemas
	# create a new workbook to store everything
	wb = openpyxl.Workbook()

	# prepare directory for generated images
	img_dir = 'img_temp'
	if os.path.isdir(img_dir):
		shutil.rmtree(img_dir)
	os.mkdir(img_dir)

	# copy data tables
	table1 = _table1.copy()
	table2 = _table2.copy()

	# calculate the sample size
	if sample_size <= 1.0:
		both_keys = list(set(table1[_key1].values).intersection(set(table2[_key2].values)))
		sample_size = np.min([int(table1.shape[0] * sample_size), int(table2.shape[0] * sample_size), len(both_keys)])
		sample_keys = np.random.choice(both_keys, sample_size, replace=False)
		table1 = table1[table1[_key1].isin(sample_keys)].reset_index(drop=True)
		table2 = table2[table2[_key2].isin(sample_keys)].reset_index(drop=True)

	# copy both schema
	schema1 = _schema1.copy()[[feature_colname1, dtype_colname1]].rename(columns={feature_colname1: 'column_1', dtype_colname1: 'type_1'})
	schema2 = _schema2.copy()[[feature_colname2, dtype_colname2]].rename(columns={feature_colname2: 'column_2', dtype_colname2: 'type_2'})

	# merge two schemas
	schema = schema1.merge(schema2, left_on='column_1', right_on='column_2', how='outer')

	# if data types are different in schema1 and schema2, move to error
	schema_error = schema[schema['type_1'] != schema['type_2']].reset_index(drop=True)
	schema_error['error'] = "inconsistent data types"
	schema_error.loc[schema_error['column_1'].isnull(), 'error'] = "column not in table1"
	schema_error.loc[schema_error['column_2'].isnull(), 'error'] = "column not in table2"
	schema_correct = schema[schema['type_1'] == schema['type_2']].reset_index(drop=True)

	# classify the features to compare
	key_features = schema_correct[schema_correct['type_1'] == 'key']['column_1'].values
	numeric_features = schema_correct[schema_correct['type_1'] == 'numeric']['column_1'].values
	string_features = schema_correct[schema_correct['type_1'] == 'str']['column_1'].values
	date_features = schema_correct[schema_correct['type_1'] == 'date']['column_1'].values

	# for key features
	# only check features in both tables
	key_features = [feat for feat in key_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(key_features) > 0:
		key_results = Parallel(n_jobs=n_jobs)(delayed(_compare_key)(col, table1[[col]], table2[[col]], img_dir) 
			for col in key_features)
		# write all results to worksheet
		ws = wb.create_sheet(title='key')
		_insert_compare_results(key_results, ws, 40, img_dir)


	# for numeric features
	# only check features in both tables
	numeric_features = [feat for feat in numeric_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(numeric_features) > 0:
		error_numeric_cols = Parallel(n_jobs=n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]], table2[[_key2, col]], _key1, _key2, img_dir) 
			for col in numeric_features)
		# write all results to worksheet
		ws = wb.create_sheet(title='numeric')
		_insert_consist_results(numeric_features, error_numeric_cols, ws, 40, img_dir)


	# for string features
	# only check features in both tables
	string_features = [feat for feat in string_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(string_features) > 0:
		error_string_cols = Parallel(n_jobs=n_jobs)(delayed(_consist_string)(col, table1[[_key1, col]], table2[[_key2, col]], _key1, _key2, img_dir) 
			for col in string_features)
		# write all results to worksheet
		ws = wb.create_sheet(title='string')
		_insert_consist_results(string_features, error_string_cols, ws, 40, img_dir)


	# for date features
	# only check features in both tables
	date_features = [feat for feat in date_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(date_features) > 0:
		# get the current time
		snapshot_date_now = str(datetime.datetime.now().date())
		for col in date_features:
			table1[col] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(table1[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
			table2[col] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(table2[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
		error_date_cols = Parallel(n_jobs=n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]], table2[[_key2, col]], 
			_key1, _key2, img_dir, date_flag=True) for col in date_features)
		# write all results to worksheet
		ws = wb.create_sheet(title='date')
		_insert_consist_results(date_features, error_date_cols, ws, 40, img_dir)


	# insert error
	ws = wb['Sheet']
	wb.remove(ws)

	# if there are some errors
	if len(schema_error) > 0:
		ws = wb.create_sheet(title='error')

		for r_idx, r in enumerate(dataframe_to_rows(schema_error, index=False, header=True)):
			ws.append(r)
			for col in ws.iter_cols(max_col=ws.max_column, min_row=ws.max_row, max_row=ws.max_row):
				for cell in col:
					if r_idx == 0:
						cell.font = Font(name='Calibri', size=11, bold=True)
					else:
						cell.font = Font(name='Calibri', size=11)

	wb.save(filename=os.path.join(output_root, 'data_consist_%s.xlsx' %(fname)))
	shutil.rmtree(img_dir)


"""
function: automatically generate ipynb for data consistency check
parameters:
_table1: pandas DataFrame
	one of the two tables to compare
_table2: pandas DataFrame
	one of the two tables to compare
_key1: string
	key for table1
_key2: string
	key for table2
_schema1: pandas DataFrame
	data schema (contains column names and corresponding data types) for _table1
_schema2: pandas DataFrame
	data schema (contains column names and corresponding data types) for _table2
fname: string
	the output file name
sample: boolean, default=False
	whether to do sampling on the original data
feature_colname1: string, default='column'
	name of the column for feature of _table1
feature_colname2: string, default='column'
	name of the column for feature of _table2
dtype_colname1: string, default='type'
	name of the column for data type of _table1
dtype_colname2: string, default='type'
	name of the column for data type of _table2
output_root: string, default=''
	the root directory for the output file
"""
def data_consist_notebook(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, sample=False, feature_colname1='column', feature_colname2='column', 
	dtype_colname1='type', dtype_colname2='type', output_root=''):

	# check _table1 and _table2
	if type(_table1) != pd.core.frame.DataFrame:
		raise ValueError('_table1: only accept pandas DataFrame')
	if type(_table2) != pd.core.frame.DataFrame:
		raise ValueError('_table2: only accept pandas DataFrame')

	# check whether keys are valid
	if not _key1 in _table1.columns.values:
		raise ValueError('_key1: does not exist in table1')
	if not _key2 in _table2.columns.values:
		raise ValueError('_key2: does not exist in table2')

	# check whether two tables are unique in key level
	if (_table1[_key1].nunique() != _table1.shape[0]):
		raise ValueError('_table1: should be unique in %s level' %(_key1))
	if (_table2[_key2].nunique() != _table2.shape[0]):
		raise ValueError('_table2: should be unique in %s level' %(_key2))

	# check _schema1 and _schema2
	if type(_schema1) != pd.core.frame.DataFrame:
		raise ValueError('_schema1: only accept pandas DataFrame')
	if type(_schema2) != pd.core.frame.DataFrame:
		raise ValueError('_schema2: only accept pandas DataFrame')

	schema1_dtypes = np.unique(_schema1[dtype_colname1].values)
	if not set(schema1_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema1: data types should be one of ['key', 'date', 'str', 'numeric']")
	schema2_dtypes = np.unique(_schema2[dtype_colname2].values)
	if not set(schema2_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema2: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check sample
	if type(sample) != bool:
		raise ValueError('sample: only accept boolean values')

	# check fname
	if type(fname) != str:
		raise ValueError('fname: only accept string')

	# check feature_colname1 and feature_colname2
	if type(feature_colname1) != str:
		raise ValueError('feature_colname1: only accept string value')
	if not feature_colname1 in _schema1.columns.values:
		raise ValueError('feature_colname1: column not in _schema1')

	if type(feature_colname2) != str:
		raise ValueError('feature_colname2: only accept string value')
	if not feature_colname2 in _schema2.columns.values:
		raise ValueError('feature_colname2: column not in _schema2')

	# check dtype_colname1 and dtype_colname2
	if type(dtype_colname1) != str:
		raise ValueError('dtype_colname1: only accept string value')
	if not dtype_colname1 in _schema1.columns.values:
		raise ValueError('dtype_colname1: column not in _schema1')

	if type(dtype_colname2) != str:
		raise ValueError('dtype_colname2: only accept string value')
	if not dtype_colname2 in _schema2.columns.values:
		raise ValueError('dtype_colname2: column not in _schema2')

	# check output_root
	if output_root != '':
		if type(output_root) != str:
			raise ValueError('output_root: only accept string')
		if not os.path.isdir(output_root):
			raise ValueError('output_root: root not exists')

	# generate output file path 
	output_path = os.path.join(output_root, 'data_consist_notebook_%s.py' %(fname))

	# delete potential generated script and notebook
	if os.path.isfile(output_path):
		os.remove(output_path)

	if os.path.isfile(output_path.replace('.py', '.ipynb')):
		os.remove(output_path.replace('.py', '.ipynb'))

	# copy both schema
	schema1 = _schema1.copy()[[feature_colname1, dtype_colname1]].rename(columns={feature_colname1: 'column_1', dtype_colname1: 'type_1'})
	schema2 = _schema2.copy()[[feature_colname2, dtype_colname2]].rename(columns={feature_colname2: 'column_2', dtype_colname2: 'type_2'})

	# merge two schemas
	schema = schema1.merge(schema2, left_on='column_1', right_on='column_2', how='outer')

	# if data types are different in schema1 and schema2, move to error
	schema_error = schema[schema['type_1'] != schema['type_2']].reset_index(drop=True)
	schema_error['error'] = "inconsistent data types"
	schema_error.loc[schema_error['column_1'].isnull(), 'error'] = "column not in table1"
	schema_error.loc[schema_error['column_2'].isnull(), 'error'] = "column not in table2"
	schema_correct = schema[schema['type_1'] == schema['type_2']].reset_index(drop=True)

	with open(output_path, "a") as outbook:
		# import packages
		outbook.write('\n"""\n')
		outbook.write('## import useful packages\n\n')
		outbook.write('"""\n\n')
		
		packages = ['import pandas as pd', 'import numpy as np', '\nimport datetime',
		'\nfrom scipy.stats import pearsonr\n', 'import matplotlib.pyplot as plt', 
		'import seaborn as sns', 'sns.set_style("white")', 'from matplotlib_venn import venn2','\n%matplotlib inline', 
		'\nimport sankey', '\nfrom pydqc.data_consist import numeric_consist_pretty']

		outbook.write('\n'.join(packages))

		# assign value to table
		outbook.write('\n"""\n')
		outbook.write('## assign values\n\n')
		outbook.write('"""\n\n')

		outbook.write('#the data table (pandas DataFrame)\n')
		outbook.write('table1 = \n')
		outbook.write('table2 = \n')
		outbook.write('print("table1 size: " + str(table1.shape))\n')
		outbook.write('print("table2 size: " + str(table2.shape))\n\n')

		outbook.write('key1 = \n')
		outbook.write('key2 = \n\n')

		if sample:
			outbook.write('#the sample size (can be integer or float <= 1.0)\n')
			outbook.write('sample_size =\n\n')

		outbook.write('#global values\n')
		outbook.write('TABLE1_DARK = "#4BACC6"\n')
		outbook.write('TABLE1_LIGHT = "#DAEEF3"\n')
		outbook.write('TABLE2_DARK = "#F79646"\n')
		outbook.write('TABLE2_LIGHT = "#FDE9D9"\n\n')

		outbook.write('#get date of today\n')
		outbook.write('snapshot_date_now = str(datetime.datetime.now().date())\n')
		outbook.write('print("date of today: " + snapshot_date_now)\n')

		# check and calculate sample size if sample=True
		if sample:
			outbook.write('\n"""\n')
			outbook.write('## calculate the sample size\n\n')
			outbook.write('"""\n\n')
			outbook.write('if sample_size <= 1.0:\n')
			outbook.write('    sample_size1 = int(table1.shape[0] * sample_size)\n')
			outbook.write('    sample_size2 = int(table2.shape[0] * sample_size)\n')
			outbook.write('    sample_size = np.min([sample_size1, sample_size2])\n')
			outbook.write('    print(sample_size)\n')
			outbook.write('else:\n')
			outbook.write('    if sample_size > np.min([table1.shape[0], table2.shape[0]]):\n')
			outbook.write('        raise ValueError("sample_size: should be smaller or equal to len(table1) and len(table2)")\n')

		# output potentail exist errors
		if len(schema_error) > 0:
			outbook.write('\n"""\n')
			outbook.write('### inconsistent columns between table1 and table2:\n\n')
			schema_error_dicts = schema_error.to_dict('record')
			for i in range(len(schema_error_dicts)):
				outbook.write('%s\n' %(schema_error_dicts[i]))
			outbook.write('"""\n\n')
		else:
			outbook.write('\n"""\n')
			outbook.write('### columns are consistent between table1 and table2!\n\n')
			outbook.write('"""\n\n')

		# only compare check columns in both table1 and table2, and follow the column order of table1
		check_cols = [col for col in _table1.columns.values if col in schema_correct['column_1'].values]
		for col in check_cols:
			# get the data type of the column
			col_type = schema_correct[schema_correct['column_1']==col]['type_1'].values[0]

			outbook.write('\n"""\n')
			outbook.write('## %s (type: %s)\n\n' %(col, col_type))
			outbook.write('"""\n\n')

			outbook.write('col="%s"\n' %(col))
			if (sample) and (col_type != 'key'):
				if col in [_key1, _key2]:
					outbook.write('df1 = table1[[col]].copy().sample(sample_size).reset_index(drop=True)\n')
					outbook.write('df2 = table2[[col]].copy().sample(sample_size).reset_index(drop=True)\n\n')
				else:
					outbook.write('df1 = table1[[key1, col]].copy().sample(sample_size).reset_index(drop=True)\n')
					outbook.write('df2 = table2[[key2, col]].copy().sample(sample_size).reset_index(drop=True)\n\n')
			else:
				if col in [_key1, _key2]:
					outbook.write('df1 = table1[[col]].copy()\n')
					outbook.write('df2 = table2[[col]].copy()\n\n')
				else:
					outbook.write('df1 = table1[[key1, col]].copy()\n')
					outbook.write('df2 = table2[[key2, col]].copy()\n\n')
				if col_type == 'date':
					outbook.write('df1[col] = pd.to_datetime(df1[col], errors="coerce")\n')
					outbook.write('df1[col + "_numeric"] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df1[col], errors="coerce")).astype("timedelta64[M]", errors="ignore")\n\n')
					outbook.write('df2[col] = pd.to_datetime(df2[col], errors="coerce")\n')
					outbook.write('df2[col + "_numeric"] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df2[col], errors="coerce")).astype("timedelta64[M]", errors="ignore")\n\n')

			# for key and str, compare intersection
			if col_type == 'key':
				outbook.write('\n"""\n')
				outbook.write('#### compare intersection\n\n')
				outbook.write('"""\n\n')
				outbook.write('#nan_rate\n')
				outbook.write('nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]\n')
				outbook.write('nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]\n')
				outbook.write('print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))\n\n')

				outbook.write('set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()\n')
				outbook.write('set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()\n')
				outbook.write('col_overlap = len(set_df1_col.intersection(set_df2_col))\n')
				outbook.write('col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)\n\n')
				outbook.write('print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))\n')

				outbook.write('\n"""\n')
				outbook.write('#### draw venn graph\n\n')
				outbook.write('"""\n\n')
				outbook.write('plt.figure(figsize=(10, 5))\n')
				outbook.write('venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"], set_colors=("#4BACC6", "#F79646"), alpha=0.8)\n')
			elif col_type == 'str':
				outbook.write('\n"""\n')
				outbook.write('#### check pairwise consistency\n\n')
				outbook.write('"""\n\n')
				outbook.write('# merge 2 tables\n')
				outbook.write('df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")\n')
				outbook.write('if ((df[col + "_x"].dropna().shape[0] == 0) or (df[col + "_y"].dropna().shape[0] == 0)):\n')
				outbook.write('    print("All values are nan in one of the 2 tables.")\n\n')

				outbook.write('# calculate consistency\n')
				outbook.write('df["diff_temp"] = df[[col + "_x", col + "_y"]].apply(lambda x: "Same" if x[col + "_x"] == x[col + "_y"] else "Diff", axis=1)\n')
				outbook.write('corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]\n\n')

				outbook.write('\n"""\n')
				outbook.write('#### draw graphs\n\n')
				outbook.write('"""\n\n')

				outbook.write('# prepare data\n')
				outbook.write('pie_values = [df[df["diff_temp"] == "Same"].shape[0], df[df["diff_temp"] == "Diff"].shape[0]]\n')
				outbook.write('pie_labels = ["Same: %.3f" %(pie_values[0] * 1.0 / np.sum(pie_values)), "Diff: %.3f" %(pie_values[1] * 1.0 / np.sum(pie_values))]\n')
				outbook.write('pie_colors = [TABLE1_LIGHT, TABLE2_LIGHT]\n\n')

				outbook.write('# draw\n')
				outbook.write('plt.figure(figsize=(12, 6))\n')
				outbook.write('plt.subplot(121)\n')
				outbook.write('plt.pie(x=pie_values, labels=pie_labels, colors=pie_colors, radius=0.6)\n')
				outbook.write('plt.title("consist rate: %.3f" %(corr))\n\n')

				outbook.write('if np.max([df[col + "_x"].nunique(), df[col + "_y"].nunique()]) <= 20:\n')
				outbook.write('    plt.subplot(122)\n')
				outbook.write('    sankey.sankey(left=df[col + "_x"], right=df[col + "_y"], fontsize=10, leftLabels=sorted(df[col + "_x"].unique()), rightLabels=sorted(df[col + "_y"].unique()), aspect=2)\n')
				outbook.write('    plt.title("corr: %.3f" %(corr))\n')
			else:
				outbook.write('\n"""\n')
				outbook.write('#### check pairwise consistency\n\n')
				outbook.write('"""\n\n')
				outbook.write('# merge 2 tables\n')
				outbook.write('df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")\n')
				outbook.write('if ((df[col + "_x"].dropna().shape[0] == 0) or (df[col + "_y"].dropna().shape[0] == 0)):\n')
				outbook.write('    print("All values are nan in one of the 2 tables.")\n')
				outbook.write('corr = round(pearsonr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)\n\n')

				outbook.write('# prepare data\n')
				outbook.write('df["diff_temp"] = df[col + "_x"] - df[col + "_y"]\n')
				outbook.write('draw_values = df["diff_temp"].dropna().values\n\n')

				outbook.write('both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])\n')
				outbook.write('both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])\n\n')

				outbook.write('# draw\n')
				outbook.write('plt.figure(figsize=(12, 6))\n')
				outbook.write('plt.subplot(121)\n')
				outbook.write('plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=15)\n')
				outbook.write('plt.plot([both_min, both_max], [both_min, both_max], "--", c="#bbbbbb")\n')
				outbook.write('plt.xlim(both_min, both_max)\n')
				outbook.write('plt.ylim(both_min, both_max)\n')
				outbook.write('plt.title("corr: %.3f" %(corr))\n\n')
				outbook.write('plt.subplot(122)\n')
				outbook.write('sns.distplot(draw_values, color=TABLE2_DARK)\n')
				if col_type == 'date':
					outbook.write('plt.title("Distribution of differences (in months)")\n')
				else:
					outbook.write('plt.title("Distribution of differences")\n')
				outbook.write('\n"""\n')
				outbook.write('"""\n\n')
				outbook.write('#you can also use the build-in draw function\n')
				if col_type == 'date':
					outbook.write('numeric_consist_pretty(df1, df2, key1, key2, col, date_flag=True)\n')
				else:
					outbook.write('numeric_consist_pretty(df1, df2, key1, key2, col)\n')

		outbook.close()

	os.system("python -m py2nb %s %s" %(output_path, output_path.replace('.py', '.ipynb')))