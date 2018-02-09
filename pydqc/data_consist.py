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
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')
from matplotlib_venn import venn2

from dqc_utils import (
	_style_range, _get_scale_draw_values, _draw_texts, 
	_adjust_column, _insert_df, _insert_numeric_results
)
from data_compare import _compare_key
from data_summary import _insert_string_results


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
	df['diff_temp'] = df['%s_y' %(col)] - df['%s_x' %(col)]
	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	# get distribution
	scale_flg = 0
	draw_value_4 = origin_value_4
	if np.max([abs(origin_value_4[0]), abs(origin_value_4[3])]) >= 1000000:
		scale_flg = 1
		draw_values, draw_value_4 = _get_scale_draw_values(draw_values, draw_value_4)

	plt.clf()
	if figsize is not None:
		plt.figure(figsize)
	else:
		plt.figure(figsize=(9, 4))

	both_min = np.min([df['%s_x' %(col)].min(), df['%s_y' %(col)].min()])
	both_max = np.max([df['%s_x' %(col)].max(), df['%s_y' %(col)].max()])

	plt.subplot(121)
	plt.title('Scatter plot for values')
	plt.scatter(df['%s_x' %(col)].values, df['%s_y' %(col)].values, c=TABLE1_DARK, s=5)
	plt.plot([both_min, both_max], [both_min, both_max], '--', c='#bbbbbb')

	plt.xlim(both_min, both_max)
	plt.ylim(both_min, both_max)

	ax2 = plt.subplot(122)
	sns.distplot(draw_values, color=TABLE2_DARK)
	y_low, y_up = ax2.get_ylim()
	_draw_texts(draw_value_4, mark=1, text_values=origin_value_4, y_low=y_low, y_up=y_up)

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif date_flag:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	plt.show()


def _consist_numeric(col, _df1, _df2, _key1, _key2, img_dir, date_flag=False):

	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	if (df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0):
		if (df['%s_x' %(col)].dropna().shape[0] == 0) and (df['%s_y' %(col)].dropna().shape[0] == 0):
			error_msg = 'all nan in both table'
		elif df['%s_x' %(col)].dropna().shape[0] == 0:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	df = df.dropna(how='any')
	df['diff_temp'] = df['%s_y' %(col)] - df['%s_x' %(col)]
	df['diff_per_temp'] = df.apply(lambda x : (x['%s_y' %(col)] - x['%s_x' %(col)]) * 1.0 / x['%s_x' %(col)] 
		if x['%s_x' %(col)] != 0 else np.nan, axis=1)
	corr = round(spearmanr(df['%s_x' %(col)].values, df['%s_y' %(col)].values)[0], 3)

	output = [
		{'feature': 'column', 'value': col, 'graph': 'consistency check'},
		{'feature': 'corr', 'value': corr},
		{'feature': 'min diff%', 'value': round(df['diff_per_temp'].min() * 100, 3)},
		{'feature': 'mean diff%', 'value': round(df['diff_per_temp'].mean() * 100, 3)},
		{'feature': 'median diff%', 'value': round(df['diff_per_temp'].median() * 100, 3)},
		{'feature': 'max diff%', 'value': round(df['diff_per_temp'].max() * 100, 3)},
	]

	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	# get distribution
	scale_flg = 0
	draw_value_4 = origin_value_4
	if np.max([abs(origin_value_4[0]), abs(origin_value_4[3])]) >= 1000000:
		scale_flg = 1
		draw_values, draw_value_4 = _get_scale_draw_values(draw_values, draw_value_4)

	# draw the scatter plot
	both_min = np.min([df['%s_x' %(col)].min(), df['%s_y' %(col)].min()])
	both_max = np.max([df['%s_x' %(col)].max(), df['%s_y' %(col)].max()])

	plt.figure(figsize=(9, 4))
	plt.subplot(121)
	plt.title('Scatter plot for values')
	plt.scatter(df['%s_x' %(col)].values, df['%s_y' %(col)].values, c=TABLE1_DARK, s=5)
	plt.plot([both_min, both_max], [both_min, both_max], '--', c='#bbbbbb')

	plt.xlim(both_min, both_max)
	plt.ylim(both_min, both_max)

	ax2 = plt.subplot(122)
	sns.distplot(draw_values, color=TABLE2_DARK)
	y_low, y_up = ax2.get_ylim()
	_draw_texts(draw_value_4, mark=1, text_values=origin_value_4, y_low=y_low, y_up=y_up)

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif date_flag:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	# save the graphs
	plt.savefig(os.path.join(img_dir, col + '.png'), transparent=True)
	return {'column': col, 'result_df': pd.DataFrame(output), 'corr': {'column': col, 'corr': corr}}


def _consist_string(col, _df1, _df2, _key1, _key2, img_dir):

	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	if (df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0):
		if (df['%s_x' %(col)].dropna().shape[0] == 0) and (df['%s_y' %(col)].dropna().shape[0] == 0):
			error_msg = 'all nan in both table'
		elif df['%s_x' %(col)].dropna().shape[0] == 0:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	df = df.dropna(how='all', subset=['%s_x' %(col), '%s_y' %(col)])
	df['diff_temp'] = df[['%s_x' %(col), '%s_y' %(col)]].apply(lambda x: "Same" 
		if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)

	corr = round(df[df['diff_temp'] == "Same"].shape[0] * 1.0 / df.shape[0], 3)
	output = [
		{'feature': 'column', 'value': col},
		{'feature': 'corr', 'value': corr}
	]

	if corr == 1:
		return {'column': col, 'result_df': [pd.DataFrame(output), pd.DataFrame()], 'corr': {'column': col, 'corr': corr}}
	else:
		diff_df = df[df['diff_temp'] == "Diff"].reset_index(drop=True)
		diff_df['diff_combo'] = diff_df['%s_x' %(col)].map(str) + ' -> ' + diff_df['%s_y' %(col)].map(str)
		diff_df_vc = pd.DataFrame(diff_df['diff_combo'].value_counts())
		diff_df_vc.columns = ['count']
		diff_df_vc['diff_combo'] = diff_df_vc.index.values
		diff_df_vc = diff_df_vc.sort_values(by='count', ascending=False).head(10)[['diff_combo', 'count']]

		return {'column': col, 'result_df': [pd.DataFrame(output), diff_df_vc], 
				'corr': {'column': col, 'corr': corr}}


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
keep_images: boolean, default=False
	whether to keep all generated images
n_jobs: int, default=1
	the number of jobs to run in parallel
"""
def data_consist(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, sample_size=1.0, feature_colname1='column', 
	feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root='', keep_images=False, n_jobs=1):

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

	corr_results = []

	# for key features
	# only check features in both tables
	key_features = [feat for feat in key_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(key_features) > 0:
		key_results = Parallel(n_jobs=n_jobs)(delayed(_compare_key)(col, table1[[col]], table2[[col]], img_dir) 
			for col in key_features)

		for key_result in key_results:
			if 'corr' in key_result.keys():
				corr_results.append(key_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='key')
		_insert_numeric_results(key_results, ws, 40, img_dir)


	# for numeric features
	# only check features in both tables
	numeric_features = [feat for feat in numeric_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(numeric_features) > 0:
		numeric_results = Parallel(n_jobs=n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]], 
			table2[[_key2, col]], _key1, _key2, img_dir) for col in numeric_features)

		for numeric_result in numeric_results:
			if 'corr' in numeric_result.keys():
				corr_results.append(numeric_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='numeric')
		_insert_numeric_results(numeric_results, ws, 45, img_dir)


	# for string features
	# only check features in both tables
	string_features = [feat for feat in string_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(string_features) > 0:
		string_results = Parallel(n_jobs=n_jobs)(delayed(_consist_string)(col, table1[[_key1, col]], 
			table2[[_key2, col]], _key1, _key2, img_dir) for col in string_features)

		for string_result in string_results:
			if 'corr' in string_result.keys():
				corr_results.append(string_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='string')
		_insert_string_results(string_results, ws, 25)


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
		date_results = Parallel(n_jobs=n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]], table2[[_key2, col]], 
			_key1, _key2, img_dir, date_flag=True) for col in date_features)

		for date_result in date_results:
			if 'corr' in date_result.keys():
				corr_results.append(date_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='date')
		_insert_numeric_results(date_results, ws, 45, img_dir, date_flag=True)


	# insert the summary 
	ws = wb['Sheet']
	ws.title = 'summary'
	summary_df = schema_correct[['column_1', 'type_1']].rename(columns={'column_1': 'column', 'type_1': 'type'})
	corr_df = pd.DataFrame(corr_results)
	summary_df = summary_df.merge(corr_df, on='column', how='left')
	summary_df['corr'] = summary_df['corr'].fillna('error')
	summary_df['error_flg'] = summary_df['corr'].apply(lambda x : 1 if x == 'error' else 0)
	error_rows = summary_df[summary_df['error_flg'] == 1].index.values

	_ = _insert_df(summary_df[['column', 'type', 'corr']], ws, header=True)

	for r_idx in error_rows:
		ws['C%d' %(r_idx + 2)].style = 'Bad'
	_adjust_column(ws, 25)

	# if there are some errors
	if len(schema_error) > 0:
		ws = wb.create_sheet(title='error')
		_ = _insert_df(schema_error, ws, header=True)
		_adjust_column(ws, 25)

	wb.save(filename=os.path.join(output_root, 'data_consist_%s.xlsx' %(fname)))
	if not keep_images:
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
def data_consist_notebook(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, 
	feature_colname1='column', feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root=''):

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
		'\nfrom scipy.stats import spearmanr\n', 'import matplotlib.pyplot as plt', 
		'import seaborn as sns', 'sns.set_style("white")', 'from matplotlib_venn import venn2','\n%matplotlib inline', 
		'\nfrom pydqc.data_consist import numeric_consist_pretty']

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

		outbook.write('#global values\n')
		outbook.write('TABLE1_DARK = "#4BACC6"\n')
		outbook.write('TABLE1_LIGHT = "#DAEEF3"\n')
		outbook.write('TABLE2_DARK = "#F79646"\n')
		outbook.write('TABLE2_LIGHT = "#FDE9D9"\n\n')

		outbook.write('#get date of today\n')
		outbook.write('snapshot_date_now = str(datetime.datetime.now().date())\n')
		outbook.write('print("date of today: " + snapshot_date_now)\n')

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
				outbook.write('df = df.dropna(how="all", subset=["%s_x" %(col), "%s_y" %(col)])\n\n')

				outbook.write('# calculate consistency\n')
				outbook.write('df["diff_temp"] = df[[col + "_x", col + "_y"]].apply(lambda x: "Same" if x[col + "_x"] == x[col + "_y"] else "Diff", axis=1)\n')
				outbook.write('corr = df[df["diff_temp"] == "Same"].shape[0] * 1.0 / df.shape[0]\n')
				outbook.write('print("consistency rate: " + str(corr))\n\n')
			else:
				outbook.write('\n"""\n')
				outbook.write('#### check pairwise consistency\n\n')
				outbook.write('"""\n\n')
				outbook.write('# merge 2 tables\n')
				outbook.write('df = df1.merge(df2, left_on=key1, right_on=key2, how="inner")\n')
				outbook.write('df = df.dropna(how="any")\n')
				outbook.write('corr = round(spearmanr(df[col + "_x"].values, df[col + "_y"].values)[0], 3)\n')
				outbook.write('print("consistency rate: " + str(corr))\n\n')

				outbook.write('\n"""\n')
				outbook.write('#### draw consistency graph\n\n')
				outbook.write('"""\n\n')

				outbook.write('# prepare data\n')
				outbook.write('df["diff_temp"] = df[col + "_y"] - df[col + "_x"]\n')
				outbook.write('draw_values = df["diff_temp"].dropna().values\n\n')

				outbook.write('both_min = np.min([df[col + "_x"].min(), df[col + "_y"].min()])\n')
				outbook.write('both_max = np.max([df[col + "_x"].max(), df[col + "_y"].max()])\n\n')

				outbook.write('# draw\n')
				outbook.write('plt.figure(figsize=(12, 6))\n')
				outbook.write('plt.subplot(121)\n')
				outbook.write('plt.scatter(df[col + "_x"].values, df[col + "_y"].values, c=TABLE1_DARK, s=5)\n')
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