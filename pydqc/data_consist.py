import pandas as pd
import numpy as np
import os
import shutil

import openpyxl

from sklearn.externals.joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('white')

import datetime
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings('ignore')

from dqc_utils import (
	_get_scale_draw_values, _draw_texts,
	_adjust_ws, _insert_df, _insert_numeric_results
)
from data_compare import _compare_key
from data_summary import _insert_string_results

# global color values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"

TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"


def numeric_consist_pretty(_df1, _df2, _key1, _key2, col, figsize=None, date_flag=False):
	"""
	Draw pretty distribution graph for checking data consistency

	Parameters
	----------
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

	# color values for graph
	TABLE1_DARK = "#4BACC6"
	TABLE2_DARK = "#F79646"

	df = _df1.merge(_df2, left_on=_key1, right_on=_key2, how="inner")
	df['diff_temp'] = df['%s_y' %(col)] - df['%s_x' %(col)]
	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	# get distribution
	scale_flg = 0
	draw_value_4 = origin_value_4
	if np.max([abs(origin_value_4[0]), abs(origin_value_4[3])]) >= pow(10, 6):
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
	if len(np.unique(draw_values)) <= 10:
		sns.countplot(draw_values, palette=sns.color_palette([TABLE2_DARK]))
		if len(np.unique(draw_values)) > 5:
			plt.xticks(rotation=90)
	else:
		sns.distplot(draw_values, color=TABLE2_DARK)
		y_low, y_up = ax2.get_ylim()
		_draw_texts(text_values=origin_value_4, draw_value_4=draw_value_4, mark=1, y_low=y_low, y_up=y_up)

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif scale_flg:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	plt.show()


def _consist_numeric(col, _df1, _df2, _key1, _key2, img_dir, date_flag=False):
	"""
	Check consistency for numeric type column

	Parameters
	----------
	col: string
		name of column to check
	_df1: pandas DataFrame
		slice of table1 containing enough information to check
	_df2: pandas DataFrame
		slice of table2 containing enough information to check
	_key1: column to merge on for table1
	_key2: column to merge on for table2
	img_dir: root directory for the generated images
	date_flag: boolean
		Whether the column is date type

	Returns
	-------
	Dictionary contains the output result
	"""

	df1, df2 = _df1.copy(), _df2.copy()
	df = pd.merge(df1, df2, left_on=_key1, right_on=_key2, how="inner")

	if (df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0):
		if (df['%s_x' %(col)].dropna().shape[0] == 0) and (df['%s_y' %(col)].dropna().shape[0] == 0):
			error_msg = 'all nan in both table'
		elif df['%s_x' %(col)].dropna().shape[0] == 0:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	df = df.dropna(how='any', subset=['%s_x' % (col), '%s_y' % (col)]).reset_index(drop=True)
	df['diff_temp'] = df['%s_y' %(col)] - df['%s_x' %(col)]
	corr = round(spearmanr(df['%s_x' %(col)].values, df['%s_y' %(col)].values)[0], 3)

	output = [
		{'feature': 'column', 'value': col, 'graph': 'consistency check'},
		{'feature': 'corr', 'value': corr},
		{'feature': 'min diff', 'value': round(df['diff_temp'].min(), 3)},
		{'feature': 'mean diff', 'value': round(df['diff_temp'].mean(), 3)},
		{'feature': 'median diff', 'value': round(df['diff_temp'].median(), 3)},
		{'feature': 'max diff', 'value': round(df['diff_temp'].max(), 3)},
	]

	draw_values = df['diff_temp'].dropna().values
	origin_value_4 = [np.min(draw_values), np.mean(draw_values), np.median(draw_values), np.max(draw_values)]

	# get distribution
	scale_flg = 0
	draw_value_4 = origin_value_4
	if np.max([abs(origin_value_4[0]), abs(origin_value_4[3])]) >= pow(10, 6):
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
	if len(np.unique(draw_values)) <= 10:
		sns.countplot(draw_values, palette=sns.color_palette([TABLE2_DARK]))
		if len(np.unique(draw_values)) > 5:
			plt.xticks(rotation=90)
	else:
		sns.distplot(draw_values, color=TABLE2_DARK)
		y_low, y_up = ax2.get_ylim()
		_draw_texts(text_values=origin_value_4, draw_value_4=draw_value_4, mark=1, y_low=y_low, y_up=y_up)

	if date_flag:
		plt.title('Distribution of differences (in months)')
	elif scale_flg:
		plt.title('Distribution of differences (log10 scale)')
	else:
		plt.title('Distribution of differences')

	# save the graphs
	# adjust graph name
	graph_name = col
	if ('/' in graph_name) or ('\\' in graph_name):
		graph_name = '(%s)' % (graph_name)
	plt.savefig(os.path.join(img_dir, graph_name + '.png'), transparent=True)
	return {'column': col, 'result_df': pd.DataFrame(output), 'corr': {'column': col, 'corr': corr}}


def _consist_string(col, _df1, _df2, _key1, _key2):
	"""
	Check consistency for string type column

	Parameters
	----------
	col: string
		name of column to check
	_df1: pandas DataFrame
		slice of table1 containing enough information to check
	_df2: pandas DataFrame
		slice of table2 containing enough information to check
	_key1: column to merge on for table1
	_key2: column to merge on for table2

	Returns
	-------
	Dictionary contains the output result
	"""

	df1, df2 = _df1.copy(), _df2.copy()
	df = pd.merge(df1, df2, left_on=_key1, right_on=_key2, how="inner")

	if (df['%s_x' %(col)].dropna().shape[0] == 0) or (df['%s_y' %(col)].dropna().shape[0] == 0):
		if (df['%s_x' %(col)].dropna().shape[0] == 0) and (df['%s_y' %(col)].dropna().shape[0] == 0):
			error_msg = 'all nan in both table'
		elif df['%s_x' %(col)].dropna().shape[0] == 0:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	df['diff_temp'] = df.apply(lambda x: "Same" if x['%s_x' %(col)] == x['%s_y' %(col)] else "Diff", axis=1)
	df['diff_temp'] = df.apply(lambda x: "Same" if (str(x['%s_x' % (col)]) == 'nan'
													and str(x['%s_y' % (col)]) == 'nan') else x['diff_temp'], axis=1)

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


def data_consist(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, sample_size=1.0, feature_colname1='column', 
	feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root='', keep_images=False, n_jobs=1):
	"""
	Check consistency between two tables

	Parameters
	----------
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

	# check whether keys are valid
	if _key1 not in _table1.columns.values:
		raise ValueError('_key1: does not exist in table1')
	if _key2 not in _table2.columns.values:
		raise ValueError('_key2: does not exist in table2')

	# check whether two tables are unique in key level
	if (_table1[_key1].nunique() != _table1.shape[0]):
		raise ValueError('_table1: should be unique in %s level' % (_key1))
	if (_table2[_key2].nunique() != _table2.shape[0]):
		raise ValueError('_table2: should be unique in %s level' % (_key2))

	schema1_dtypes = np.unique(_schema1[dtype_colname1].values)
	if not set(schema1_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema1: data types should be one of ['key', 'date', 'str', 'numeric']")
	schema2_dtypes = np.unique(_schema2[dtype_colname2].values)
	if not set(schema2_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema2: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check sample_size
	if sample_size > 1:
		if int(sample_size) != sample_size:
			raise ValueError('sample_size: only accept integer when it is > 1.0')
		if (sample_size > _table1.shape[0]) or (sample_size > _table2.shape[0]):
			print('sample_size: %d is smaller than %d or %d...' % (sample_size, _table1.shape[0], _table2.shape[0]))

	# check feature_colname1 and feature_colname2
	if feature_colname1 not in _schema1.columns.values:
		raise ValueError('feature_colname1: column not in _schema1')

	if feature_colname2 not in _schema2.columns.values:
		raise ValueError('feature_colname2: column not in _schema2')

	# check dtype_colname1 and dtype_colname2
	if dtype_colname1 not in _schema1.columns.values:
		raise ValueError('dtype_colname1: column not in _schema1')

	if dtype_colname2 not in _schema2.columns.values:
		raise ValueError('dtype_colname2: column not in _schema2')

	# check output_root
	if output_root != '':
		if not os.path.isdir(output_root):
			raise ValueError('output_root: root not exists')

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
		_n_jobs = np.min([n_jobs, len(key_features)])
		key_results = Parallel(n_jobs=_n_jobs)(delayed(_compare_key)(col, table1[[col]], table2[[col]], img_dir)
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
		_n_jobs = np.min([n_jobs, len(numeric_features)])
		numeric_results = Parallel(n_jobs=_n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]],
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
		_n_jobs = np.min([n_jobs, len(string_features)])
		string_results = Parallel(n_jobs=_n_jobs)(delayed(_consist_string)(col, table1[[_key1, col]],
			table2[[_key2, col]], _key1, _key2) for col in string_features)

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
		_n_jobs = np.min([n_jobs, len(date_features)])
		date_results = Parallel(n_jobs=_n_jobs)(delayed(_consist_numeric)(col, table1[[_key1, col]], table2[[_key2, col]],
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
	_adjust_ws(ws=ws, row_height=25)

	# if there are some errors
	if len(schema_error) > 0:
		ws = wb.create_sheet(title='error')
		_ = _insert_df(schema_error, ws, header=True)
		_adjust_ws(ws=ws, row_height=25)

	wb.save(filename=os.path.join(output_root, 'data_consist_%s.xlsx' %(fname)))
	if not keep_images:
		shutil.rmtree(img_dir)


def data_consist_notebook(_table1, _table2, _key1, _key2, _schema1, _schema2, fname, 
	feature_colname1='column', feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root=''):
	"""
	Automatically generate ipynb for checking data consistency

	Parameters
	----------
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

	# check whether keys are valid
	if _key1 not in _table1.columns.values:
		raise ValueError('_key1: does not exist in table1')
	if _key2 not in _table2.columns.values:
		raise ValueError('_key2: does not exist in table2')

	# check whether two tables are unique in key level
	if (_table1[_key1].nunique() != _table1.shape[0]):
		raise ValueError('_table1: should be unique in %s level' % (_key1))
	if (_table2[_key2].nunique() != _table2.shape[0]):
		raise ValueError('_table2: should be unique in %s level' % (_key2))

	schema1_dtypes = np.unique(_schema1[dtype_colname1].values)
	if not set(schema1_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema1: data types should be one of ['key', 'date', 'str', 'numeric']")
	schema2_dtypes = np.unique(_schema2[dtype_colname2].values)
	if not set(schema2_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("_schema2: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check feature_colname1 and feature_colname2
	if feature_colname1 not in _schema1.columns.values:
		raise ValueError('feature_colname1: column not in _schema1')

	if feature_colname2 not in _schema2.columns.values:
		raise ValueError('feature_colname2: column not in _schema2')

	# check dtype_colname1 and dtype_colname2
	if dtype_colname1 not in _schema1.columns.values:
		raise ValueError('dtype_colname1: column not in _schema1')

	if dtype_colname2 not in _schema2.columns.values:
		raise ValueError('dtype_colname2: column not in _schema2')

	# check output_root
	if output_root != '':
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

	dir_path = os.path.dirname(os.path.realpath(__file__))
	main_line = open(dir_path + '/templates/data_consist_main.txt').read()
	key_line = open(dir_path + '/templates/data_consist_key.txt').read()
	str_line = open(dir_path + '/templates/data_consist_str.txt').read()
	numeric_line = open(dir_path + '/templates/data_consist_numeric.txt').read()
	date_line = open(dir_path + '/templates/data_consist_date.txt').read()

	with open(output_path, "a") as outbook:
		outbook.write(main_line)

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

			outbook.write('\n"""\n## %s (type: %s)\n\n"""\n\n' %(col, col_type))
			outbook.write('col = "%s"\n' %(col))

			# for key and str, compare intersection
			if col_type == 'key':
				outbook.write(key_line)
			elif col_type == 'str':
				outbook.write(str_line)
			elif col_type == 'numeric':
				outbook.write(numeric_line)
			else:
				outbook.write(date_line)
		outbook.close()

	os.system("python -m py2nb %s %s" %(output_path, output_path.replace('.py', '.ipynb')))