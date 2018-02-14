import pandas as pd 
import numpy as np 
import os
import shutil

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.formatting.rule import ColorScaleRule, FormulaRule, DataBar, FormatObject, Rule

import datetime

from sklearn.externals.joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('white')

from dqc_utils import (
	_style_range, _get_scale_draw_values, _draw_texts, 
	_adjust_column, _insert_df, _insert_numeric_results
)

import warnings
warnings.filterwarnings('ignore')

# global color values
VER_LINE = "#4BACC6"
TEXT_LIGHT = "#DAEEF3"
DIS_LINE = "#F79646"


"""
function: draw pretty distribution graph for a column
parameters:
_value_df: pandas DataFrame
	slice of dataframe containing enough information to check
col: string
	name of column to check
figsize: tuple, default=None
	figure size
date_flag: bool, default=False
	whether it is checking date features
"""
def distribution_summary_pretty(_value_df, col, figsize=None, date_flag=False):

	# check _value_df
	if type(_value_df) != pd.core.frame.DataFrame:
		raise ValueError('_value_df: only accept pandas DataFrame')

	# check col
	if type(col) != str:
		raise ValueError('col: only accept string')
	if col not in _value_df.columns.values:
		raise ValueError('col: does not exist')

	# check figsize
	if figsize is not None:
		if type(figsize) != tuple:
			raise ValueError('figsize: should be a tuple')
		if len(figsize) != 2:
			raise ValueError('figsize: should contain 2 elements: (width, height)')

	# check date_flag
	if type(date_flag) != bool:
		raise ValueError('date_flag: only accept boolean values')

	# colors for graph
	VER_LINE = "#4BACC6"
	TEXT_LIGHT = "#DAEEF3"
	DIS_LINE = "#F79646"

	# copy the raw dataframe
	value_df = _value_df.copy()

	if date_flag:
		numeric_col = '%s_numeric' %(col)
		if not numeric_col in value_df.columns.values:
			snapshot_date_now = str(datetime.datetime.now().date())
			value_df[numeric_col] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(value_df[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
	else:
		numeric_col = col

	# get min, mean, median, max
	value_min = value_df[numeric_col].min()
	value_mean = value_df[numeric_col].mean()
	value_median = value_df[numeric_col].median()
	value_max = value_df[numeric_col].max()

	if date_flag:
		date_min = pd.to_datetime(value_df[col], errors='coerce').min()
		date_max = pd.to_datetime(value_df[col], errors='coerce').max()

	num_uni = value_df[col].dropna().nunique()
	value_dropna = value_df[numeric_col].dropna().values

	# get distribution
	scale_flg = 0
	draw_values = value_dropna
	draw_value_4 = [value_min, value_mean, value_median, value_max]
	if np.max([abs(value_min), abs(value_max)]) >= 1000000:
		scale_flg = 1
		draw_values, draw_value_4 = _get_scale_draw_values(draw_values, draw_value_4)

	# draw and save distribution graph
	plt.clf()
	if figsize is not None:
		plt.figure(figsize)
	else:
		plt.figure(figsize=(10, 5))

	if scale_flg:
		plt.title('%s (log10 scale)' %(col))
	else:
		plt.title('%s' %(col))

	# if unique level is less than 10, draw countplot instead
	if num_uni <= 10:
		temp_df = pd.DataFrame(draw_values, columns=['value'])
		sns.countplot(temp_df['value'], color=DIS_LINE)
		if num_uni > 5:
			plt.xticks(rotation=90)
	else:
		ax = sns.distplot(draw_values, color=DIS_LINE, norm_hist=True, hist=False)
		y_low, y_up = ax.get_ylim()

		if date_flag:
			_draw_texts(draw_value_4, mark=1, text_values=[date_min, date_max], y_low=y_low, y_up=y_up, date_flag=True)
		else:
			_draw_texts(draw_value_4, mark=1, text_values=[value_min, value_mean, value_median, value_max], y_low=y_low, y_up=y_up)
	plt.show()


def _check_numeric(col, _value_df, img_dir, date_flag=False):

	value_df = _value_df.copy()

	# ensure all values are numeric
	value_df[col] = pd.to_numeric(value_df[col], errors='coerce')

	# percentage of nan
	nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]

	# if all values are nan
	if nan_rate == 1:
		return {'column': col, 'error_msg': 'all values are nan'}
	else:
		# unique_level
		num_uni = value_df[col].dropna().nunique()

		# get clean values
		value_dropna = value_df[col].dropna().values

		# get sample value
		sample_value = np.random.choice(value_dropna, 1)[0]

		# get min, mean, median, max
		value_min = value_df[col].min()
		value_mean = value_df[col].mean()
		value_median = value_df[col].median()
		value_max = value_df[col].max()
		if date_flag:
			date_min = pd.to_datetime(value_df[col.replace('_numeric', '')], errors='coerce').min()
			date_max = pd.to_datetime(value_df[col.replace('_numeric', '')], errors='coerce').max()

		# get distribution
		scale_flg = 0
		draw_values = value_dropna
		draw_value_4 = [value_min, value_mean, value_median, value_max]
		if np.max([abs(value_min), abs(value_max)]) >= 1000000:
			scale_flg = 1
			draw_values, draw_value_4 = _get_scale_draw_values(draw_values, draw_value_4)

		# draw and save distribution graph
		if date_flag:
			plt.figure(figsize=(9, 5.5))
		else:
			plt.figure(figsize=(9, 4.5))
		if scale_flg:
			plt.title('%s (log10 scale)' %(col))
		else:
			plt.title('%s' %(col))

		# if unique level is less than 10, draw countplot instead
		if num_uni <= 10:
			temp_df = pd.DataFrame(draw_values, columns=['value'])
			sns.countplot(temp_df['value'], color=DIS_LINE)
			if num_uni > 5:
				plt.xticks(rotation=90)
		else:
			ax = sns.distplot(draw_values, color=DIS_LINE, norm_hist=True, hist=False)
			y_low, y_up = ax.get_ylim()

			if date_flag:
				_draw_texts(draw_value_4, mark=1, text_values=[date_min, date_max], y_low=y_low, y_up=y_up, date_flag=True)
			else:
				_draw_texts(draw_value_4, mark=1, text_values=[value_min, value_mean, value_median, value_max], y_low=y_low, y_up=y_up)

		# save the graphs
		plt.savefig(os.path.join(img_dir, col + '.png'), transparent=True)

		output = [
			{'feature': 'column', 'value': col, 'graph': 'Distribution'},
			{'feature': 'sample_value', 'value': sample_value},
			{'feature': 'nan_rate', 'value': nan_rate},
			{'feature': 'num_uni', 'value': '%d/%d' %(num_uni, len(value_dropna))},
			{'feature': 'value_min', 'value': value_min},
			{'feature': 'value_mean', 'value': value_mean},
			{'feature': 'value_median', 'value': value_median},
			{'feature': 'value_max', 'value': value_max},
		]

		if date_flag:
			output.append({'feature': 'date_min', 'value': date_min})
			output.append({'feature': 'date_max', 'value': date_max})

		return {'column': col, 'result_df': pd.DataFrame(output)}


def _check_string(col, _value_df):
	
	value_df = _value_df.copy()

	# percentage of nan
	nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]

	# if all sample values are nan
	if nan_rate == 1:
		return {'column': col, 'error_msg': 'all values are nan'}
	else:
		# unique_level
		num_uni = value_df[col].dropna().nunique()

		# get clean values
		value_dropna = value_df[col].dropna().values

		# get sample value
		sample_value = np.random.choice(value_dropna, 1)[0]

		# get the top 10 value counts
		value_counts_df = pd.DataFrame(pd.Series(value_dropna).value_counts(), columns=['count'])
		value_counts_df[col] = value_counts_df.index.values
		value_counts_df = value_counts_df.sort_values(by='count', ascending=False).head(10)[[col, 'count']]

		output = [
			{'feature': 'column', 'value': col},
			{'feature': 'sample_value', 'value': sample_value},
			{'feature': 'nan_rate', 'value': nan_rate},
			{'feature': 'num_uni', 'value': '%d/%d' %(num_uni, len(value_dropna))}
		]
		return {'column': col, 'result_df': [pd.DataFrame(output), value_counts_df]}


def _check_date(col, value_df, img_dir):

	numeric_output = _check_numeric(col, value_df, img_dir, date_flag=True)
	col = numeric_output['column']
	if 'result_df' in numeric_output.keys():
		result_df = numeric_output['result_df']
		result_df.loc[result_df['feature']=='column', 'value'] = col.replace('_numeric', '')
		result_df.loc[0, 'graph'] = 'Distribution (months)'

		return {'column': col.replace('_numeric', ''), 'result_df': result_df}
	else:
		return {'column': col.replace('_numeric', ''), 'error_msg': numeric_output['error_msg']}


def _insert_string_results(string_results, ws, col_height):
	# construct thin border
	thin = Side(border_style="thin", color="000000")
	border = Border(top=thin, left=thin, right=thin, bottom=thin)

	# loop and output result
	for result in string_results:
		column = result['column']
		if not 'result_df' in result.keys():
			ws.append([column, result['error_msg']])
			for col in ['A', 'B']:
				ws['%s%d' %(col, ws.max_row)].style = 'Bad'
			ws.append([''])
			continue
		result_df = result['result_df'][0]
		value_counts_df = result['result_df'][1]
		head_row = _insert_df(result_df, ws)

		# if there is value counts result
		if len(value_counts_df) > 0:
			value_counts_df = value_counts_df.rename(columns={column: 'top 10 values'})
			databar_head = _insert_df(value_counts_df, ws, header=True, head_style='60 % - Accent5')

			# add conditional formatting: data bar
			first = FormatObject(type='min')
			second = FormatObject(type='max')
			data_bar = DataBar(cfvo=[first, second], color=DIS_LINE.replace('#', ''), showValue=True, minLength=None, maxLength=None)

			# assign the data bar to a rule
			rule = Rule(type='dataBar', dataBar=data_bar)
			ws.conditional_formatting.add('B%d:B%d' %(databar_head, databar_head+len(value_counts_df)), rule)

			# draw the thick outline border
			_style_range(ws, 'A%d:B%d'%(head_row, databar_head+len(value_counts_df)), border=border)
		else:
			_style_range(ws, 'A%d:B%d'%(head_row, head_row+result_df.shape[0]-1), border=border)
			
		# add gap
		ws.append([''])

	# adjust the worksheet
	_adjust_column(ws, col_height)


"""
function: summary basic information of all columns in a data table based on the provided data schema
parameters:
table_schema: pandas DataFrame
	schema of the table, should contain data types of each column
_table: pandas DataFrame
	the data table
fname: string
	the output file name
sample_size: integer or float(<=1.0), default=1.0
	int: number of sample rows to do the summary (useful for large tables)
	float: sample size in percentage
feature_colname: string
	name of the column for feature
dtype_colname: string
	name of the column for data type
output_root: string, default=''
	the root directory for the output file
keep_images: boolean, default=False
	whether to keep all generated images
n_jobs: int, default=1
	the number of jobs to run in parallel
"""
def data_summary(table_schema, _table, fname, sample_size=1.0, feature_colname='column', 
	dtype_colname='type', output_root='', keep_images=False, n_jobs=1):

	# check table_schema
	if type(table_schema) != pd.core.frame.DataFrame:
		raise ValueError('table_schema: only accept pandas DataFrame')
	schema_dtypes = np.unique(table_schema[dtype_colname].values)
	if not set(schema_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("table_schema: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check _table
	if type(_table) != pd.core.frame.DataFrame:
		raise ValueError('_table: only accept pandas DataFrame')

	# check sample_size
	if (type(sample_size) != int) and (type(sample_size) != float):
		raise ValueError('sample_size: only accept integer or float value')
	if sample_size > 1:
		if int(sample_size) != sample_size:
			raise ValueError('sample_size: only accept integer when it is > 1.0')
		if sample_size > _table.shape[0]:
			print("sample_size: %d is larger than the data size: %d" %(sample_size, _table.shape[0]))
	else:
		if sample_size <= 0:
			raise ValueError('sample_size: should be larger than 0')

	# check fname
	if type(fname) != str:
		raise ValueError('fname: only accept string')

	# check feature_colname
	if type(feature_colname) != str:
		raise ValueError('feature_colname: only accept string value')
	if not feature_colname in table_schema.columns.values:
		raise ValueError('feature_colname: column not in schema')

	# check dtype_colname
	if type(dtype_colname) != str:
		raise ValueError('dtype_colname: only accept string value')
	if not dtype_colname in table_schema.columns.values:
		raise ValueError('dtype_colname: column not in schema')

	# check output_root
	if output_root != '':
		if type(output_root) != str:
			raise ValueError('output_root: only accept string')
		if not os.path.isdir(output_root):
			raise ValueError('output_root: root not exists')

	# check n_jobs
	if type(n_jobs) != int:
		raise ValueError('n_jobs: only accept integer value') 

	# make a copy of the raw table
	table = _table.copy()

	# calculate the sample size
	if sample_size <= 1.0:
		sample_size = int(table.shape[0] * sample_size)

	if sample_size < table.shape[0]:
		table = table.sample(sample_size).reset_index(drop=True) 

	# classify features based on data type
	key_features = table_schema[table_schema[dtype_colname] == 'key'][feature_colname].values
	numeric_features = table_schema[table_schema[dtype_colname] == 'numeric'][feature_colname].values
	string_features = table_schema[table_schema[dtype_colname] == 'str'][feature_colname].values
	date_features = table_schema[table_schema[dtype_colname] == 'date'][feature_colname].values

	# temp dir to store all the images generated
	img_dir = 'img_temp'
	if os.path.isdir(img_dir):
		shutil.rmtree(img_dir)
	os.mkdir(img_dir)

	# create a new workbook to store everything
	wb = openpyxl.Workbook()

	# for key features
	# only check features in table
	key_features = [feat for feat in key_features if feat in table.columns.values]
	if len(key_features) > 0:
		# get the check result
		key_results = Parallel(n_jobs=n_jobs)(delayed(_check_string)(col, table[[col]]) for col in key_features)
		ws = wb.create_sheet(title='key')
		# write the final result to work sheet
		_insert_string_results(key_results, ws, 25)


	# for numeric features
	# only check features in table
	numeric_features = [feat for feat in numeric_features if feat in table.columns.values]
	if len(numeric_features) > 0:
		# get the check result
		numeric_results = Parallel(n_jobs=n_jobs)(delayed(_check_numeric)(col, table[[col]], img_dir) for col in numeric_features)
		ws = wb.create_sheet(title='numeric')
		# write the final result to work sheet
		_insert_numeric_results(numeric_results, ws, 35, img_dir)


	# for string features
	# only check features in table
	string_features = [feat for feat in string_features if feat in table.columns.values]
	if len(string_features) > 0:
		string_results = Parallel(n_jobs=n_jobs)(delayed(_check_string)(col, table[[col]]) for col in string_features)
		ws = wb.create_sheet(title='string')
		# write the final result to work sheet
		_insert_string_results(string_results, ws, 25)


	# for date features
	# only check features in table
	date_features = [feat for feat in date_features if feat in table.columns.values]
	if len(date_features) > 0:
		# get the current time
		snapshot_date_now = str(datetime.datetime.now().date())
		for col in date_features:
			table['%s_numeric' %(col)] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(table[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
		date_results = Parallel(n_jobs=n_jobs)(delayed(_check_date)('%s_numeric' %(col), 
			table[['%s_numeric' %(col), col]], img_dir) for col in date_features)

		ws = wb.create_sheet(title='date')
		# write the final result to work sheet
		_insert_numeric_results(date_results, ws, 35, img_dir, date_flag=True)


	# write schema
	ws = wb['Sheet']
	ws.title = 'schema'
	_ = _insert_df(table_schema[[feature_colname, dtype_colname]], ws, header=True)
	_adjust_column(ws, 25)

	wb.save(filename=os.path.join(output_root, 'data_summary_%s.xlsx' %(fname)))
	# remove all temp images
	if not keep_images:
		shutil.rmtree(img_dir)


"""
function: automatically generate ipynb for data summary
table_schema: pandas DataFrame
	schema of the table, should contain data types of each column
_table: pandas DataFrame
	the data table
fname: string
	the output file name
feature_colname: string
	name of the column for feature
dtype_colname: string
	name of the column for data type
output_root: string, default=''
	the root directory for the output file
"""
def data_summary_notebook(table_schema, _table, fname, feature_colname='column', dtype_colname='type', output_root=''):
	# check table_schema
	if type(table_schema) != pd.core.frame.DataFrame:
		raise ValueError('table_schema: only accept pandas DataFrame')
	schema_dtypes = np.unique(table_schema[dtype_colname].values)
	if not set(schema_dtypes) <= set(['key', 'date', 'str', 'numeric']):
		raise ValueError("table_schema: data types should be one of ['key', 'date', 'str', 'numeric']")

	# check _table
	if type(_table) != pd.core.frame.DataFrame:
		raise ValueError('_table: only accept pandas DataFrame')

	# check fname
	if type(fname) != str:
		raise ValueError('fname: only accept string')

	# check feature_colname
	if type(feature_colname) != str:
		raise ValueError('feature_colname: only accept string value')
	if not feature_colname in table_schema.columns.values:
		raise ValueError('feature_colname: column not in schema')

	# check dtype_colname
	if type(dtype_colname) != str:
		raise ValueError('dtype_colname: only accept string value')
	if not dtype_colname in table_schema.columns.values:
		raise ValueError('dtype_colname: column not in schema')

	# check output_root
	if output_root != '':
		if type(output_root) != str:
			raise ValueError('output_root: only accept string')
		if not os.path.isdir(output_root):
			raise ValueError('output_root: root not exists')

	# generate output file path 
	output_path = os.path.join(output_root, 'data_summary_notebook_%s.py' %(fname))

	# delete potential generated script and notebook
	if os.path.isfile(output_path):
		os.remove(output_path)

	if os.path.isfile(output_path.replace('.py', '.ipynb')):
		os.remove(output_path.replace('.py', '.ipynb'))

	with open(output_path, "a") as outbook:
		# import packages
		outbook.write('\n"""\n')
		outbook.write('## import useful packages\n\n')
		outbook.write('"""\n\n')
		
		packages = ['import pandas as pd', 'import numpy as np', '\nimport datetime\n',
		'import matplotlib.pyplot as plt', 'import seaborn as sns', 'sns.set_style("white")', 
		'\n%matplotlib inline', '\nfrom pydqc.data_summary import distribution_summary_pretty']

		outbook.write('\n'.join(packages))

		# assign value to table
		outbook.write('\n"""\n')
		outbook.write('## assign values\n\n')
		outbook.write('"""\n\n')

		outbook.write('#the data table (pandas DataFrame)\n')
		outbook.write('table = \n')
		outbook.write('print("table size: " + str(table.shape))\n\n')
		outbook.write('#global values\n')
		outbook.write('VER_LINE = "#4BACC6"\n')
		outbook.write('TEXT_LIGHT = "#DAEEF3"\n')
		outbook.write('DIS_LINE = "#F79646"\n\n')
		outbook.write('#get date of today\n')
		outbook.write('snapshot_date_now = str(datetime.datetime.now().date())\n')
		outbook.write('print("date of today: " + snapshot_date_now)\n')

		# only compare check columns in both table_schema and table
		schema_col_set = set(table_schema[feature_colname].values)
		_table_col_set = set(_table.columns.values)
		col_overlap = schema_col_set.intersection(_table_col_set)
		col_only_schema, col_only_table = (schema_col_set - _table_col_set), (_table_col_set - schema_col_set)

		# output potentail exist errors
		if len(col_only_schema) > 0:
			outbook.write('\n"""\n')
			outbook.write('### columns only in table_schema but not in table\n\n')
			outbook.write('%s\n' %(list(col_only_schema)))
			outbook.write('"""\n\n')
		elif len(col_only_table) > 0:
			outbook.write('\n"""\n')
			outbook.write('### columns only in table but not in table_schema\n\n')
			outbook.write('%s\n' %(list(col_only_table)))
			outbook.write('"""\n\n')
		else:
			# or output the consistent result
			outbook.write('\n"""\n')
			outbook.write('### columns are consistent between table_schema and table! \n\n')
			outbook.write('"""\n\n')

		# columns follow the order from table
		check_cols = [col for col in _table.columns.values if col in list(col_overlap)]
		for col in check_cols:
			# get the data type of the column
			col_type = table_schema[table_schema[feature_colname]==col][dtype_colname].values[0]

			outbook.write('\n"""\n')
			outbook.write('## %s (type: %s)\n\n' %(col, col_type))
			outbook.write('"""\n\n')

			outbook.write('col="%s"\n' %(col))
			outbook.write('value_df = table[[col]].copy()\n')

			# basic statistics
			outbook.write('nan_rate = value_df[value_df[col].isnull()].shape[0] * 1.0 / value_df.shape[0]\n')
			outbook.write('num_uni = value_df[col].dropna().nunique()\n\n')
			outbook.write('print("nan_rate: " + str(nan_rate))\n')
			outbook.write('print("num_uni out of " + str(value_df[col].dropna().shape[0]) + ": " + str(num_uni))\n')

			# for key and str, check simple value counts
			if (col_type == 'key') or (col_type == 'str'):
				outbook.write('\n"""\n')
				outbook.write('#### check value counts\n\n')
				outbook.write('"""\n\n')
				outbook.write('value_df[col].value_counts().head(10)\n')
			else:
				if col_type == 'date':
					# for date, first turn it to numeric
					outbook.write('\nvalue_df[col] = pd.to_datetime(value_df[col], errors="coerce")\n')
					outbook.write('value_df[col + "_numeric"] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(value_df[col], errors="coerce")).astype("timedelta64[M]", errors="ignore")\n\n')

				outbook.write('\n"""\n')
				outbook.write('#### check basic stats\n\n')
				outbook.write('"""\n\n')

				if col_type == 'date':
					outbook.write('date_min=value_df[col].min()\n')
					outbook.write('date_max=value_df[col].max()\n')
				else:
					outbook.write('value_min=value_df[col].min()\n')
					outbook.write('value_mean=value_df[col].mean()\n')
					outbook.write('value_median=value_df[col].median()\n')
					outbook.write('value_max=value_df[col].max()\n')

				if col_type == 'date':
					outbook.write('\nprint("min date: " + str(date_min))\n')
					outbook.write('print("max date: " + str(date_max))\n')
				else:
					outbook.write('\nprint("min: " + str(value_min))\n')
					outbook.write('print("mean: " + str(value_mean))\n')
					outbook.write('print("median: " + str(value_median))\n')
					outbook.write('print("max: " + str(value_max))\n')

				# check distribution by plotting distribution graph
				outbook.write('\n"""\n')
				outbook.write('#### check distribution\n\n')
				outbook.write('"""\n\n')
				if col_type == 'date':
					outbook.write('value_dropna = value_df[col + "_numeric"].dropna().values\n')
				else:
					outbook.write('value_dropna = value_df[col].dropna().values\n')
				# draw the graph
				outbook.write('plt.figure(figsize=(10, 5))\n')
				outbook.write('plt.title(col)\n')
				outbook.write('sns.distplot(value_dropna, color="#F79646", norm_hist=True, hist=False)\n')
				outbook.write('\n"""\n')
				outbook.write('"""\n\n')

				# or use the build in function
				outbook.write('#you can also use the build-in draw function\n')
				if col_type == 'date':
					outbook.write('distribution_summary_pretty(value_df, col, figsize=None, date_flag=True)\n')
				else:
					outbook.write('distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)\n')

		outbook.close()

	os.system("python -m py2nb %s %s" %(output_path, output_path.replace('.py', '.ipynb')))

