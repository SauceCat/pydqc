import pandas as pd
import numpy as np
import os
import shutil
from scipy.stats import spearmanr

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.formatting.rule import DataBar, FormatObject, Rule

from sklearn.externals.joblib import Parallel, delayed
import xlsxwriter

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style('white')

from matplotlib_venn import venn2
import datetime

from dqc_utils import (
	_style_range, _get_scale_draw_values, _draw_texts, 
	_adjust_ws, _insert_df, _insert_numeric_results
)

import warnings
warnings.filterwarnings('ignore')


# global color values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"

TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"


"""
function: draw pretty distribution graph for comparing a column between two tables
parameters:
_df1: pandas DataFrame
	slice of table1 containing enough information to check
_df2: pandas DataFrame
	slice of table2 containing enough information to check
col: string
	name of column to check
figsize: tuple, default=None
	figure size
date_flag: bool, default=False
	whether it is checking date features
"""
def distribution_compare_pretty(_df1, _df2, col, figsize=None, date_flag=False):

	# check _df1
	if type(_df1) != pd.core.frame.DataFrame:
		raise ValueError('_df1: only accept pandas DataFrame')

	# check _df2
	if type(_df2) != pd.core.frame.DataFrame:
		raise ValueError('_df2: only accept pandas DataFrame')

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

	if date_flag:
		numeric_col = '%s_numeric' %(col)
		if not numeric_col in df1.columns.values:
			snapshot_date_now = str(datetime.datetime.now().date())
			df1[numeric_col] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df1[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
		if not numeric_col in df2.columns.values:
			snapshot_date_now = str(datetime.datetime.now().date())
			df2[numeric_col] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df2[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
	else:
		numeric_col = col

	value_mins = [df1[numeric_col].min(), df2[numeric_col].min()]
	value_means = [df1[numeric_col].mean(), df2[numeric_col].mean()]
	value_medians = [df1[numeric_col].median(), df2[numeric_col].median()]
	value_maxs = [df1[numeric_col].max(), df2[numeric_col].max()]

	if date_flag:
		date_mins = [pd.to_datetime(df1[col], errors='coerce').min(), pd.to_datetime(df2[col], errors='coerce').min()]
		date_maxs = [pd.to_datetime(df1[col], errors='coerce').max(), pd.to_datetime(df2[col], errors='coerce').max()]

	both_value_max = np.max([abs(v) for v in value_maxs] + [abs(v) for v in value_mins])

	# get clean values
	df1_sample_dropna_values = df1[numeric_col].dropna().values
	df2_sample_dropna_values = df2[numeric_col].dropna().values

	# get distribution
	scale_flg = 0
	df1_draw_values = df1_sample_dropna_values
	df1_draw_value_4 = [value_mins[0], value_means[0], value_medians[0], value_maxs[0]]

	df2_draw_values = df2_sample_dropna_values
	df2_draw_value_4 = [value_mins[1], value_means[1], value_medians[1], value_maxs[1]]

	if both_value_max >= 1000000:
		scale_flg = 1
		df1_draw_values, df1_draw_value_4 = _get_scale_draw_values(df1_draw_values, df1_draw_value_4)
		df2_draw_values, df2_draw_value_4 = _get_scale_draw_values(df2_draw_values, df2_draw_value_4)

	# draw the graph
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
	both_num_uni = np.max([df1[col].dropna().nunique(), df2[col].dropna().nunique()])
	if both_num_uni <= 10:
		df1_temp = pd.DataFrame(df1_sample_dropna_values, columns=['value'])
		df1_temp['type'] = 'table1'
		df2_temp = pd.DataFrame(df2_sample_dropna_values, columns=['value'])
		df2_temp['type'] = 'table2'
		full_temp = pd.concat([df1_temp, df2_temp], axis=0)
		sns.countplot(full_temp['value'], hue=full_temp['type'], palette=sns.color_palette([TABLE1_DARK, TABLE2_DARK]))
		if both_num_uni > 5:
			plt.xticks(rotation=90)
		plt.legend(loc=1)
	else:
		ax1 = sns.distplot(df1_draw_values, color=TABLE1_DARK, hist=False, label='table1')
		ax2 = sns.distplot(df2_draw_values, color=TABLE2_DARK, hist=False, label='table2')
		y_low_1, y_up_1 = ax1.get_ylim()
		y_low_2, y_up_2 = ax2.get_ylim()
		y_low, y_up = np.min([y_low_1, y_low_2]), np.max([y_up_1, y_up_2])
		plt.ylim((y_low, y_up))

		if date_flag:
			_draw_texts(text_values=[date_mins[0], date_maxs[0]], draw_value_4=df1_draw_value_4, mark=1,
						y_low=y_low, y_up=y_up, date_flag=True)
			_draw_texts(text_values=[date_mins[1], date_maxs[1]], draw_value_4=df2_draw_value_4, mark=2,
						y_low=y_low, y_up=y_up, date_flag=True)
		else:
			_draw_texts(text_values=[value_mins[0], value_means[0], value_medians[0], value_maxs[0]],
						draw_value_4=df1_draw_value_4, mark=1, y_low=y_low, y_up=y_up)
			_draw_texts(text_values=[value_mins[1], value_means[1], value_medians[1], value_maxs[1]],
						draw_value_4=df2_draw_value_4, mark=2, y_low=y_low, y_up=y_up)

	plt.show()


def _simple_stats(col, _df1, _df2, stat_type):

	df1 = _df1.copy()
	df2 = _df2.copy()

	# default output 
	output = {'sample_value': np.nan, 'nan_rate': np.nan, 'num_uni': np.nan, 'value_min': np.nan, 
			'value_mean': np.nan, 'value_median': np.nan, 'value_max': np.nan, 'date_min': np.nan, 'date_max': np.nan}
	
	# nan_rate
	nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]
	nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]
	output['nan_rate'] = '%s\n%s' %(str(round(nan_rate1, 3)), str(round(nan_rate2, 3)))

	if nan_rate1 == 1 or nan_rate2 == 1:
		return output

	# sample value
	try:
		sample_value1 = df1[col].dropna().sample(1).values[0]
	except:
		sample_value1 = ''

	try:
		sample_value2 = df2[col].dropna().sample(1).values[0]
	except:
		sample_value2 = ''
	output['sample_value'] = '%s\n%s' %(str(sample_value1), str(sample_value2))

	# num_uni
	num_uni1 = df1[col].dropna().nunique()
	num_uni2 = df2[col].dropna().nunique()
	output['num_uni'] = '%s/%s\n%s/%s' %(str(num_uni1), str(df1.dropna().shape[0]), str(num_uni2), str(df2.dropna().shape[0]))

	if (stat_type == 'key') or (stat_type == 'str'):
		return output

	# value_min
	value_min1 = df1[col].min()
	value_min2 = df2[col].min()
	output['value_min'] = '%s\n%s' %(str(value_min1), str(value_min2))

	# value_mean
	value_mean1 = df1[col].mean()
	value_mean2 = df2[col].mean()
	output['value_mean'] = '%s\n%s' %(str(value_mean1), str(value_mean2))

	# value_median
	value_median1 = df1[col].median()
	value_median2 = df2[col].median()
	output['value_median'] = '%s\n%s' %(str(value_median1), str(value_median2))

	# value_max
	value_max1 = df1[col].max()
	value_max2 = df2[col].max()
	output['value_max'] = '%s\n%s' %(str(value_max1), str(value_max2))

	if stat_type == 'numeric':
		return output

	# date_min
	date_min1 = pd.to_datetime(df1[col.replace('_numeric', '')], errors='coerce').min()
	date_min2 = pd.to_datetime(df2[col.replace('_numeric', '')], errors='coerce').min()
	output['date_min'] = '%s\n%s' %(str(date_min1), str(date_min2))

	# date_max
	date_max1 = pd.to_datetime(df1[col.replace('_numeric', '')], errors='coerce').max()
	date_max2 = pd.to_datetime(df2[col.replace('_numeric', '')], errors='coerce').max()
	output['date_max'] = '%s\n%s' %(str(date_max1), str(date_max2))

	return output


def _compare_key(key, _df1, _df2, img_dir):

	df1 = _df1.copy()
	df2 = _df2.copy()

	# get basic stats information
	stat_output = _simple_stats(key, df1, df2, 'key')

	# basic check for key
	nan_rates = [pd.to_numeric(v) for v in stat_output['nan_rate'].split('\n')]
	nan_rate1, nan_rate2 = nan_rates[0], nan_rates[1]

	if (nan_rate1 == 1) or (nan_rate2 == 1): 
		if (nan_rate1 == 1) and (nan_rate2 == 1):
			error_msg = 'all nan in both table'
		elif nan_rate1 == 1:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	set_df1_key = set(df1[key].dropna().values) if nan_rate1 < 1 else set()
	set_df2_key = set(df2[key].dropna().values) if nan_rate2 < 1 else set()
	key_overlap = len(set_df1_key.intersection(set_df2_key))
	key_only_df1, key_only_df2 = len(set_df1_key - set_df2_key), len(set_df2_key - set_df1_key)
	overlap_rate = key_overlap * 1.0 / (key_overlap + key_only_df1 + key_only_df2)

	# generate the output
	output = [
		{'feature': 'column', 'value': key, 'graph': 'venn graph'},
		{'feature': 'sample_value', 'value': stat_output['sample_value']},
		{'feature': 'nan_rate', 'value': stat_output['nan_rate']},
		{'feature': 'num_uni', 'value': stat_output['num_uni']},
		{'feature': 'overlap', 'value': key_overlap},
		{'feature': 'only in table1', 'value': key_only_df1},
		{'feature': 'only in table2', 'value': key_only_df2},
		{'feature': 'overlap rate', 'value': round(overlap_rate, 3)}
	]

	# draw the venn graph
	plt.figure(figsize=(9, 5))
	venn2([set_df1_key, set_df2_key], set_labels=['table1', 'table2'], set_colors=(TABLE1_DARK, TABLE2_DARK), alpha=0.8)

	# save the graphs
	plt.savefig(os.path.join(img_dir, key + '.png'), transparent=True)

	return {'column': key, 'result_df': pd.DataFrame(output), 'corr': {'column': key, 'corr': round(overlap_rate, 3)}}


def _compare_numeric(col, _df1, _df2, img_dir, date_flag=False):

	# sampling 
	df1_sample = _df1.copy()
	df2_sample = _df2.copy()

	stat_output = _simple_stats(col, df1_sample, df2_sample, 'numeric')

	nan_rates = [pd.to_numeric(v) for v in stat_output['nan_rate'].split('\n')]
	nan_rate1, nan_rate2 = nan_rates[0], nan_rates[1]
	if (nan_rate1 == 1) or (nan_rate2 == 1): 
		if (nan_rate1 == 1) and (nan_rate2 == 1):
			error_msg = 'all nan in both table'
		elif nan_rate1 == 1:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	# generate the output
	output = [
		{'feature': 'column', 'value': col, 'graph': 'Distribution'},
		{'feature': 'sample_value', 'value': stat_output['sample_value']},
		{'feature': 'nan_rate', 'value': stat_output['nan_rate']},
		{'feature': 'num_uni', 'value': stat_output['num_uni']},
		{'feature': 'value_min', 'value': stat_output['value_min']},
		{'feature': 'value_mean', 'value': stat_output['value_mean']},
		{'feature': 'value_median', 'value': stat_output['value_median']},
		{'feature': 'value_max', 'value': stat_output['value_max']}
	]

	both_value_max = np.max([abs(pd.to_numeric(v)) for v in stat_output['value_max'].split('\n')] + \
		[abs(pd.to_numeric(v)) for v in stat_output['value_min'].split('\n')])

	# get clean values
	df1_sample_dropna_values = df1_sample[col].dropna().values
	df2_sample_dropna_values = df2_sample[col].dropna().values

	value_mins = [pd.to_numeric(v) for v in stat_output['value_min'].split('\n')]
	value_means = [pd.to_numeric(v) for v in stat_output['value_mean'].split('\n')]
	value_medians = [pd.to_numeric(v) for v in stat_output['value_median'].split('\n')]
	value_maxs = [pd.to_numeric(v) for v in stat_output['value_max'].split('\n')]

	if date_flag:
		dt1 = pd.to_datetime(df1_sample[col.replace('_numeric', '')], errors='coerce')
		dt2 = pd.to_datetime(df2_sample[col.replace('_numeric', '')], errors='coerce')
		date_min1, date_max1 = dt1.min(), dt1.max()
		date_min2, date_max2 = dt2.min(), dt2.max()

	# get distribution
	scale_flg = 0
	df1_draw_values = df1_sample_dropna_values
	df1_draw_value_4 = [value_mins[0], value_means[0], value_medians[0], value_maxs[0]]

	df2_draw_values = df2_sample_dropna_values
	df2_draw_value_4 = [value_mins[1], value_means[1], value_medians[1], value_maxs[1]]
	
	if both_value_max >= 1000000:
		scale_flg = 1
		df1_draw_values, df1_draw_value_4 = _get_scale_draw_values(df1_draw_values, df1_draw_value_4)
		df2_draw_values, df2_draw_value_4 = _get_scale_draw_values(df2_draw_values, df2_draw_value_4)

	# calculate correlation between two distributions
	num_uni1 = int(stat_output['num_uni'].split('\n')[0].split('/')[0])
	num_uni2 = int(stat_output['num_uni'].split('\n')[1].split('/')[0])
	if np.max([num_uni1, num_uni2]) <= 100:
		vc1, vc2 = _value_counts_df(df1_draw_values), _value_counts_df(df2_draw_values)
		vc = vc1.merge(vc2, on='value', how='outer').fillna(0)
		obs1, obs2 = vc['count_x'].values * 1.0 / vc['count_x'].sum(), vc['count_y'].values * 1.0 / vc['count_y'].sum()
	else:
		both_min = np.min([np.min(df1_draw_values), np.min(df2_draw_values)])
		both_max = np.max([np.max(df1_draw_values), np.max(df2_draw_values)])
		hist1 = np.histogram(df1_draw_values, bins=100, range=(both_min, both_max), normed=False, density=False)
		hist2 = np.histogram(df2_draw_values, bins=100, range=(both_min, both_max), normed=False, density=False)
		obs1, obs2 = hist1[0] / (np.sum(hist1[0]) * 1.0), hist2[0] / (np.sum(hist2[0]) * 1.0)

	if len(obs1) == 1:
		corr = np.min([1. - nan_rate1, 1. - nan_rate2]) * 1.0 / np.max([1. - nan_rate1, 1. - nan_rate2])
	elif list(obs1) == list(obs2):
		corr = 1.0
	else:
		corr = spearmanr(obs1, obs2)[0]		

	# draw and save distribution graph
	if date_flag:
		plt.figure(figsize=(9, 8))
	else:
		plt.figure(figsize=(9, 6))
	if scale_flg:
		plt.title('%s (log10 scale)' %(col))
	else:
		plt.title('%s' %(col))

	# if unique level is less than 10, draw countplot instead
	both_num_uni = np.max([num_uni1, num_uni2])
	if both_num_uni <= 10:
		df1_temp = pd.DataFrame(df1_sample_dropna_values, columns=['value'])
		df1_temp['type'] = 'table1'
		df2_temp = pd.DataFrame(df2_sample_dropna_values, columns=['value'])
		df2_temp['type'] = 'table2'
		full_temp = pd.concat([df1_temp, df2_temp], axis=0)
		sns.countplot(full_temp['value'], hue=full_temp['type'], palette=sns.color_palette([TABLE1_DARK, TABLE2_DARK]))
		if both_num_uni > 5:
			plt.xticks(rotation=90)
		plt.legend(loc=1)
	else:
		ax1 = sns.distplot(df1_draw_values, color=TABLE1_DARK, hist=False, label='table1')
		ax2 = sns.distplot(df2_draw_values, color=TABLE2_DARK, hist=False, label='table2')
		y_low_1, y_up_1 = ax1.get_ylim()
		y_low_2, y_up_2 = ax2.get_ylim()
		y_low, y_up = np.min([y_low_1, y_low_2]), np.max([y_up_1, y_up_2])
		plt.ylim((y_low, y_up))

		if date_flag:
			_draw_texts(text_values=[date_min1, date_max1], draw_value_4=df1_draw_value_4, mark=1,
						y_low=y_low, y_up=y_up, date_flag=True)
			_draw_texts(text_values=[date_min2, date_max2], draw_value_4=df2_draw_value_4, mark=2,
						y_low=y_low, y_up=y_up, date_flag=True)
		else:
			_draw_texts(text_values=[value_mins[0], value_means[0], value_medians[0], value_maxs[0]],
						draw_value_4=df1_draw_value_4, mark=1, y_low=y_low, y_up=y_up)
			_draw_texts(text_values=[value_mins[1], value_means[1], value_medians[1], value_maxs[1]],
						draw_value_4=df2_draw_value_4, mark=2, y_low=y_low, y_up=y_up)
			
	# save the graphs
	plt.savefig(os.path.join(img_dir, col + '.png'), transparent=True)

	if date_flag:
		output.append({'feature': 'date_min', 'value': '%s\n%s' %(date_min1, date_min2)})
		output.append({'feature': 'date_max', 'value': '%s\n%s' %(date_max1, date_max2)})
	output.append({'feature': 'corr', 'value': round(corr, 3)})

	return {'column': col, 'result_df': pd.DataFrame(output), 'corr': {'column': col, 'corr': round(corr, 3)}}


def _value_counts_df(values):
	temp = pd.DataFrame(pd.Series(values).value_counts(), columns=['count'])
	temp['value'] = temp.index.values
	return temp.reset_index(drop=True)


def _compare_string(col, _df1, _df2, img_dir):

	# sampling
	df1_sample = _df1.copy()
	df2_sample = _df2.copy()

	# get basic stats information
	stat_output = _simple_stats(col, df1_sample, df2_sample, 'str')

	nan_rates = [pd.to_numeric(v) for v in stat_output['nan_rate'].split('\n')]
	nan_rate1, nan_rate2 = nan_rates[0], nan_rates[1]

	if (nan_rate1 == 1) or (nan_rate2 == 1): 
		if (nan_rate1 == 1) and (nan_rate2 == 1):
			error_msg = 'all nan in both table'
		elif nan_rate1 == 1:
			error_msg = 'all nan in table1'
		else:
			error_msg = 'all nan in table2'
		return {'column': col, 'error_msg': error_msg}

	# basic check for category features
	set_df1_col = set(df1_sample[col].dropna().values) if nan_rate1 < 1 else set()
	set_df2_col = set(df2_sample[col].dropna().values) if nan_rate2 < 1 else set()
	col_overlap = len(set_df1_col.intersection(set_df2_col))
	col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)

	# generate the output
	output = [
		{'feature': 'column', 'value': col, 'graph': ''},
		{'feature': 'sample_value', 'value': stat_output['sample_value']},
		{'feature': 'nan_rate', 'value': stat_output['nan_rate']},
		{'feature': 'num_uni', 'value': stat_output['num_uni']},
		{'feature': 'overlap', 'value': col_overlap},
		{'feature': 'only in table1', 'value': col_only_df1},
		{'feature': 'only in table2', 'value': col_only_df2},
	]

	# get clean data for ploting the graph
	df1_sample_dropna_values = df1_sample[col].dropna().values
	df2_sample_dropna_values = df2_sample[col].dropna().values

	# draw the count graph
	value_counts_df1 = _value_counts_df(df1_sample_dropna_values)
	value_counts_df2 = _value_counts_df(df2_sample_dropna_values)

	value_counts_df1_top10 = value_counts_df1.sort_values('count', ascending=False).head(10)
	value_counts_df2_top10 = value_counts_df2.sort_values('count', ascending=False).head(10)

	value_counts_df = value_counts_df1.merge(value_counts_df2, on='value', how='outer').fillna(0)
	value_counts_df['count_x_per'] = value_counts_df['count_x'] * 1.0 / value_counts_df['count_x'].sum()
	value_counts_df['count_y_per'] = value_counts_df['count_y'] * 1.0 / value_counts_df['count_y'].sum()

	if len(value_counts_df) == 1:
		corr = np.min([1. - nan_rate1, 1. - nan_rate2]) * 1.0 / np.max([1. - nan_rate1, 1. - nan_rate2])
	elif list(value_counts_df['count_x_per'].values) == list(value_counts_df['count_y_per'].values):
		corr = 1.0
	else:
		corr = spearmanr(value_counts_df['count_x_per'].values, value_counts_df['count_y_per'].values)[0]
	output.append({'feature': 'corr', 'value': round(corr, 3)})

	value_counts_df_top10 = value_counts_df1_top10.merge(value_counts_df2_top10, on='value', how='outer').fillna(0)
	value_counts_df_top10 = value_counts_df_top10[['value', 'count_x', 'count_y']]

	return {'column': col, 'result_df': [pd.DataFrame(output), value_counts_df_top10], 
			'corr': {'column': col, 'corr': round(corr, 3)}}


def _compare_date(col, _df1, _df2, img_dir):
	numeric_output = _compare_numeric(col, _df1, _df2, img_dir, date_flag=True)
	col = numeric_output['column']

	if 'result_df' in numeric_output.keys():
		result_df = numeric_output['result_df']
		result_df.loc[result_df['feature']=='column', 'value'] = col.replace('_numeric', '')
		result_df.loc[0, 'graph'] = 'Distribution (months)'

		return {'column': col.replace('_numeric', ''), 'result_df': result_df, 
				'corr': {'column': col.replace('_numeric', ''), 'corr': numeric_output['corr']['corr']}}
	else:
		return {'column': col.replace('_numeric', ''), 'error_msg': numeric_output['error_msg']}


def _insert_compare_string_results(string_results, ws, row_height):
	# construct thick border
	thin = Side(border_style="thin", color="000000")
	border = Border(top=thin, left=thin, right=thin, bottom=thin)

	row_heights = {}

	# loop and output result
	for result in string_results:
		column = result['column']
		if not 'result_df' in result.keys():
			ws.append([column, result['error_msg']])
			for col in ['A', 'B']:
				ws['%s%d' %(col, ws.max_row)].style = 'Bad'
			ws.append([''])
			continue
		result_df = result['result_df'][0][['feature', 'value', 'graph']]
		value_counts_df = result['result_df'][1]
		head_row = _insert_df(result_df, ws)

		# if there is value counts result
		if len(value_counts_df) > 0:
			value_counts_df = value_counts_df.rename(columns={'value': 'top 10 values', 'count_x': 'count_1', 'count_y': 'count_2'})
			databar_head = _insert_df(value_counts_df, ws, header=True, head_style='60 % - Accent5')
			for row_idx in range(databar_head, databar_head+value_counts_df.shape[0]+1):
				row_heights[row_idx] = 25

			# add conditional formatting: data bar
			first = FormatObject(type='min')
			second = FormatObject(type='max')
			data_bar1 = DataBar(cfvo=[first, second], color=TABLE1_DARK.replace('#', ''), showValue=True, minLength=None, maxLength=None)
			data_bar2 = DataBar(cfvo=[first, second], color=TABLE2_DARK.replace('#', ''), showValue=True, minLength=None, maxLength=None)

			# assign the data bar to a rule
			rule1 = Rule(type='dataBar', dataBar=data_bar1)
			ws.conditional_formatting.add('B%d:B%d' %(databar_head, databar_head+len(value_counts_df)), rule1)
			rule2 = Rule(type='dataBar', dataBar=data_bar2)
			ws.conditional_formatting.add('C%d:C%d' %(databar_head, databar_head+len(value_counts_df)), rule2)

			# draw the thick outline border
			_style_range(ws, 'A%d:C%d'%(head_row, databar_head+len(value_counts_df)), border=border)
		else:
			_style_range(ws, 'A%d:C%d'%(head_row, head_row+result_df.shape[0]-1), border=border)

		# add gap
		ws.append([''])

	_adjust_ws(ws, row_height=row_height, row_heights=row_heights, adjust_type='str')


"""
function: compare values of same columns between two tables
parameters:
_table1: pandas DataFrame
	one of the two tables to compare
_table2: pandas DataFrame
	one of the two tables to compare
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
def data_compare(_table1, _table2, _schema1, _schema2, fname, sample_size=1.0, feature_colname1='column', feature_colname2='column', 
	dtype_colname1='type', dtype_colname2='type', output_root='', keep_images=False, n_jobs=1):

	# check _table1 and _table2
	if type(_table1) != pd.core.frame.DataFrame:
		raise ValueError('_table1: only accept pandas DataFrame')
	if type(_table2) != pd.core.frame.DataFrame:
		raise ValueError('_table2: only accept pandas DataFrame')

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
		sample_size1 = int(table1.shape[0] * sample_size)
		sample_size2 = int(table2.shape[0] * sample_size)
		sample_size = np.min([sample_size1, sample_size2])

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

	# do sampling here
	if sample_size < table1.shape[0]:
		table1 = table1.sample(sample_size).reset_index(drop=True)
	if sample_size < table2.shape[0]:
		table2 = table2.sample(sample_size).reset_index(drop=True)

	# for numeric features
	# only check features in both tables
	numeric_features = [feat for feat in numeric_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(numeric_features) > 0:
		numeric_results = Parallel(n_jobs=n_jobs)(delayed(_compare_numeric)(col, table1[[col]], table2[[col]], img_dir) 
			for col in numeric_features)

		for numeric_result in numeric_results:
			if 'corr' in numeric_result.keys():
				corr_results.append(numeric_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='numeric')
		_insert_numeric_results(numeric_results, ws, 40, img_dir)


	# for string features
	# only check features in both tables
	string_features = [feat for feat in string_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(string_features) > 0:
		string_results = Parallel(n_jobs=n_jobs)(delayed(_compare_string)(col, table1[[col]], table2[[col]], img_dir) 
			for col in string_features)

		for string_result in string_results:
			if 'corr' in string_result.keys():
				corr_results.append(string_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='string')
		_insert_compare_string_results(string_results, ws, 40)


	# for date features
	# only check features in both tables
	date_features = [feat for feat in date_features if (feat in table1.columns.values) and (feat in table2.columns.values)]
	if len(date_features) > 0:
		# get the current time
		snapshot_date_now = str(datetime.datetime.now().date())
		for col in date_features:
			table1['%s_numeric' %(col)] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(table1[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
			table2['%s_numeric' %(col)] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(table2[col], 
				errors='coerce')).astype('timedelta64[M]', errors='ignore')
		date_results = Parallel(n_jobs=n_jobs)(delayed(_compare_date)('%s_numeric' %(col), 
			table1[['%s_numeric' %(col), col]], table2[['%s_numeric' %(col), col]], img_dir) for col in date_features)

		for date_result in date_results:
			if 'corr' in date_result.keys():
				corr_results.append(date_result['corr'])

		# write all results to worksheet
		ws = wb.create_sheet(title='date')
		_insert_numeric_results(date_results, ws, 40, img_dir, date_flag=True)


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

	wb.save(filename=os.path.join(output_root, 'data_compare_%s.xlsx' %(fname)))
	if not keep_images:
		shutil.rmtree(img_dir)


"""
function: automatically generate ipynb for data comparison
parameters:
_table1: pandas DataFrame
	one of the two tables to compare
_table2: pandas DataFrame
	one of the two tables to compare
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
def data_compare_notebook(_table1, _table2, _schema1, _schema2, fname, feature_colname1='column', feature_colname2='column', 
	dtype_colname1='type', dtype_colname2='type', output_root=''):

	# check _table1 and _table2
	if type(_table1) != pd.core.frame.DataFrame:
		raise ValueError('_table1: only accept pandas DataFrame')
	if type(_table2) != pd.core.frame.DataFrame:
		raise ValueError('_table2: only accept pandas DataFrame')

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
	output_path = os.path.join(output_root, 'data_compare_notebook_%s.py' %(fname))

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
		
		packages = ['import pandas as pd', 'import numpy as np', '\nimport datetime\n', 'import matplotlib.pyplot as plt', 
		'import seaborn as sns', 'sns.set_style("white")', 'from matplotlib_venn import venn2','\n%matplotlib inline', 
		'\nfrom pydqc.data_compare import distribution_compare_pretty']

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
			outbook.write('df1 = table1[[col]].copy()\n')
			outbook.write('df2 = table2[[col]].copy()\n\n')
			if col_type == 'date':
				outbook.write('df1[col] = pd.to_datetime(df1[col], errors="coerce")\n')
				outbook.write('df1[col + "_numeric"] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df1[col], errors="coerce")).astype("timedelta64[M]", errors="ignore")\n\n')
				outbook.write('df2[col] = pd.to_datetime(df2[col], errors="coerce")\n')
				outbook.write('df2[col + "_numeric"] = (pd.to_datetime(snapshot_date_now) - pd.to_datetime(df2[col], errors="coerce")).astype("timedelta64[M]", errors="ignore")\n\n')

			# basic statistics comparison
			outbook.write('\n"""\n')
			outbook.write('#### basic comparison\n\n')
			outbook.write('"""\n\n')
			outbook.write('#sample values\n')
			outbook.write('\n"""\n')
			outbook.write('"""\n\n')
			outbook.write('df1.sample(5)\n')
			outbook.write('\n"""\n')
			outbook.write('"""\n\n')
			outbook.write('df2.sample(5)\n')

			outbook.write('\n"""\n')
			outbook.write('"""\n\n')
			outbook.write('#nan_rate\n')
			outbook.write('nan_rate1 = df1[df1[col].isnull()].shape[0] * 1.0 / df1.shape[0]\n')
			outbook.write('nan_rate2 = df2[df2[col].isnull()].shape[0] * 1.0 / df2.shape[0]\n\n')
			outbook.write('print("table1 nan_rate: " + str(nan_rate1) + "; table2 nan_rate: " + str(nan_rate2))\n')

			outbook.write('\n"""\n')
			outbook.write('"""\n\n')
			outbook.write('#num_uni\n')
			outbook.write('num_uni1 = df1[col].dropna().nunique()\n')
			outbook.write('num_uni2 = df2[col].dropna().nunique()\n\n')
			outbook.write('print("table1 num_uni out of " + str(df1[col].dropna().shape[0]) + ": " + str(num_uni1))\n')
			outbook.write('print("table2 num_uni out of " + str(df2[col].dropna().shape[0]) + ": " + str(num_uni2))\n\n')

			# for key and str, compare intersection
			if (col_type == 'key') or (col_type == 'str'):
				outbook.write('\n"""\n')
				outbook.write('#### compare intersection\n\n')
				outbook.write('"""\n\n')
				outbook.write('set_df1_col = set(df1[col].dropna().values) if nan_rate1 < 1 else set()\n')
				outbook.write('set_df2_col = set(df2[col].dropna().values) if nan_rate2 < 1 else set()\n')
				outbook.write('col_overlap = len(set_df1_col.intersection(set_df2_col))\n')
				outbook.write('col_only_df1, col_only_df2 = len(set_df1_col - set_df2_col), len(set_df2_col - set_df1_col)\n\n')
				outbook.write('print("col_overlap: " + str(col_overlap) + "; col_only_df1: " + str(col_only_df1) + "; col_only_df2: " + str(col_only_df2))\n')

				# draw venn graph for key
				if col_type == 'key':
					outbook.write('\n"""\n')
					outbook.write('#### draw venn graph\n\n')
					outbook.write('"""\n\n')
					outbook.write('plt.figure(figsize=(10, 5))\n')
					outbook.write('venn2([set_df1_col, set_df2_col], set_labels=["table1", "table2"], set_colors=("#4BACC6", "#F79646"), alpha=0.8)\n')
				else:
					# check simple value counts for str
					outbook.write('\n"""\n')
					outbook.write('#### check value counts\n\n')
					outbook.write('"""\n\n')
					outbook.write('value_counts_df1 = pd.DataFrame(df1[col].value_counts())\n')
					outbook.write('value_counts_df1.columns = ["count_1"]\n')
					outbook.write('value_counts_df1[col] = value_counts_df1.index.values\n')
					outbook.write('value_counts_df1 = value_counts_df1.reset_index(drop=True)[[col, "count_1"]]\n')
					outbook.write('value_counts_df1 = value_counts_df1.sort_values(by="count_1", ascending=False).head(10)\n\n')
					outbook.write('value_counts_df2 = pd.DataFrame(df2[col].value_counts())\n')
					outbook.write('value_counts_df2.columns = ["count_2"]\n')
					outbook.write('value_counts_df2[col] = value_counts_df2.index.values\n')
					outbook.write('value_counts_df2 = value_counts_df2.reset_index(drop=True)[[col, "count_2"]]\n')
					outbook.write('value_counts_df2 = value_counts_df2.sort_values(by="count_2", ascending=False).head(10)\n\n')
					outbook.write('value_counts_df = value_counts_df1.merge(value_counts_df2, on=col, how="outer").fillna(0)\n')
					
					outbook.write('\n"""\n')
					outbook.write('"""\n\n')
					outbook.write('value_counts_df\n')
			else:
				if col_type == 'date':
					outbook.write('\n"""\n')
					outbook.write('"""\n\n')
					outbook.write('#min date\n')
					outbook.write('date_min1=df1[col].min()\n')
					outbook.write('date_min2=df2[col].min()\n')
					outbook.write('print("table1 date_min: " + str(date_min1) + "; table2 date_min: " + str(date_min2))\n\n')

					outbook.write('#max date\n')
					outbook.write('date_max1=df1[col].max()\n')
					outbook.write('date_max2=df2[col].max()\n')
					outbook.write('print("table1 date_max: " + str(date_max1) + "; table2 date_max: " + str(date_max2))\n\n')

					outbook.write('#min value\n')
					outbook.write('value_min1=df1[col + "_numeric"].min()\n')
					outbook.write('value_min2=df2[col + "_numeric"].min()\n')
					outbook.write('print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))\n\n')

					outbook.write('#mean value\n')
					outbook.write('value_mean1=df1[col + "_numeric"].mean()\n')
					outbook.write('value_mean2=df2[col + "_numeric"].mean()\n')
					outbook.write('print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))\n\n')

					outbook.write('#median value\n')
					outbook.write('value_median1=df1[col + "_numeric"].median()\n')
					outbook.write('value_median2=df2[col + "_numeric"].median()\n')
					outbook.write('print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))\n\n')

					outbook.write('#max value\n')
					outbook.write('value_max1=df1[col + "_numeric"].max()\n')
					outbook.write('value_max2=df2[col + "_numeric"].max()\n')
					outbook.write('print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))\n\n')
				else:
					outbook.write('#min value\n')
					outbook.write('value_min1=df1[col].min()\n')
					outbook.write('value_min2=df2[col].min()\n')
					outbook.write('print("table1 min: " + str(value_min1) + "; table2 min: " + str(value_min2))\n\n')

					outbook.write('#mean value\n')
					outbook.write('value_mean1=df1[col].mean()\n')
					outbook.write('value_mean2=df2[col].mean()\n')
					outbook.write('print("table1 mean: " + str(value_mean1) + "; table2 mean: " + str(value_mean2))\n\n')

					outbook.write('#median value\n')
					outbook.write('value_median1=df1[col].median()\n')
					outbook.write('value_median2=df2[col].median()\n')
					outbook.write('print("table1 median: " + str(value_median1) + "; table2 median: " + str(value_median2))\n\n')

					outbook.write('#max value\n')
					outbook.write('value_max1=df1[col].max()\n')
					outbook.write('value_max2=df2[col].max()\n')
					outbook.write('print("table1 max: " + str(value_max1) + "; table2 max: " + str(value_max2))\n\n')

				outbook.write('\n"""\n')
				outbook.write('#### check distribution\n\n')
				outbook.write('"""\n\n')

				if col_type == 'date':
					outbook.write('value_dropna_df1 = df1[col + "_numeric"].dropna().values\n')
					outbook.write('value_dropna_df2 = df2[col + "_numeric"].dropna().values\n')
				else:
					outbook.write('value_dropna_df1 = df1[col].dropna().values\n')
					outbook.write('value_dropna_df2 = df2[col].dropna().values\n')

				# draw the distribution graph
				outbook.write('plt.figure(figsize=(10, 5))\n')
				outbook.write('sns.distplot(value_dropna_df1, color="#4BACC6", norm_hist=True, hist=False, label="table1")\n')
				outbook.write('sns.distplot(value_dropna_df2, color="#F79646", norm_hist=True, hist=False, label="table2")\n')
				outbook.write('\n"""\n')
				outbook.write('"""\n\n')
				outbook.write('#you can also use the build-in draw function\n')
				if col_type == 'date':
					outbook.write('distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=True)\n')
				else:
					outbook.write('distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)\n')

		outbook.close()

	os.system("python -m py2nb %s %s" %(output_path, output_path.replace('.py', '.ipynb')))