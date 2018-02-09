import openpyxl
from openpyxl.styles import Font, Alignment, Border, Side
import xlsxwriter

import numpy as np
import matplotlib.pyplot as plt

from openpyxl.utils.dataframe import dataframe_to_rows
import os


# global color values
TABLE1_DARK = "#4BACC6"
TABLE1_LIGHT = "#DAEEF3"

TABLE2_DARK = "#F79646"
TABLE2_LIGHT = "#FDE9D9"


def _style_range(ws, cell_range, border=Border()):
	top = Border(top=border.top)
	left = Border(left=border.left)
	right = Border(right=border.right)
	bottom = Border(bottom=border.bottom)

	first_cell = ws[cell_range.split(":")[0]]
	rows = ws[cell_range]

	for cell in rows[0]:
		cell.border = cell.border + top
	for cell in rows[-1]:
		cell.border = cell.border + bottom

	for row in rows:
		l = row[0]
		r = row[-1]
		l.border = l.border + left
		r.border = r.border + right


def _get_scale_draw_values(sample_dropna_values, values_four):
	signs = np.sign(sample_dropna_values)
	draw_values = signs * np.log10(abs(sample_dropna_values) + 1)

	draw_value_4_signs = [np.sign(values_four[i]) for i in range(4)]
	draw_value_4_scale = [np.log10(abs(values_four[i]) + 1) for i in range(4)]
	draw_value_4 = [draw_value_4_signs[i] * draw_value_4_scale[i] for i in range(4)]

	return draw_values, draw_value_4


def _draw_texts(draw_value_4, mark, text_values, y_low, y_up, date_flag=False):

	color_dark = TABLE1_DARK if mark == 1 else TABLE2_DARK
	color_light = TABLE1_LIGHT if mark == 1 else TABLE2_LIGHT
	plt.axvline(x=draw_value_4[0], color=color_dark, linestyle='--', linewidth=1.5)
	plt.axvline(x=draw_value_4[3], color=color_dark, linestyle='--', linewidth=1.5)

	if date_flag:
		plt.text(draw_value_4[0], y_low + (y_up - y_low) * 0.1 * mark, 'max:' + str(text_values[1]), 
			ha="center", va="center", bbox=dict(boxstyle="square", facecolor=color_light, edgecolor='none'))
		plt.text(draw_value_4[3], y_low + (y_up - y_low) * (0.6 + 0.1 * mark), 'min:' + str(text_values[0]), 
			ha="center", va="center", bbox=dict(boxstyle="square", facecolor=TABLE1_LIGHT, edgecolor='none'))
	else:
		plt.axvline(x=draw_value_4[1], color=color_dark, linestyle='--', linewidth=1.5)
		plt.axvline(x=draw_value_4[2], color=color_dark, linestyle='--', linewidth=1.5)

		indicators = ['min', 'mean', 'median', 'max']
		for i in range(4):
			text = '%s:'%(indicators[i]) + str(round(text_values[i], 3))
			plt.text(draw_value_4[i], y_low + (y_up - y_low) * (0.2 * i + 0.1 * mark), text,
				ha="center", va="center", bbox=dict(boxstyle="square", facecolor=color_light, edgecolor='none'))


def _adjust_column(ws, col_height, col_heights=None, adjust_type=None):
	col_widths = {}
	for i, col in enumerate(ws.columns):
		col_name = xlsxwriter.utility.xl_col_to_name(i)
		col_widths[col_name] = 0
		for cell in col:
			cell.alignment = Alignment(horizontal='left', wrap_text=True)
			if cell:
				try:
					cell_length = len(str(cell.value))
				except:
					cell_length = len(cell.value)
				if cell_length > col_widths[col_name]:
					col_widths[col_name] = cell_length

	for key in col_widths.keys():
		col_widths[key] *= 1.5

	for i, col in enumerate(range(ws.max_column)):
		col_name = xlsxwriter.utility.xl_col_to_name(i)
		ws.column_dimensions[col_name].width = np.min([col_widths[col_name], 100])

	if adjust_type == 'str':
		ws.column_dimensions['C'].width = col_widths['B']
		for i in range(ws.max_row + 1):
			try:
				ws.row_dimensions[i].height = col_heights[i]
			except:
				ws.row_dimensions[i].height = col_height
	else:
		for i in range(ws.max_row + 1):
			ws.row_dimensions[i].height = col_height


def _insert_df(result_df, ws, header=False, head_color=True, head_style='Accent5'):
	max_col = result_df.shape[1]
	for r_idx, r in enumerate(dataframe_to_rows(result_df, index=False, header=header)):
		ws.append(r)
		for cell_idx, cell in enumerate(ws.iter_cols(max_col=max_col, min_row=ws.max_row, max_row=ws.max_row)):
			cell = cell[0]
			cell.font = Font(name='Calibri', size=11)
			if r_idx == 0:
				if head_color:
					cell.style = head_style
				head_row = ws.max_row
			else:
				if cell_idx == 0:
					cell.font = Font(bold=True)
	return head_row


def _insert_numeric_results(numeric_results, ws, col_height, img_dir, date_flag=False):
	# construct the thin border
	thin = Side(border_style="thin", color="000000")
	border = Border(top=thin, left=thin, right=thin, bottom=thin)

	# loop and output the results
	for result in numeric_results:
		column = result['column']
		if not 'result_df' in result.keys():
			ws.append([column, result['error_msg']])
			for col in ['A', 'B']:
				ws['%s%d' %(col, ws.max_row)].style = 'Bad'
			ws.append([''])
			continue

		result_df = result['result_df']
		result_df = result_df[['feature', 'value', 'graph']]
		head_row = _insert_df(result_df, ws)

		# merge cells for the graph
		ws.merge_cells('C%d:C%d' %(head_row+1, head_row+result_df.shape[0]-1))
		_style_range(ws, 'A%d:C%d'%(head_row, head_row+result_df.shape[0]-1), border=border)
		ws['C%d' %(head_row+1)].border = Border(top=None, left=None, right=thin, bottom=thin)
		
		# add gap
		ws.append([''])

		# insert graph
		try:
			if date_flag:
				img = openpyxl.drawing.image.Image(os.path.join(img_dir, '%s.png' %('%s_numeric' %(column))))
			else:
				img = openpyxl.drawing.image.Image(os.path.join(img_dir, '%s.png' %(column)))
			ws.add_image(img, 'C%d' %(head_row+1))
		except:
			continue
	# adjust worksheet
	_adjust_column(ws, col_height)
	ws.column_dimensions['C'].width = 90
