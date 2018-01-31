from openpyxl.styles import Font, Alignment, Border
import xlsxwriter


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

