import pandas as pd;
def open_file(path):
	"""
	open and read an excel file
	"""
	# open excel file and turn create dictionary of sheets
	xl_file = pd.ExcelFile(path)
	dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}

	# get the data for the region of maine where the data is in the format:
	# [day, hour, demand, drybulb_temp]
	maine_data = dfs['ME']
	maine_data = maine_data.as_matrix()
	maine_data = maine_data[:, [0, 1, 3, 12]]

	# convert pandas timestamp object to ordinal day of the year
	for i in range(maine_data.shape[0]):
		maine_data[i, 0] = maine_data[i, 0].dayofyear
		print maine_data

