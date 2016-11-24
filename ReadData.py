import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_zero_values(energy_array):
	"""
	filter small values from the energy array

	args:
		- energy_array: numpy array containing energy data for days and times
	returns:
		- energy_array: modified from original version to interpolate 0 values
	"""

	# assumes that energy values are in the  third column
	row_num = 0
	for rows in energy_array:
		if rows[2] < 100:
			new_val = energy_array[row_num - 1, 2] + energy_array[row_num + 1, 2]
			energy_array[row_num, 2] = new_val * 0.5
		row_num += 1

	return energy_array

def read_region(path, region, start_year, end_year):
	"""
	open and read an excel file

	args:
		- path: data path of region data
		- region: string containing region abbreviation
		- start_year: what year to start reading data
		- end_year: what year to stop reading data

	returns:
		- region_data_total: concatenated version of data for all years of region specified
	"""
	region_data_total = np.ndarray((1,3))

	for i in range(start_year, end_year+1):
		# open excel file and turn create dictionary of sheets
		current_path = path + "%d" %(i)
		current_path = current_path + "_smd_hourly.xls"
		xl_file = pd.ExcelFile(current_path)
		dfs = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}

		# get the data for the region of maine where the data is in the format:
		# [day, hour, demand, drybulb_temp]
		region_data = dfs[region]
		region_data = region_data.as_matrix()
		region_data = region_data[:, [0, 1, 3]]

		# convert pandas timestamp object to ordinal day of the year
		for i in range(region_data.shape[0]):
			region_data[i, 0] = region_data[i, 0].dayofyear

		region_data_total = np.append(region_data_total, region_data, axis=0)


	region_data_total = region_data_total[1:, :]
	region_data_total = filter_zero_values(region_data_total)
	np.save('data/'+region, region_data_total)
	return region_data_total

def compare_05_06():
	data_05 = read_region("iso_data/", "ME", start_year=2005, end_year=2005)
	data_06 = read_region("iso_data/", "ME", start_year=2006, end_year=2006)
	x = np.arange(0, data_05.shape[0])
	plt.plot(x, data_05[:, 2], 'r')
	plt.plot(x, data_06[:, 2], 'b')
	plt.show()

def split_data(energy_array, train=0.6, val=0.2, test=0.2):
	"""
	splits data into training, validation, and test sets

	args:
		- energy_array: array containing the data
		- train, val, test: how much of data to put into each set (must add to 1)

	returns:
		- energy_train: training data
		- energy_val: validation data
		- energy_test: test data
	"""
	end_dex_train = np.floor(energy_array.shape[0] * train)
	start_dex_val = end_dex_train + 1
	end_dex_val = np.floor(energy_array.shape[0] * (train + val))
	start_dex_test = end_dex_val + 1

	energy_train = energy[0:end_dex_train, :]
	energy_val = energy[start_dex_val:end_dex_val, :]
	energy_test = energy[start_dex_test:, :]

	return energy_train, energy_val, energy_test


#region_data = read_region("iso_data/", 'ME', start_year=2003, end_year=2016)
region_data = np.load('data/ME.npy')

x = np.arange(0, region_data.shape[0])
y = region_data[:, 2]
plt.plot(x,y)
plt.show()

# compare data from 05 and 06
# compare_05_06()