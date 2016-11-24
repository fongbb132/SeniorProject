import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def filter_zero_values(energy_array):
	# assumes that energy values are in the  third column
	row_num = 0
	for rows in energy_array:
		if rows[2] < 100:
			energy_array[row_num, 2] = (1/2) * (energy_array[row_num - 1, 2] + energy_array[row_num + 1, 2])
		row_num += 1

	return energy_array

def read_region(path, region, start_year, end_year):
	"""
	open and read an excel file
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



#region_data = read_region("iso_data/", 'ME', start_year=2003, end_year=2016)
region_data = np.load('data/ME.npy')

x = np.arange(0, region_data.shape[0])
y = region_data[:, 2]
# plt.plot(x,y)
# plt.show()

# compare data from 05 and 06
compare_05_06()