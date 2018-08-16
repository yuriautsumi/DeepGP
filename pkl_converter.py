import os
import csv
import pickle

data_path = '/Users/yuriautsumi/IAP_2018_UROP/kerasGP/'
pkl_data_path = 'pkl_adni_data_all_norm_MMSE.pkl'
csv_data_path = 'csv_data'

# csv to pkl
with open(data_path + 'adni_data_all_norm_MMSE.csv', 'r') as f:
	csv_data = []
	for row in csv.reader(f):
		csv_data.append(row)
with open(data_path + pkl_data_path, 'wb') as f:
        pickle.dump(csv_data, f, pickle.HIGHEST_PROTOCOL)

        
##with open (pkl_data_path + "\\" + file.split(".csv")[0] + ".pkl", 'wb') as f:
##	pickle.dump(csv_data, f, pickle.HIGHEST_PROTOCOL)
##
