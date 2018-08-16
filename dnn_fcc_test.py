'''
DNN + FCC Test Model: 
    
- Optimize DNN until convergence 
- Callback 
- 8 fold for train, 1 fold for callback, 1 fold for test 
- Optimize DNN + FCC until convergence, get activation, pass through MATLAB to get results 
- Then compare to MATLAB results for regular data 
'''

# Import methods
from call_gpml import *
from get_activation import *
from compute_error import *
from call_pgp import *
from call_deep_pgp import * 
from deep_gp_helper_fxn import *

# Import libraries 
import os
import csv
import numpy as np
import itertools 
import pathlib
import pickle as pkl
import scipy.io

from keras import initializers 
from keras import backend as K 
from keras.optimizers import Adadelta 
from keras.layers import Input, Dense
from keras.models import Model as kerasModel
from keras.models import Sequential 
from keras.callbacks import EarlyStopping
from kgp.models import Model as kgpModel
from kgp.layers import GP 

from random import shuffle 

# Set to use GPU... '0' or '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

######################
##### PREP DATA ######
######################

print('----- PREPARING DATA -----')

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv') #data from Oggi
PKL_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.pkl')

ID_DIR = os.path.join(CURRENT_DIR, 'Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv')

# Create list of IDs
with open(ID_DIR, 'r') as f:
    reader = csv.reader(f)
    ID_all = list(reader)
    ID_all = list(itertools.chain.from_iterable(ID_all)) #flatten list
    ID_all = list(map(int, ID_all)) #cast each element as int

# Load CSV data 
data_all = load_data(CSV_DATA_DIR)

# Define callback
cb = [EarlyStopping(monitor = 'loss',
                    min_delta = 0,
                    patience = 2,
                    verbose = 1, 
                    mode = 'auto')]

'''
DNN + FCC Test Model: 
    
- Optimize DNN until convergence 
- Callback 
- 8 fold for train, 1 fold for callback, 1 fold for test 
- Optimize DNN + FCC until convergence, get activation, pass through MATLAB to get results 
- Then compare to MATLAB results for regular data 
'''

print('----- DNN + FCC Test Model -----')

i = 0 

tst_ind = ID_all[0:10]
val_ind = ID_all[10:20]
tr_ind_source = np.setdiff1d(ID_all, tst_ind+val_ind)

# Create X_all, Y_all, value_all         
X_all = {}
Y_all = {}
value_all = {}

for ID in ID_all:
    patient_data_all = data_all[data_all[:,0] == ID] #retrieves all rows where first column has ID 
    x_patient = patient_data_all[:, 1:-8]
    y_patient = patient_data_all[:, -8:-7]
    value_patient = patient_data_all[:, -4:-3]
#        x_patient = patient_data_all[:, 3:-2]
#        y_patient = patient_data_all[:, -2:-1]
#        value_patient = patient_data_all[:, -1:]
    X_all[ID] = x_patient 
    Y_all[ID] = y_patient 
    value_all[ID] = value_patient 

# Create x_s, y_s
x_dict_s = {key:value for key, value in X_all.items() if key in tr_ind_source}
y_dict_s = {key:value for key, value in Y_all.items() if key in tr_ind_source}

x_list_s = tuple(x_dict_s.values())
y_list_s = tuple(y_dict_s.values())

x_s = np.vstack(x_list_s)
y_s = np.vstack(y_list_s)

# Create x_a, y_a 
x_a = {key:value for key, value in X_all.items() if key in tst_ind}
y_a = {key:value for key, value in Y_all.items() if key in tst_ind}

# Ground truth 
g_t = y_a 

# Create g_t_all 
g_list_t = tuple(g_t.values())
g_t_all = np.vstack(g_list_t)

# Create xtest_all, ytest_all
x_dict_test_all = {key:value for key, value in X_all.items() if key in tst_ind}
y_dict_test_all = {key:value for key, value in Y_all.items() if key in tst_ind}

x_list_test_all = tuple(x_dict_test_all.values())
y_list_test_all = tuple(y_dict_test_all.values())

xtest_all = np.vstack(x_list_test_all)
ytest_all = np.vstack(y_list_test_all)

# Create x_val, y_val
x_val = {key:value for key, value in X_all.items() if key in val_ind}
y_val = {key:value for key, value in Y_all.items() if key in val_ind}

x_list_val_all = tuple(x_val.values())
y_list_val_all = tuple(y_val.values())

x_val_all = np.vstack(x_list_val_all)
y_val_all = np.vstack(y_list_val_all)

    
print('----- BUILDING BASE MODEL -----')

# Build base model 
inputs = Input(shape = (x_s.shape[1], ))
deep_layer = Dense(256, activation = 'relu')(inputs)
outputs = Dense(y_s.shape[1], activation = 'linear')(deep_layer) #last layer - units equal output dimension 

base_model = kerasModel(inputs, outputs)

base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
     
# Fit on training data
cb = [EarlyStopping(monitor = 'loss',
                    min_delta = 0,
                    patience = 2,
                    verbose = 1, 
                    mode = 'auto')]

base_model.fit(x = x_s, y = y_s,
                  validation_data = (x_val_all, y_val_all), 
                  batch_size = 100,
                  epochs = 5,
                  callbacks = cb,
                  verbose = 1)

# Get modified inputs
x_s_activations = get_activations(base_model, x_s)
x_s_activations = x_s_activations[-2] 

xtest_activations = get_activations(base_model, xtest_all)
xtest_activations = xtest_activations[-2]

source_data = np.hstack((x_s_activations, y_s))
test_data = np.hstack((xtest_activations, ytest_all))

# Save data 
np.savetxt(os.path.join(CURRENT_DIR, 'dnn_fcc_xs_256.csv'), source_data, delimiter=',')
np.savetxt(os.path.join(CURRENT_DIR, 'dnn_fcc_xtest_256.csv'), test_data, delimiter=',')
