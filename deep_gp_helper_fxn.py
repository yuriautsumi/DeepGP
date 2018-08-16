
########################################
## HELPER FUNCTIONS FOR DEEP GP MODEL ##
########################################

# Note: extract_data functions have been modified for adni_adas13_100_fl1_l4.csv data 
from call_gpml import *
from get_activation import *
from compute_error import *
from call_pgp import *
from call_deep_pgp import * 
from deep_gp_helper_fxn import *

import numpy as np 
import pickle as pkl 
import os 
import csv
import scipy.io

from keras import regularizers 
from keras import initializers 
from keras import backend as K 
from keras.optimizers import Adadelta 
from keras.layers import Input, Dense, Dropout
from keras.models import Model as kerasModel
from keras.models import Sequential 
from keras.callbacks import EarlyStopping
from kgp.models import Model as kgpModel
from kgp.layers import GP 

def load_data(csv_data_dir):
    # Input: csv data directory 
    # Output: numpy array of all csv data 
    
    pkl_data_dir = csv_data_dir[:-3] + 'pkl'
    
    if not os.path.exists(pkl_data_dir): # if pkl data does not exist, make pkl file 
        with open(csv_data_dir, 'r') as f:
        	csv_data = []
        	for row in csv.reader(f):
        		csv_data.append(row)
        with open(pkl_data_dir, 'wb') as f:
                pkl.dump(csv_data, f, pkl.HIGHEST_PROTOCOL)
                
    with open(pkl_data_dir, 'rb') as f:
        data_all = pkl.load(f)
        data_all = map_nlist(data_all)
        data_all = np.array(data_all)
        
    return data_all 

def extract_data_4fold(fold_tst_ind, fold_tr_ind, ID_all, data_all): 
    # Input: list of test IDs, list of training IDs, list of IDs, array of all data 
    # Output: source, test, and adaptation data (features and labels) 
    
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
    x_dict_s = {key:value for key, value in X_all.items() if key in fold_tr_ind}
    y_dict_s = {key:value for key, value in Y_all.items() if key in fold_tr_ind}
    
    x_list_s = tuple(x_dict_s.values())
    y_list_s = tuple(y_dict_s.values())

    x_s = np.vstack(x_list_s)
    y_s = np.vstack(y_list_s)
    
    # Create x_a, y_a 
    x_a = {key:value for key, value in X_all.items() if key in fold_tst_ind}
    y_a = {key:value for key, value in Y_all.items() if key in fold_tst_ind}
    
    # Ground truth 
    g_t = y_a 

    # Indicators 
    tst_inds = {key:value for key, value in value_all.items() if key in fold_tst_ind}
    tr_inds = {key:value for key, value in value_all.items() if key in fold_tr_ind}
    
    # Create g_t_all 
    g_list_t = tuple(g_t.values())
    g_t_all = np.vstack(g_list_t)
    
    # Create xtest_all, ytest_all
    x_dict_test_all = {key:value for key, value in X_all.items() if key in fold_tst_ind}
    y_dict_test_all = {key:value for key, value in Y_all.items() if key in fold_tst_ind}
    
    x_list_test_all = tuple(x_dict_test_all.values())
    y_list_test_all = tuple(y_dict_test_all.values())
    
    xtest_all = np.vstack(x_list_test_all)
    ytest_all = np.vstack(y_list_test_all)
    
    return x_dict_s, y_dict_s, x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_inds, tr_inds

def extract_data_5fold(fold, ID_all, data_all):
    # Input: fold number, list of IDs, array of all data 
    # Output: source, test, and adaptation data (features and labels) 

    tst_ind = ID_all[fold*20:fold*20+20] # MODIFIED FOR 20/80 SPLIT - 5 FOLDS 
#    tst_ind = ID_all[fold*10:fold*10+10]
    tr_ind_source = np.setdiff1d(ID_all, tst_ind)
    
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
    
    return x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source

def extract_data_10fold(fold, ID_all, data_all):
    # Input: fold number, list of IDs, array of all data 
    # Output: source, test, and adaptation data (features and labels) 

    tst_ind = ID_all[fold*10:fold*10+10]
    tr_ind_source = np.setdiff1d(ID_all, tst_ind)
    
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
    
    # Indicators 
    tst_inds = {key:value for key, value in value_all.items() if key in tst_ind}
    tr_inds = {key:value for key, value in value_all.items() if key in tr_ind_source}
    
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
    
    return x_dict_s, y_dict_s, x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source, tst_inds, tr_inds 

def call_base_model(x_s, y_s, xtest_all, ytest_all, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    
    # Build base model 
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
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
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified input (z)
    x_s_activations = get_activations(base_model, x_s)
    z = x_s_activations[-2] 
    
    return base_weights, z, base_model

def call_base_model_drop_reg(x_s, y_s, xtest_all, ytest_all, drop, regularizer, deep_layer_dims = 128, epochs=5): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    
    # Build base model 
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu', kernel_regularizer=regularizers.l2(regularizer), activity_regularizer=regularizers.l1(regularizer))(inputs)
    drop_layer = Dropout(drop)(deep_layer)
    outputs = Dense(y_s.shape[1], activation = 'linear')(drop_layer) #last layer - units equal output dimension
    
    base_model = kerasModel(inputs, outputs)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
         
    # Fit on training data
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s, y = y_s,
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = epochs,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified input (z)
    x_s_activations = get_activations(base_model, x_s)
    z = x_s_activations[-2] 
    
    return base_weights, z, base_model

def rebuild_base_model(x_s, y_s, base_weights, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    # Note: data is in order that they will be used to train base model 
    
    # Build base model 
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
    outputs = Dense(y_s.shape[1], activation = 'linear')(deep_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    # Set weights and compile 
    base_model.set_weights(base_weights)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
    
    return base_model

def call_base_model_m3(x_s1, y_s1, xtest_all1, ytest_all1, x_s2, y_s2, xtest_all2, ytest_all2, x_s3, y_s3, xtest_all3, ytest_all3, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    # Note: data is in order that they will be used to train base model 
    
    # Build base model 
    assert x_s1.shape[1] == x_s2.shape[1]
    assert x_s1.shape[1] == x_s3.shape[1]
    assert y_s1.shape[1] == y_s2.shape[1]
    assert y_s1.shape[1] == y_s3.shape[1]
    inputs = Input(shape = (x_s1.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
    outputs = Dense(y_s1.shape[1], activation = 'linear')(deep_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
         
    # Fit on training data 1
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s1, y = y_s1,
                      validation_data = (xtest_all1, ytest_all1), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)

    # Fit on training data 2
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s2, y = y_s2,
                      validation_data = (xtest_all2, ytest_all2), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)

    # Fit on training data 3
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s3, y = y_s3,
                      validation_data = (xtest_all3, ytest_all3), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified inputs (z1, z2, z3)
    x_s_activations1 = get_activations(base_model, x_s1)
    z1 = x_s_activations1[-2] 
    x_s_activations2 = get_activations(base_model, x_s2)
    z2 = x_s_activations2[-2] 
    x_s_activations3 = get_activations(base_model, x_s3)
    z3 = x_s_activations3[-2] 
    
    return base_weights, z1, z2, z3, base_model

def call_base_model_m3_all(x_s, y_s, xtest_all, ytest_all, g1_x_s, g2_x_s, g3_x_s, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    
    # Build base model 
    assert x_s.shape[1] == xtest_all.shape[1]
    assert y_s.shape[1] == ytest_all.shape[1]
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
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
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified inputs (g1_z, g2_z, g3_z)
    x_s_activations1 = get_activations(base_model, g1_x_s)
    g1_z = x_s_activations1[-2] 
    x_s_activations2 = get_activations(base_model, g2_x_s)
    g2_z = x_s_activations2[-2] 
    x_s_activations3 = get_activations(base_model, g3_x_s)
    g3_z = x_s_activations3[-2] 
    
    return base_weights, g1_z, g2_z, g3_z, base_model

def call_base_model_m3_all_drop(x_s, y_s, xtest_all, ytest_all, g1_x_s, g2_x_s, g3_x_s, drop, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dropout, dimensions for deep layer (default = 128)
    
    # Build base model 
    assert x_s.shape[1] == xtest_all.shape[1]
    assert y_s.shape[1] == ytest_all.shape[1]
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
    drop_layer = Dropout(drop)(deep_layer)
    outputs = Dense(y_s.shape[1], activation = 'linear')(drop_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
         
    # Fit on training data
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s, y = y_s,
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified inputs (g1_z, g2_z, g3_z)
    x_s_activations1 = get_activations(base_model, g1_x_s)
    g1_z = x_s_activations1[-2] 
    x_s_activations2 = get_activations(base_model, g2_x_s)
    g2_z = x_s_activations2[-2] 
    x_s_activations3 = get_activations(base_model, g3_x_s)
    g3_z = x_s_activations3[-2] 
    
    return base_weights, g1_z, g2_z, g3_z, base_model

def call_base_model_m3_all_reg(x_s, y_s, xtest_all, ytest_all, g1_x_s, g2_x_s, g3_x_s, regularizer, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    
    # Build base model 
    assert x_s.shape[1] == xtest_all.shape[1]
    assert y_s.shape[1] == ytest_all.shape[1]
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu', kernel_regularizer=regularizers.l2(regularizer), activity_regularizer=regularizers.l1(regularizer))(inputs)
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
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = 5,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified inputs (g1_z, g2_z, g3_z)
    x_s_activations1 = get_activations(base_model, g1_x_s)
    g1_z = x_s_activations1[-2] 
    x_s_activations2 = get_activations(base_model, g2_x_s)
    g2_z = x_s_activations2[-2] 
    x_s_activations3 = get_activations(base_model, g3_x_s)
    g3_z = x_s_activations3[-2] 
    
    return base_weights, g1_z, g2_z, g3_z, base_model

def call_base_model_m3_all_drop_reg(x_s, y_s, xtest_all, ytest_all, g1_x_s, g2_x_s, g3_x_s, drop, regularizer, deep_layer_dims = 128, epochs=5): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    
    # Build base model 
    assert x_s.shape[1] == xtest_all.shape[1]
    assert y_s.shape[1] == ytest_all.shape[1]
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu', kernel_regularizer=regularizers.l2(regularizer), activity_regularizer=regularizers.l1(regularizer))(inputs)
    drop_layer = Dropout(drop)(deep_layer)
    outputs = Dense(y_s.shape[1], activation = 'linear')(drop_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
         
    # Fit on training data
    cb = [EarlyStopping(monitor = 'loss',
                        min_delta = 0,
                        patience = 2,
                        verbose = 1, 
                        mode = 'auto')]
    
    base_model.fit(x = x_s, y = y_s,
                      validation_data = (xtest_all, ytest_all), 
                      batch_size = 100,
                      epochs = epochs,
                      callbacks = cb,
                      verbose = 1)
    
    # Get trained weights of base model 
    base_weights = base_model.get_weights()
    print(base_model.summary())
    # Get modified inputs (g1_z, g2_z, g3_z)
    x_s_activations1 = get_activations(base_model, g1_x_s)
    g1_z = x_s_activations1[-2] 
    x_s_activations2 = get_activations(base_model, g2_x_s)
    g2_z = x_s_activations2[-2] 
    x_s_activations3 = get_activations(base_model, g3_x_s)
    g3_z = x_s_activations3[-2] 
    
    return base_weights, g1_z, g2_z, g3_z, base_model

def rebuild_base_model_m3(x_s1, y_s1, base_weights, deep_layer_dims = 128): 
    # Input: source data (features and labels), validation data (features and labels), dimensions for deep layer (default = 128)
    # Output: base model 
    # Note: data is in order that they will be used to train base model 
    
    # Build base model 
    inputs = Input(shape = (x_s1.shape[1], ))
    deep_layer = Dense(deep_layer_dims, activation = 'relu')(inputs)
    outputs = Dense(y_s1.shape[1], activation = 'linear')(deep_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    # Set weights and compile 
    base_model.set_weights(base_weights)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
    
    return base_model

def save_to_mat(MAT_FOLDER_DIR, ID, x_s, y_s, x_a, y_a, g_t, model, m_s, s_s, tst_inds, optimized_params=None, gp_layer=None, kgp_model=True, tst_pat=True):
    # Input: .mat variable folder directory, patient ID, data, optimized parameters (if GPML model), model, gp_layer (if KGP model), gpml predictions, indicators for test (or training) data, kgp model boolean (True if KGP model, False if GPML model), test patient boolean 
    # Output: patient predictions, adaptation data 
    # Saves variables as .mat files for compute_results.m script 
    
    ID = int(ID)
    
    x_a_patient = x_a[ID][:-1,:]
    y_a_patient = y_a[ID][:-1,:]
    xtest = x_a[ID]
    
    g_t_patient = g_t[ID]
    
    if tst_pat: 
        tst_ind_patient = tst_inds[ID]
    else:
        tr_ind_patient = tst_inds[ID]
    
    if kgp_model: # if KGP model 
        #note: this returns modified (activation) versions of x_s, x_a_patient, xtest 
        m_s_patient, s_s_patient, ls, mul, var, sn2, x_s, x_a_patient, xtest = call_deep_source(model, gp_layer, x_s, y_s, x_a_patient, xtest) 
#        m_s_patient, s_s_patient, ls, mul, var, sn2, x_s_modified, x_a_modified, xtest_modified = call_deep_source(deep_model, gp_layer, x_s, y_s, x_a_patient, xtest) 
        
    else: # if GPML model 
        ls = np.exp(optimized_params['cov'][0])
        mul = [x_a_patient.shape[1]]
        var = np.exp(2*optimized_params['cov'][1])
        sn2 = np.exp(2*optimized_params['lik'])
        
        m_s_patient, s_s_patient = predictGP(x_s, y_s, xtest, model)
    
    #save adaptation variables to mat file 
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_x_a.mat'%(ID)), {'x_a': x_a_patient})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_y_a.mat'%(ID)), {'y_a': y_a_patient})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_x_s.mat'%(ID)), {'x_s': x_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_y_s.mat'%(ID)), {'y_s': y_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_xtest.mat'%(ID)), {'xtest': xtest})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_m_s.mat'%(ID)), {'m_s': m_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_s_s.mat'%(ID)), {'s_s': s_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_ls.mat'%(ID)), {'ls': float(ls)})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_mul.mat'%(ID)), {'mul': mul})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_var.mat'%(ID)), {'var': float(var)})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_sn2.mat'%(ID)), {'sn2': float(sn2)})
    
    K_ts_star_all = new_K(ls, mul, var, x_s, xtest) 
    K_tt_all = new_K(ls, mul, var, x_a_patient)
    K_t_star_all = new_K(ls, mul, var, x_a_patient, xtest)
    
    K_s = new_K(ls, mul, var, x_s) 
    dim_K_s = K_s.shape[0]
    L_arg = K_s + sn2*np.identity(dim_K_s) 
    L = jitChol(L_arg) 
    
    np.nan_to_num(L,copy=False)
#        alpha_denom = np.linalg.lstsq(L,y_s)[-1] 
    alpha_denom = np.linalg.solve(L, y_s)
    np.nan_to_num(alpha_denom,copy=False)
    alpha = np.linalg.solve(L.transpose(),alpha_denom)
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_ts_star_all.mat'%(ID)), {'K_ts_star_all': K_ts_star_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_tt_all.mat'%(ID)), {'K_tt_all': K_tt_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_t_star_all.mat'%(ID)), {'K_t_star_all': K_t_star_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_L.mat'%(ID)), {'L': L})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_alpha.mat'%(ID)), {'alpha': alpha})
    
#        K_ts_star_all = new_K(ls, mul, var, x_a_patient, xtest) #20 x 21, x_a x xtest <-- same as K_t_star_all 
    k_star_star_all = new_Kdiag(var, xtest) #1 x 21 
#        K_s_all = new_K(ls, mul, var, x_a_patient) #20 x 20 <-- same as K_tt_all
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_k_star_star_all.mat'%(ID)), {'k_star_star_all': k_star_star_all})
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_g_t_patient.mat'%(ID)), {'g_t_patient': g_t_patient}) 
    
    if tst_pat: 
        scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_tst_ind_patient.mat'%(ID)), {'tst_ind_patient': tst_ind_patient}) 
    else: 
        scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_tr_ind_patient.mat'%(ID)), {'tr_ind_patient': tr_ind_patient}) 
    
    return m_s_patient, s_s_patient, y_a_patient 

def save_to_mat_final(MAT_FOLDER_DIR, ID, x_s, y_s, x_a, y_a, g_t, deep_model, gp_layer, m_s, s_s, tst_inds, base_weights, deep_layer_dims = 128, tst_pat=True):
    # Input: .mat variable folder directory, patient ID, data, deep model, gp_layer, source model predictions, test (or training) patient indicators, base model weights, test patient boolean 
    # Output: patient predictions, adaptation data 
    # Saves variables as .mat files for compute_results.m script 
    
    ID = int(ID)
    
    x_a_patient = x_a[ID][:-1,:]
    y_a_patient = y_a[ID][:-1,:]
    xtest = x_a[ID]
    
    #rebuild baseline model 
    base_model = rebuild_base_model(x_s, y_s, base_weights, deep_layer_dims = deep_layer_dims)
    
    #predictions for baseline model 
    m_b_patient = base_model.predict(xtest)
    
    g_t_patient = g_t[ID]
    
    if tst_pat: 
        tst_ind_patient = tst_inds[ID]
    else:
        tr_ind_patient = tst_inds[ID]
    
    #predictions for deep model 
    #note: this returns modified (activation) versions of x_s, x_a_patient, xtest 
    m_s_patient, s_s_patient, ls, mul, var, sn2, x_s, x_a_patient, xtest = call_deep_source(deep_model, gp_layer, x_s, y_s, x_a_patient, xtest) 
    
    #save adaptation variables to mat file 
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_m_b_patient.mat'%(ID)), {'m_b_patient': m_b_patient})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_x_a.mat'%(ID)), {'x_a': x_a_patient})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_y_a.mat'%(ID)), {'y_a': y_a_patient})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_x_s.mat'%(ID)), {'x_s': x_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_y_s.mat'%(ID)), {'y_s': y_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_xtest.mat'%(ID)), {'xtest': xtest})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_m_s.mat'%(ID)), {'m_s': m_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_s_s.mat'%(ID)), {'s_s': s_s})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_ls.mat'%(ID)), {'ls': float(ls)})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_mul.mat'%(ID)), {'mul': mul})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_var.mat'%(ID)), {'var': float(var)})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_sn2.mat'%(ID)), {'sn2': float(sn2)})
    
    K_ts_star_all = new_K(ls, mul, var, x_s, xtest) 
    K_tt_all = new_K(ls, mul, var, x_a_patient)
    K_t_star_all = new_K(ls, mul, var, x_a_patient, xtest)
    
    K_s = new_K(ls, mul, var, x_s) 
    dim_K_s = K_s.shape[0]
    L_arg = K_s + sn2*np.identity(dim_K_s) 
    L = jitChol(L_arg) 
    
    np.nan_to_num(L,copy=False)
#        alpha_denom = np.linalg.lstsq(L,y_s)[-1] 
    alpha_denom = np.linalg.solve(L, y_s)
    np.nan_to_num(alpha_denom,copy=False)
    alpha = np.linalg.solve(L.transpose(),alpha_denom)
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_ts_star_all.mat'%(ID)), {'K_ts_star_all': K_ts_star_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_tt_all.mat'%(ID)), {'K_tt_all': K_tt_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_K_t_star_all.mat'%(ID)), {'K_t_star_all': K_t_star_all})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_L.mat'%(ID)), {'L': L})
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_alpha.mat'%(ID)), {'alpha': alpha})
    
#        K_ts_star_all = new_K(ls, mul, var, x_a_patient, xtest) #20 x 21, x_a x xtest <-- same as K_t_star_all 
    k_star_star_all = new_Kdiag(var, xtest) #1 x 21 
#        K_s_all = new_K(ls, mul, var, x_a_patient) #20 x 20 <-- same as K_tt_all
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_k_star_star_all.mat'%(ID)), {'k_star_star_all': k_star_star_all})
    
    scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_g_t_patient.mat'%(ID)), {'g_t_patient': g_t_patient})    
    
    if tst_pat: 
        scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_tst_ind_patient.mat'%(ID)), {'tst_ind_patient': tst_ind_patient}) 
    else: 
        scipy.io.savemat(os.path.join(MAT_FOLDER_DIR, 'id_%s_tr_ind_patient.mat'%(ID)), {'tr_ind_patient': tr_ind_patient}) 
    
    return m_s_patient, s_s_patient, y_a_patient 

def map_nlist(nlist, fun = lambda x: float(x)):
    # Input: nested list 
    # Output: nested list with float elements 
    
    new_list=[]
    
    for i in range(len(nlist)):
        if isinstance(nlist[i],list):
            new_list += [map_nlist(nlist[i],fun)]
        else:
            new_list += [fun(nlist[i])]
            
    return new_list

#def var_define(data, names):
#    # Input: list of data, list of variable names (string)
#    # Output: variables assigned to corresponding data 
#    
#    length = len(names)
#    
#    for i in range(0, length):
#        globals()[names[i]] = data[i]
