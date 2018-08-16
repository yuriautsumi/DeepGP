## Model 0 

#####################################
## Wrapper for Multi-GP Deep Model ##
#####################################

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
import time 

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
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

######################
##### PREP DATA ######
######################

print('----- PREPARING DATA -----')

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv') #data from Oggi
PKL_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.pkl')

ID_DIR = os.path.join(CURRENT_DIR, 'Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv')

RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'kgp_results')
pathlib.Path(RESULTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

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




#########################################
## MODEL 0 - GP Test: Generic GP Model ## 
#########################################

'''
Obtains results from generic GP model. 
- GP Model: Uses MATLAB GPML package in backend to train and predict on GP model. 
'''

print('----- MODEL 0 - GP Test: Generic GP Model -----')

# Create folder 
M0_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'model_0')
pathlib.Path(M0_FOLDER_DIR).mkdir(parents=True, exist_ok=True)

MAT_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'm0_mat')
pathlib.Path(MAT_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

# Create CSV files 
M0_GT_MEAN_DIR = os.path.join(M0_FOLDER_DIR, 'gt_mean.csv')
M0_ERROR_DIR = os.path.join(M0_FOLDER_DIR, 'mse.csv')
M0_HYP_DIR = os.path.join(M0_FOLDER_DIR, 'hyp.csv')

# Load patient classification data 
CLASSIF_CSV_DIR = os.path.join(CURRENT_DIR, 'patient_classification.csv')
classif_data = np.genfromtxt(CLASSIF_CSV_DIR, delimiter=',')

# Get indices of patients in each group 
g1_inds = list(np.where(classif_data[:,1] == 1)[0])
g2_inds = list(np.where(classif_data[:,1] == 2)[0])
g3_inds = list(np.where(classif_data[:,1] == 3)[0])

g1_fold_inds = np.array_split(g1_inds, 4)
g2_fold_inds = np.array_split(g2_inds, 4)
g3_fold_inds = np.array_split(g3_inds, 4)

g1_ID_all = list(classif_data[:,0][g1_inds])
g2_ID_all = list(classif_data[:,0][g2_inds])
g3_ID_all = list(classif_data[:,0][g3_inds])

# Loop for 4 folds 
for i in range(0, 4): #0 to 3

    fold_num = i+1

    print('FOLD:', fold_num)
    
    MAT_FOLD_FOLDER_DIR = os.path.join(MAT_FOLDER_DIR, 'fold_%s'%(i+1))
    pathlib.Path(MAT_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    # Extract necessary data from data_all
    g1_fold_tst_ind = list(classif_data[:, 0][g1_fold_inds[i]])
    g2_fold_tst_ind = list(classif_data[:, 0][g2_fold_inds[i]])
    g3_fold_tst_ind = list(classif_data[:, 0][g3_fold_inds[i]])
    
    g1_fold_tr_ind = np.setdiff1d(g1_ID_all, g1_fold_tst_ind)
    g2_fold_tr_ind = np.setdiff1d(g2_ID_all, g2_fold_tst_ind)
    g3_fold_tr_ind = np.setdiff1d(g3_ID_all, g3_fold_tst_ind)

    g1_data = extract_data_4fold(g1_fold_tst_ind, g1_fold_tr_ind, g1_ID_all, data_all)
    g2_data = extract_data_4fold(g2_fold_tst_ind, g2_fold_tr_ind, g2_ID_all, data_all)
    g3_data = extract_data_4fold(g3_fold_tst_ind, g3_fold_tr_ind, g3_ID_all, data_all)
    
    group_data = (g1_data, g2_data, g3_data)
    
    g1_x_dict_s, g1_y_dict_s, g1_x_s, g1_y_s, g1_xtest_all, g1_ytest_all, g1_x_a, g1_y_a, g1_g_t, g1_g_t_all, g1_tst_inds, g1_tr_inds = group_data[0]
    g2_x_dict_s, g2_y_dict_s, g2_x_s, g2_y_s, g2_xtest_all, g2_ytest_all, g2_x_a, g2_y_a, g2_g_t, g2_g_t_all, g2_tst_inds, g2_tr_inds = group_data[1]
    g3_x_dict_s, g3_y_dict_s, g3_x_s, g3_y_s, g3_xtest_all, g3_ytest_all, g3_x_a, g3_y_a, g3_g_t, g3_g_t_all, g3_tst_inds, g3_tr_inds = group_data[2]
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    # Group 1: 
    # Calculate initial parameters
    g1_max_x_s = np.amax(g1_x_s, axis = 0)
    g1_min_x_s = np.amin(g1_x_s, axis = 0)
    g1_var_y_s = np.sum(g1_y_s**2)/(len(g1_y_s)-1) - (len(g1_y_s))*np.mean(g1_y_s)**2/(len(g1_y_s)-1) 
    g1_initial_lik = np.log(np.sqrt(0.1*g1_var_y_s))
    g1_initial_cov = np.array([[np.log(np.median(g1_max_x_s - g1_min_x_s))], [np.log(np.sqrt(g1_var_y_s))]])
    print('Group 1 Initial Parameters:', g1_initial_lik, g1_initial_cov) 
    
    # Input features for GP
    g1_dimensions = g1_x_s.shape[1]
    g1_initial_hyp = {'lik': g1_initial_lik, 'mean': [], 'cov': g1_initial_cov}
    g1_train_opt = {}
    g1_inf_method = 'infExact'
    g1_mean_fxn = 'meanZero'
    g1_cov_fxn = 'covSEiso'
    g1_lik_fxn = 'likGauss'
    g1_dlik_fxn = 'dlikExact'
    
    # Train and predict GP model 
    g1_gpml_model, g1_optimized_params = trainGP(g1_x_s, g1_y_s, g1_dimensions, g1_initial_hyp, g1_train_opt, g1_inf_method, g1_mean_fxn, g1_cov_fxn, g1_lik_fxn, g1_dlik_fxn, iters=200) 
    g1_gpml_train_m_s, g1_gpml_train_s_s = predictGP(g1_x_s, g1_y_s, g1_x_s, g1_gpml_model)
    g1_gpml_test_m_s, g1_gpml_test_s_s = predictGP(g1_x_s, g1_y_s, g1_xtest_all, g1_gpml_model)
    
    # Group 2: 
    # Calculate initial parameters
    g2_max_x_s = np.amax(g2_x_s, axis = 0)
    g2_min_x_s = np.amin(g2_x_s, axis = 0)
    g2_var_y_s = np.sum(g2_y_s**2)/(len(g2_y_s)-1) - (len(g2_y_s))*np.mean(g2_y_s)**2/(len(g2_y_s)-1) 
    g2_initial_lik = np.log(np.sqrt(0.1*g2_var_y_s))
    g2_initial_cov = np.array([[np.log(np.median(g2_max_x_s - g2_min_x_s))], [np.log(np.sqrt(g2_var_y_s))]])
    print('Group 2 Initial Parameters:', g2_initial_lik, g2_initial_cov) 
    
    # Input features for GP
    g2_dimensions = g2_x_s.shape[1]
    g2_initial_hyp = {'lik': g2_initial_lik, 'mean': [], 'cov': g2_initial_cov}
    g2_train_opt = {}
    g2_inf_method = 'infExact'
    g2_mean_fxn = 'meanZero'
    g2_cov_fxn = 'covSEiso'
    g2_lik_fxn = 'likGauss'
    g2_dlik_fxn = 'dlikExact'
    
    # Train and predict GP model 
    g2_gpml_model, g2_optimized_params = trainGP(g2_x_s, g2_y_s, g2_dimensions, g2_initial_hyp, g2_train_opt, g2_inf_method, g2_mean_fxn, g2_cov_fxn, g2_lik_fxn, g2_dlik_fxn, iters=200) 
    g2_gpml_train_m_s, g2_gpml_train_s_s = predictGP(g2_x_s, g2_y_s, g2_x_s, g2_gpml_model)
    g2_gpml_test_m_s, g2_gpml_test_s_s = predictGP(g2_x_s, g2_y_s, g2_xtest_all, g2_gpml_model)
    
    # Group 3: 
    # Calculate initial parameters
    g3_max_x_s = np.amax(g3_x_s, axis = 0)
    g3_min_x_s = np.amin(g3_x_s, axis = 0)
    g3_var_y_s = np.sum(g3_y_s**2)/(len(g3_y_s)-1) - (len(g3_y_s))*np.mean(g3_y_s)**2/(len(g3_y_s)-1) 
    g3_initial_lik = np.log(np.sqrt(0.1*g3_var_y_s))
    g3_initial_cov = np.array([[np.log(np.median(g3_max_x_s - g3_min_x_s))], [np.log(np.sqrt(g3_var_y_s))]])
    print('Group 3 Initial Parameters:', g3_initial_lik, g3_initial_cov) 
    
    # Input features for GP
    g3_dimensions = g3_x_s.shape[1]
    g3_initial_hyp = {'lik': g3_initial_lik, 'mean': [], 'cov': g3_initial_cov}
    g3_train_opt = {}
    g3_inf_method = 'infExact'
    g3_mean_fxn = 'meanZero'
    g3_cov_fxn = 'covSEiso'
    g3_lik_fxn = 'likGauss'
    g3_dlik_fxn = 'dlikExact'
    
    # Train and predict GP model 
    g3_gpml_model, g3_optimized_params = trainGP(g3_x_s, g3_y_s, g3_dimensions, g3_initial_hyp, g3_train_opt, g3_inf_method, g3_mean_fxn, g3_cov_fxn, g3_lik_fxn, g3_dlik_fxn, iters=200) 
    g3_gpml_train_m_s, g3_gpml_train_s_s = predictGP(g3_x_s, g3_y_s, g3_x_s, g3_gpml_model)
    g3_gpml_test_m_s, g3_gpml_test_s_s = predictGP(g3_x_s, g3_y_s, g3_xtest_all, g3_gpml_model)
    
    #TRAIN ADAPTATION AND TARGET MODELS
    
    print('----- TRAINING ADAPTATION AND TARGET MODELS -----')
    
    #output mean and variance predictions for source, adaptation, and target models 
    #iterate over test patients 
    g1_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g1_final')
    pathlib.Path(g1_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True)     

    g2_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g2_final')
    pathlib.Path(g2_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True)     
    
    g3_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g3_final')
    pathlib.Path(g3_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True)     
    
    for ID in g1_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        _  = save_to_mat(g1_MAT_FINAL_FOLDER_DIR, ID, g1_x_s, g1_y_s, g1_x_a, g1_y_a, g1_g_t, g1_gpml_model, g1_gpml_test_m_s, g1_gpml_test_s_s, g1_tst_inds, optimized_params=g1_optimized_params, gp_layer=None, kgp_model=False)

    for ID in g2_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        _  = save_to_mat(g2_MAT_FINAL_FOLDER_DIR, ID, g2_x_s, g2_y_s, g2_x_a, g2_y_a, g2_g_t, g2_gpml_model, g2_gpml_test_m_s, g2_gpml_test_s_s, g2_tst_inds, optimized_params=g2_optimized_params, gp_layer=None, kgp_model=False)

    for ID in g3_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        _  = save_to_mat(g3_MAT_FINAL_FOLDER_DIR, ID, g3_x_s, g3_y_s, g3_x_a, g3_y_a, g3_g_t, g3_gpml_model, g3_gpml_test_m_s, g3_gpml_test_s_s, g3_tst_inds, optimized_params=g3_optimized_params, gp_layer=None, kgp_model=False)


##compute all results in MATLAB 
#import matlab.engine 
#eng = matlab.engine.start_matlab()
#eng.compute_final_results(0, nargout=0)
