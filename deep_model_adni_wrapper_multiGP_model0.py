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

all_error = []

# Loop for 10 folds 
for i in range(0, 10): #0 to 9 
    
    MAT_FOLD_FOLDER_DIR = os.path.join(MAT_FOLDER_DIR, 'fold_%s'%(i+1))
    pathlib.Path(MAT_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    # Extract necessary data from data_all 
    x_dict_s, y_dict_s, x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source, tst_inds, tr_inds = extract_data_10fold(i, ID_all, data_all)
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    #TODO: start: call_gpml 
    # Calculate initial parameters
    max_x_s = np.amax(x_s, axis = 0)
    min_x_s = np.amin(x_s, axis = 0)
    var_y_s = np.sum(y_s**2)/(len(y_s)-1) - (len(y_s))*np.mean(y_s)**2/(len(y_s)-1) 
    initial_lik = np.log(np.sqrt(0.1*var_y_s))
    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(var_y_s))]])
    print('Initial Parameters:', initial_lik, initial_cov) 
    
    # Input features for GP
    dimensions = x_s.shape[1]
    initial_hyp = {'lik': initial_lik, 'mean': [], 'cov': initial_cov}
    train_opt = {}
    inf_method = 'infExact'
    mean_fxn = 'meanZero'
    cov_fxn = 'covSEiso'
    lik_fxn = 'likGauss'
    dlik_fxn = 'dlikExact'
    
    # Train and predict GP model 
    gpml_model, optimized_params = trainGP(x_s, y_s, dimensions, initial_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn, iters=200) 
    gpml_m_s, gpml_s_s = predictGP(x_s, y_s, xtest_all, gpml_model)
    
    # Save optimized hyperparameters for fold 
    with open(M0_HYP_DIR, 'a') as myfile:
        myfile.write(str(optimized_params).replace('\n', ''))
        myfile.write('\n')
    
    # Compute error
    error = mean_absolute_error(gpml_m_s, ytest_all)
    all_error.append(error)

    #TRAIN ADAPTATION AND TARGET MODELS
    
    print('----- TRAINING ADAPTATION AND TARGET MODELS -----')
    
    error_all = {}
    
    #output mean and variance predictions for source, adaptation, and target models 
    #iterate over test patients 
    MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'final')
    pathlib.Path(MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True)     
    
    for ID in tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        m_s_patient, s_s_patient, y_a_patient  = save_to_mat(MAT_FINAL_FOLDER_DIR, ID, x_s, y_s, x_a, y_a, g_t, gpml_model, gpml_m_s, gpml_s_s, tst_inds, optimized_params=optimized_params, gp_layer=None, kgp_model=False)

##compute all results in MATLAB 
#import matlab.engine 
#eng = matlab.engine.start_matlab()
#eng.compute_final_results(0, nargout=0)
