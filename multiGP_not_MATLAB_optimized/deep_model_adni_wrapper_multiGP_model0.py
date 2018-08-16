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
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

######################
##### PREP DATA ######
######################

print('----- PREPARING DATA -----')

# Define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.csv') #data from Oggi
PKL_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_adas13_100_fl1_l4.pkl')
#CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_data_all_norm_MMSE.csv') #data from Oggi
#PKL_DATA_DIR = os.path.join(CURRENT_DIR, 'pkl_adni_data_all_norm_MMSE.pkl')

ID_DIR = os.path.join(CURRENT_DIR, 'Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv')

RESULTS_FOLDER_DIR = os.path.join(CURRENT_DIR, 'kgp_results')
pathlib.Path(RESULTS_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

##HEADER INFORMATION
##write csv headers 
#with open(GT_MEAN_DIR, 'w') as mean_csv_file:
#    w = csv.writer(mean_csv_file)
#    w.writerow(['ID', 'ground truth', 'base model mu', 'source model mu', 'adapted model mu', 'target model mu'])
#
#with open(GT_MEAN_EXTRACTED_DIR, 'w') as mean_extracted_csv_file:
#    w = csv.writer(mean_extracted_csv_file)
#    w.writerow(['ID', 'ground truth', 'base model mu', 'source model mu', 'adapted model mu', 'target model mu'])
#
#with open(ERROR_DIR, 'w') as error_csv_file:
#    w = csv.writer(error_csv_file)
#    w.writerow(['ID', 'base model error', 'source model error', 'adapted model error', 'target model error'])

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

# Create CSV files 
M0_GT_MEAN_DIR = os.path.join(M0_FOLDER_DIR, 'gt_mean.csv')
M0_ERROR_DIR = os.path.join(M0_FOLDER_DIR, 'mse.csv')
M0_HYP_DIR = os.path.join(M0_FOLDER_DIR, 'hyp.csv')

all_error = []

# Loop for 10 folds 
for i in range(0, 10): #0 to 9 
    
    # Extract necessary data from data_all 
    x_dict_s, y_dict_s, x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source = extract_data_10fold(i, ID_all, data_all)
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    #TODO: start: call_gpml 
    # Calculate initial parameters
    max_x_s = np.amax(x_s, axis = 0)
    min_x_s = np.amin(x_s, axis = 0)
    initial_lik = np.log(np.sqrt(0.1*np.var(y_s)))
    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(np.var(y_s)))]])
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
    gpml_model, optimized_params = trainGP(x_s, y_s, dimensions, initial_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn) 
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
    
    #make folders 
#    GT_MEAN_FOLD_FOLDER = os.path.join(GT_MEAN_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(GT_MEAN_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
    
#    GT_MEAN_EXTRACTED_FOLD_FOLDER = os.path.join(GT_MEAN_EXTRACTED_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(GT_MEAN_EXTRACTED_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
    
#    ERROR_FOLD_FOLDER_DIR = os.path.join(ERROR_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(ERROR_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #make folder 
    
    #output mean and variance predictions for source, adaptation, and target models 
    #iterate over test patients 
    for ID in tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        x_a_patient = x_a[ID][:-1,:]
        y_a_patient = y_a[ID][:-1,:]
        xtest = x_a[ID]
        
        ls = np.exp(optimized_params['cov'][0])
        mul = [x_a_patient.shape[1]]
        var = np.exp(2*optimized_params['cov'][1])
        sn2 = np.exp(2*optimized_params['lik'])
        
        g_t_patient = g_t[ID]
        m_s_patient, s_s_patient = predictGP(x_s, y_s, xtest, gpml_model)
        try: 
            m_a_patient, s_a_patient = call_adapt(x_a_patient, y_a_patient, x_s, y_s, xtest, gpml_m_s, gpml_s_s, ls, mul, var, sn2)
        except:
            m_a_patient = np.full((g_t_patient.shape[0], y_a_patient.shape[1]), 0)
            s_a_patient = np.full((g_t_patient.shape[0], y_a_patient.shape[1]), 0)
        try: 
            m_t_patient, s_t_patient = call_target(x_a_patient, y_a_patient, x_s, y_s, xtest, gpml_m_s, gpml_s_s, ls, mul, var, sn2)
        except:
            m_t_patient = np.full((g_t_patient.shape[0], y_a_patient.shape[1]), 0)
            s_t_patient = np.full((g_t_patient.shape[0], y_a_patient.shape[1]), 0)
        
        x_rows, x_cols = g_t_patient.shape
        g_t_patient = g_t[ID]
        m_s_patient = np.reshape(m_s_patient, (x_rows, y_a_patient.shape[1]))
        m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
        m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))

        #compute mean absolute error
        e_s = mean_absolute_error(m_s_patient, g_t_patient)
        e_a = mean_absolute_error(m_a_patient, g_t_patient)
        e_t = mean_absolute_error(m_t_patient, g_t_patient)

        error_all[ID] = [e_s, e_a, e_t]

        #write ground truth and mu values to csv
        with open(M0_GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g_t_patient)):
                w.writerow([ID, g_t_patient[i], m_s_patient[i], m_a_patient[i], m_t_patient[i]])
    
    #write error values to csv
    with open(M0_ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in error_all.items():
            value = list(value)
            w.writerow([key, value[0][0], value[1][0], value[2][0]])
        
#write average error values to csv 
column_sums = None 

with open (M0_ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (M0_ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)