## Hard Coded Model 
# Note: Must update model 0, 2, 3 (see independent scripts)

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
        m_a_patient, s_a_patient = call_adapt(x_a_patient, y_a_patient, x_s, y_s, xtest, gpml_m_s, gpml_s_s, ls, mul, var, sn2)
        m_t_patient, s_t_patient = call_target(x_a_patient, y_a_patient, x_s, y_s, xtest, gpml_m_s, gpml_s_s, ls, mul, var, sn2)
        
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

    
    
    
'''
########################################################################################
## MODEL 1 - KGP Test: Tests KGP layer by initializing DNN with unit/identity weights ## 
########################################################################################

''''''
Compares results from GPML model to KGP model. 
- GPML Model: Uses MATLAB GPML package in backend to train and predict on GP model. 
- KGP Model: DNN + KGP model. DNN initialized with unit weights and zero bias (trainable=False). KGP initialized with optimized hyperparameters from GPML model. 
''''''

print('----- MODEL 1 - KGP Test: Tests KGP Layer -----')
    
## TEST RESULTS: PASSED 
# GPML ERROR: [0.09684007]
# KGP ERROR: [0.09684007]

# Create folder 
M1_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'model_1')
pathlib.Path(M1_FOLDER_DIR).mkdir(parents=True, exist_ok=True)

# Create CSV files 
M1_GT_MEAN_DIR = os.path.join(M1_FOLDER_DIR, 'gt_mean.csv')
M1_ERROR_DIR = os.path.join(M1_FOLDER_DIR, 'mse.csv')

print('----- LOADING DATA -----')

# Load truncated data 
TRUNC_CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_data_all_norm_MMSE_trunc.csv')
trunc_data_all = load_data(TRUNC_CSV_DATA_DIR)

all_error = []

# Loop for 10 folds 
for i in range(0, 1):
#for i in range(0, 10): #0 to 9 
    
    # Extract necessary data from data_all 
    x_s = trunc_data_all[:620, 3:-2]
    y_s = trunc_data_all[:620, -2:-1]
    xtest_all = trunc_data_all[620:, 3:-2]
    ytest_all = trunc_data_all[620:, -2:-1]
    value_all = trunc_data_all[620:, -1:]
            
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
    
    # Build deep GP model 
    
    for a in range(1): 
            
        gpml_model, optimized_params = trainGP(x_s, y_s, dimensions, initial_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn) 
        gpml_m_s, gpml_s_s = predictGP(x_s, y_s, xtest_all, gpml_model)
        
        ######################
        ##### FIT MODEL ######
        ######################
        
        print('----- FITTING MODEL -----')
        
        #prepare GP layer with optimized parameters
        gp_layer = GP(
                        hyp = optimized_params,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 620,
                        nb_train_samples = x_s.shape[0])
        
        #build model with GP layer
        
        inputs_dgp = Input(shape = (x_s.shape[1], ))
#        inputs_dgp = Input(shape = x_s.shape)
        deep_layer_dgp = Dense(x_s.shape[1], activation = 'linear', kernel_initializer = initializers.Identity(gain=1.0), bias_initializer = 'zeros', trainable = False)(inputs_dgp)
#        deep_layer_dgp = Dense(x_s.shape[0], activation = 'relu', kernel_initializer = initializers.Identity(gain=1.0), bias_initializer = 'zeros')(inputs_dgp)
        outputs_dgp = gp_layer(deep_layer_dgp)
        
        deep_model = kgpModel(inputs = inputs_dgp, outputs = outputs_dgp)
        
        #compile model
        deep_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        #fit model
        deep_model.fit(x_s, y_s,
                       validation_data = (xtest_all, ytest_all), 
                       batch_size = 620,
#                       epochs = 10,
#                       callbacks = cb,
                       verbose = 1)

        #predict and compute error
        y_predict, var_predict = deep_model.predict(xtest_all, x_s, y_s, return_var = True, verbose = 0)
        y_predict = y_predict[0]
        error = mean_absolute_error(y_predict, ytest_all)
        all_error.append(error)

# Print errors 
print('GPML ERROR:', mean_absolute_error(gpml_m_s, ytest_all))
print('KGP ERROR:', error)
'''




###########################################################
## MODEL 2 - DNN + KGP: Trains and Tests DNN + KGP Model ## 
###########################################################

'''
Trains and tests DNN + KGP model. 
- DNN + KGP Model: Trains DNN + FCC (linear) model. DNN weights duplicated. KGP initialized with optimized hyperparameters from GPML model. Trains and predicts on DNN + KGP model. 
'''

print('----- MODEL 2 - DNN + KGP: Trains and Tests DNN + KGP Model -----')

# Create folder 
M2_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'model_2')
pathlib.Path(M2_FOLDER_DIR).mkdir(parents=True, exist_ok=True)

# Create CSV files 
M2_GT_MEAN_DIR = os.path.join(M2_FOLDER_DIR, 'gt_mean.csv')
M2_ERROR_DIR = os.path.join(M2_FOLDER_DIR, 'mse.csv')
M2_HYP_DIR = os.path.join(M2_FOLDER_DIR, 'hyp.csv')
#M2_WEIGHTS_DIR = os.path.join(M2_FOLDER_DIR, 'dnn_weights.h5')
#M2_ACTIV_DIR = os.path.join(M2_FOLDER_DIR, 'activ.csv')


all_train_error = {}
all_test_error = {}

# Loop for 10 folds, plot error 
for i in range(10): #0 to 9
    
    fold_num = i+1
    
    print('FOLD:', fold_num)
    
    # Extract necessary data from data_all 
    x_dict_s, y_dict_s, x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source = extract_data_10fold(i, ID_all, data_all)
    
    ######################
    #### BASE MODEL ######
    ######################
    
    print('----- BUILDING BASE MODEL -----')
    
    base_weights, z, base_model = call_base_model(x_s, y_s, xtest_all, ytest_all)
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    #TODO: start: call_gpml 
    # Calculate initial parameters
    max_x_s = np.amax(z, axis = 0)
    min_x_s = np.amin(z, axis = 0)
    initial_lik = np.log(np.sqrt(0.1*np.var(y_s)))
    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(np.var(y_s)))]])
    print('Initial Parameters:', initial_lik, initial_cov)
    
    # Input features for GP
    dimensions = z.shape[1]
    initial_hyp = {'lik': initial_lik, 'mean': [], 'cov': initial_cov}
    train_opt = {}
    inf_method = 'infExact'
    mean_fxn = 'meanZero'
    cov_fxn = 'covSEiso'
    lik_fxn = 'likGauss'
    dlik_fxn = 'dlikExact'
    
    # Build and train deep GP model 
    sgp_fold_train_error = []
    sgp_fold_test_error = []
    pgp_fold_train_error = [] # added 
    pgp_fold_test_error = [] # added 
    tgp_fold_train_error = [] # added 
    tgp_fold_test_error = [] # added 
        
    for b in range(25):
#    for b in range(100):
        print('LOOP COUNT:', b+1)
        
        if b == 0:
            z_modified = z
            hyp = initial_hyp 
            dnn_weights = base_weights[:2] # Get weights corresponding to DNN layer 
        
        # Save hyperparameters 
        with open(M2_HYP_DIR, 'a') as myfile:
            myfile.write(str(hyp).replace('\n',''))
            myfile.write('\n')

        # OPTIMIZE DNN LAYER ONLY 
        #prepare GP layer with optimized parameters
        gp_layer1 = GP(
                        hyp = hyp,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = z.shape[0])
        
        inputs_dgp1 = Input(shape = (x_s.shape[1], ))
        deep_layer_dgp1 = Dense(128, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1)
        outputs_dgp1 = gp_layer1(deep_layer_dgp1)
        
        deep_model1 = kgpModel(inputs = inputs_dgp1, outputs = outputs_dgp1)
        
        #compile model 
        deep_model1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        #fit model
        deep_model1.fit(x_s, y_s, 
                        validation_data = (xtest_all, ytest_all), 
                        batch_size = 100, 
                        epochs = 10, 
                        callbacks = cb,
                        verbose = 1)
        
        # Get new dnn_weights and new z_modified 
        x_s_activations = get_activations(deep_model1, x_s)
        z_modified = x_s_activations[-2] 
        dnn_weights = deep_model1.get_weights()
        
        print('Deep Model 1:', deep_model1.summary())

        # OPTIMIZE KGP LAYER ONLY 
        gpml_model, optimized_params = trainGP(z_modified, y_s, dimensions, hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn)
        
        ######################
        ##### FIT MODEL ######
        ######################
        
        print('----- FITTING MODEL -----')
        
        #prepare GP layer with optimized parameters
        gp_layer2 = GP(
                        hyp = optimized_params,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = z.shape[0])
        
        inputs_dgp2 = Input(shape = (x_s.shape[1], ))
        deep_layer_dgp2 = Dense(128, activation = 'relu', weights = dnn_weights, trainable = False)(inputs_dgp2)
        outputs_dgp2 = gp_layer2(deep_layer_dgp2)
        
        deep_model2 = kgpModel(inputs = inputs_dgp2, outputs = outputs_dgp2)
        
#        #set appropriate weights to deep model 
#        deep_layer_dgp.set_weights(dnn_weights)
        
        #compile model
        deep_model2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        #fit model
        deep_model2.fit(x_s, y_s,
                       validation_data = (xtest_all, ytest_all), 
                       batch_size = 100,
                       epochs = 10,
                       callbacks = cb,
                       verbose = 1)
        
        print('Deep Model 2:', deep_model2.summary())
        
        #predict and compute error
        sgp_train_m, sgp_train_s = deep_model2.predict(x_s, x_s, y_s, return_var = True, verbose = 0)
        sgp_test_m, sgp_test_s = deep_model2.predict(xtest_all, x_s, y_s, return_var = True, verbose = 0)
        
        sgp_train_error = mean_absolute_error(sgp_train_m[0], y_s)
        sgp_test_error = mean_absolute_error(sgp_test_m[0], ytest_all)
        
        sgp_fold_train_error.append(sgp_train_error)
        sgp_fold_test_error.append(sgp_test_error)
        
        # added 
        #predict pgp results and compute error 
        all_patient_pgp_fold_train_error = []
        all_patient_pgp_fold_test_error = []
        all_patient_tgp_fold_train_error = []
        all_patient_tgp_fold_test_error = []
        
        for ID in tr_ind_source: 
            
            print('----- TRAINING PATIENT: %s -----'%(ID))
            
            x_a_patient = x_dict_s[ID][:-1,:]
            y_a_patient = y_dict_s[ID][:-1,:]
#            xtest = x_a[ID]
            
            predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, x_a_patient)
            
            g_t_patient = g_t[ID]
            m_a_patient = predictions['adapted model mu']
            m_t_patient = predictions['target model mu']
    
            x_rows, x_cols = g_t_patient.shape 
            m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
            m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
            
            #compute mean absolute error
            e_a = mean_absolute_error(m_a_patient, g_t_patient)
            e_t = mean_absolute_error(m_t_patient, g_t_patient)
    
            #append to list 
            all_patient_pgp_fold_train_error.append(e_a)
            all_patient_tgp_fold_train_error.append(e_t)
        
        for ID in tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            x_a_patient = x_a[ID][:-1,:]
            y_a_patient = y_a[ID][:-1,:]
            xtest = x_a[ID]
            
            predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, xtest)
            
            g_t_patient = g_t[ID]
            m_a_patient = predictions['adapted model mu']
            m_t_patient = predictions['target model mu']
    
            x_rows, x_cols = g_t_patient.shape 
            m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
            m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
            
            #compute mean absolute error
            e_a = mean_absolute_error(m_a_patient, g_t_patient)
            e_t = mean_absolute_error(m_t_patient, g_t_patient)
    
            #append to list 
            all_patient_pgp_fold_test_error.append(e_a)
            all_patient_tgp_fold_test_error.append(e_t)
            
        pgp_fold_train_error.append(sum(all_patient_pgp_fold_train_error)/len(all_patient_pgp_fold_train_error))
        pgp_fold_test_error.append(sum(all_patient_pgp_fold_test_error)/len(all_patient_pgp_fold_test_error))
        tgp_fold_train_error.append(sum(all_patient_tgp_fold_train_error)/len(all_patient_tgp_fold_train_error))
        tgp_fold_test_error.append(sum(all_patient_tgp_fold_test_error)/len(all_patient_tgp_fold_test_error))
        
        # added 
        
        #get z for next iteration 
        x_s_activations = get_activations(deep_model2, x_s)
        z_modified = x_s_activations[-2] 
        
        #get hyperparameters for next iteration 
        hyp = optimized_params
        print('parameters for next iteration', hyp)
        
        #get weights for next iteration 
        dnn_weights = deep_model2.get_weights()
        
        #save weights 
        M2_WEIGHTS_DIR = os.path.join(M2_FOLDER_DIR, 'dnn_weights_f%s_loop%s.h5'%(fold_num, b+1))
        deep_model2.save_weights(M2_WEIGHTS_DIR)
        
        #clear session 
        K.clear_session()
    
    # Save and add errors to dictionary, indexed by fold number 
    all_train_error[fold_num] = sgp_fold_train_error
    all_test_error[fold_num] = sgp_fold_test_error
    
    SGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_sgp_train_error.csv'%(fold_num))
    SGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_sgp_test_error.csv'%(fold_num))
    np.savetxt(SGP_FOLD_TRAIN_ERROR_CSV_DIR, sgp_fold_train_error, delimiter=",")
    np.savetxt(SGP_FOLD_TEST_ERROR_CSV_DIR, sgp_fold_test_error, delimiter=",")
    
    # added 
    PGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_pgp_train_error.csv'%(fold_num))
    PGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_pgp_test_error.csv'%(fold_num))
    np.savetxt(PGP_FOLD_TRAIN_ERROR_CSV_DIR, pgp_fold_train_error, delimiter=",")
    np.savetxt(PGP_FOLD_TEST_ERROR_CSV_DIR, pgp_fold_test_error, delimiter=",")    

    TGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_tgp_train_error.csv'%(fold_num))
    TGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_tgp_test_error.csv'%(fold_num))
    np.savetxt(TGP_FOLD_TRAIN_ERROR_CSV_DIR, tgp_fold_train_error, delimiter=",")
    np.savetxt(TGP_FOLD_TEST_ERROR_CSV_DIR, tgp_fold_test_error, delimiter=",")    
    # added 
    
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
        
        #predictions for baseline model 
        m_b_patient = base_model.predict(xtest)
        
        predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, xtest)
        
        g_t_patient = g_t[ID]
        m_s_patient = predictions['source model mu']
        m_a_patient = predictions['adapted model mu']
        m_t_patient = predictions['target model mu']

        x_rows, x_cols = g_t_patient.shape 
        m_s_patient = np.reshape(m_s_patient, (x_rows, y_a_patient.shape[1]))
        m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
        m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
        
        #compute mean absolute error
        e_b = mean_absolute_error(m_b_patient, g_t_patient)
        e_s = mean_absolute_error(m_s_patient, g_t_patient)
        e_a = mean_absolute_error(m_a_patient, g_t_patient)
        e_t = mean_absolute_error(m_t_patient, g_t_patient)

        error_all[ID] = [e_b, e_s, e_a, e_t]

        #write ground truth and mu values to csv
        with open(M2_GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g_t_patient)):
                w.writerow([ID, g_t_patient[i], m_b_patient[i], m_s_patient[i], m_a_patient[i], m_t_patient[i]])
    
    #write error values to csv
    with open(M2_ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in error_all.items():
            value = list(value)
            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])

#write average error values to csv 
column_sums = None 

with open (M2_ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (M2_ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)




###############################################################
## MODEL 3 - DNN + 3 KGP: Trains and Tests DNN + 3 KGP Model ## 
###############################################################

'''
Trains and tests DNN + KGP model. 
- DNN + KGP Model: Trains DNN + FCC (linear) model. DNN weights duplicated. KGP initialized with optimized hyperparameters from GPML model. Trains and predicts on DNN + KGP model. 
'''

print('----- MODEL 3 - DNN + 3 KGP: Trains and Tests DNN + 3 KGP Model -----')

# Create folder 
M3_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'model_3')
pathlib.Path(M3_FOLDER_DIR).mkdir(parents=True, exist_ok=True)

# Create CSV files 
g1_M3_GT_MEAN_DIR = os.path.join(M3_FOLDER_DIR, 'g1_gt_mean.csv')
g1_M3_ERROR_DIR = os.path.join(M3_FOLDER_DIR, 'g1_mse.csv')
g1_M3_HYP_DIR = os.path.join(M3_FOLDER_DIR, 'g1_hyp.csv')
g2_M3_GT_MEAN_DIR = os.path.join(M3_FOLDER_DIR, 'g2_gt_mean.csv')
g2_M3_ERROR_DIR = os.path.join(M3_FOLDER_DIR, 'g2_mse.csv')
g2_M3_HYP_DIR = os.path.join(M3_FOLDER_DIR, 'g2_hyp.csv')
g3_M3_GT_MEAN_DIR = os.path.join(M3_FOLDER_DIR, 'g3_gt_mean.csv')
g3_M3_ERROR_DIR = os.path.join(M3_FOLDER_DIR, 'g3_mse.csv')
g3_M3_HYP_DIR = os.path.join(M3_FOLDER_DIR, 'g3_hyp.csv')

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

g1_all_train_error = {}
g1_all_test_error = {}
g2_all_train_error = {}
g2_all_test_error = {}
g3_all_train_error = {}
g3_all_test_error = {}

# Loop for 4 folds, plot error 
for i in range(4): #0 to 3
    
    fold_num = i+1

    print('FOLD:', fold_num)
    
    # Extract necessary data from data_all 
    g1_fold_tst_ind = list(classif_data[:, 0][g1_fold_inds[i]])
    g2_fold_tst_ind = list(classif_data[:, 0][g2_fold_inds[i]])
    g3_fold_tst_ind = list(classif_data[:, 0][g3_fold_inds[i]])
    
    g1_fold_tr_ind = np.setdiff1d(ID_all, g1_fold_tst_ind)
    g2_fold_tr_ind = np.setdiff1d(ID_all, g2_fold_tst_ind)
    g3_fold_tr_ind = np.setdiff1d(ID_all, g3_fold_tst_ind)

    g1_data = extract_data_4fold(g1_fold_tst_ind, g1_fold_tr_ind, ID_all, data_all)
    g2_data = extract_data_4fold(g2_fold_tst_ind, g2_fold_tr_ind, ID_all, data_all)
    g3_data = extract_data_4fold(g3_fold_tst_ind, g3_fold_tr_ind, ID_all, data_all)
    
    group_data = (g1_data, g2_data, g3_data)
    
#    g1_x_s, g1_y_s, g1_xtest_all, g1_ytest_all, g1_x_a, g1_y_a, g1_g_t, g1_g_t_all = extract_data_4fold(g1_fold_tst_ind, g1_fold_tr_ind, ID_all, data_all)
#    g2_x_s, g2_y_s, g2_xtest_all, g2_ytest_all, g2_x_a, g2_y_a, g2_g_t, g2_g_t_all = extract_data_4fold(g1_fold_tst_ind, g1_fold_tr_ind, ID_all, data_all)
#    g3_x_s, g3_y_s, g3_xtest_all, g3_ytest_all, g3_x_a, g3_y_a, g3_g_t, g3_g_t_all = extract_data_4fold(g1_fold_tst_ind, g1_fold_tr_ind, ID_all, data_all)
    
    ######################
    #### BASE MODEL ######
    ######################
    
    # Randomly choose order to train base model (note: shuffle(x) mutates list)
    x = [0, 1, 2]
    shuffle(x)
    # x = [2, 1, 0]
    
    x_s1, y_s1, xtest_all1, ytest_all1, x_a1, y_a1, g_t1, g_t_all1 = group_data[x[0]]
    x_s2, y_s2, xtest_all2, ytest_all2, x_a2, y_a2, g_t2, g_t_all2 = group_data[x[1]]
    x_s3, y_s3, xtest_all3, ytest_all3, x_a3, y_a3, g_t3, g_t_all3 = group_data[x[2]]
    
    print('----- BUILDING BASE MODEL -----')

    base_model_data = call_base_model_m3(x_s1, y_s1, xtest_all1, ytest_all1, x_s2, y_s2, xtest_all2, ytest_all2, x_s3, y_s3, xtest_all3, ytest_all3)    
#    base_weights, z1, z2, z3, base_model = call_base_model_m3(x_s1, y_s1, xtest_all1, ytest_all1, x_s2, y_s2, xtest_all2, ytest_all2, x_s3, y_s3, xtest_all3, ytest_all3)
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    #TODO: start: call_gpml 
    # Calculate initial parameters 
    g1_z = base_model_data[1 + x.index(0)]
    g2_z = base_model_data[1 + x.index(1)]
    g3_z = base_model_data[1 + x.index(2)]
    g1_x_s, g1_y_s, g1_xtest_all, g1_ytest_all, g1_x_a, g1_y_a, g1_g_t, g1_g_t_all = group_data[0]
    g2_x_s, g2_y_s, g2_xtest_all, g2_ytest_all, g2_x_a, g2_y_a, g2_g_t, g2_g_t_all = group_data[1]
    g3_x_s, g3_y_s, g3_xtest_all, g3_ytest_all, g3_x_a, g3_y_a, g3_g_t, g3_g_t_all = group_data[2]
    # For group 1: 
    g1_max_x_s = np.amax(g1_z, axis = 0)
    g1_min_x_s = np.amin(g1_z, axis = 0)
    g1_initial_lik = np.log(np.sqrt(0.1*np.var(g1_y_s)))
    g1_initial_cov = np.array([[np.log(np.median(g1_max_x_s - g1_min_x_s))], [np.log(np.sqrt(np.var(g1_y_s)))]])
    print('Initial Parameters:', g1_initial_lik, g1_initial_cov)
    # For group 2: 
    g2_max_x_s = np.amax(g2_z, axis = 0)
    g2_min_x_s = np.amin(g2_z, axis = 0)
    g2_initial_lik = np.log(np.sqrt(0.1*np.var(g2_y_s)))
    g2_initial_cov = np.array([[np.log(np.median(g2_max_x_s - g2_min_x_s))], [np.log(np.sqrt(np.var(g2_y_s)))]])
    print('Initial Parameters:', g2_initial_lik, g2_initial_cov)
    # For group 3: 
    g3_max_x_s = np.amax(g3_z, axis = 0)
    g3_min_x_s = np.amin(g3_z, axis = 0)
    g3_initial_lik = np.log(np.sqrt(0.1*np.var(g3_y_s)))
    g3_initial_cov = np.array([[np.log(np.median(g3_max_x_s - g3_min_x_s))], [np.log(np.sqrt(np.var(g3_y_s)))]])
    print('Initial Parameters:', g3_initial_lik, g3_initial_cov)
    
    # Input features for GP
    train_opt = {}
    inf_method = 'infExact'
    mean_fxn = 'meanZero'
    cov_fxn = 'covSEiso'
    lik_fxn = 'likGauss'
    dlik_fxn = 'dlikExact'  
    # For group 1: 
    g1_dimensions = g1_z.shape[1]
    g1_initial_hyp = {'lik': g1_initial_lik, 'mean': [], 'cov': g1_initial_cov}  
    # For group 2: 
    g2_dimensions = g2_z.shape[1]
    g2_initial_hyp = {'lik': g2_initial_lik, 'mean': [], 'cov': g2_initial_cov}   
    # For group 3: 
    g3_dimensions = g3_z.shape[1]
    g3_initial_hyp = {'lik': g3_initial_lik, 'mean': [], 'cov': g3_initial_cov}
    
    # Build and train deep GP model 
    g1_sgp_fold_train_error = []
    g1_sgp_fold_test_error = []
    g2_sgp_fold_train_error = []
    g2_sgp_fold_test_error = []
    g3_sgp_fold_train_error = []
    g3_sgp_fold_test_error = []
    
    base_weights, z1, z2, z3, base_model = base_model_data
    
    for b in range(100):
        print('LOOP COUNT:', b+1)
        
        if b == 0:
            g1_z_modified = g1_z
            g1_hyp = g1_initial_hyp 
            g2_z_modified = g2_z
            g2_hyp = g2_initial_hyp 
            g3_z_modified = g3_z
            g3_hyp = g3_initial_hyp 
            # Get weights corresponding to DNN layer 
            g1_dnn_weights = base_weights[:2] 
            g2_dnn_weights = base_weights[:2] 
            g3_dnn_weights = base_weights[:2] 
            
        # Save hyperparameters 
        with open(g1_M3_HYP_DIR, 'a') as myfile:
            myfile.write(str(g1_hyp).replace('\n',''))
            myfile.write('\n')

        with open(g2_M3_HYP_DIR, 'a') as myfile:
            myfile.write(str(g2_hyp).replace('\n',''))
            myfile.write('\n')
            
        with open(g3_M3_HYP_DIR, 'a') as myfile:
            myfile.write(str(g3_hyp).replace('\n',''))
            myfile.write('\n')
        
        # OPTIMIZE DNN LAYER ONLY 
        #prepare GP layer with optimized parameters
        # For group 1: 
        g1_gp_layer1 = GP(
                        hyp = g1_hyp,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g1_z.shape[0])
        
        g1_inputs_dgp1 = Input(shape = (g1_x_s.shape[1], ))
        g1_deep_layer_dgp1 = Dense(128, activation = 'relu', weights = g1_dnn_weights, trainable = True)(g1_inputs_dgp1)
        g1_outputs_dgp1 = g1_gp_layer1(g1_deep_layer_dgp1)
        
        g1_deep_model1 = kgpModel(inputs = g1_inputs_dgp1, outputs = g1_outputs_dgp1)        
        # For group 2: 
        g2_gp_layer1 = GP(
                        hyp = g2_hyp,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g2_z.shape[0])
        
        g2_inputs_dgp1 = Input(shape = (g2_x_s.shape[1], ))
        g2_deep_layer_dgp1 = Dense(128, activation = 'relu', weights = g2_dnn_weights, trainable = True)(g2_inputs_dgp1)
        g2_outputs_dgp1 = g2_gp_layer1(g2_deep_layer_dgp1)
        
        g2_deep_model1 = kgpModel(inputs = g2_inputs_dgp1, outputs = g2_outputs_dgp1)        
        # For group 3: 
        g3_gp_layer1 = GP(
                        hyp = g3_hyp,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g3_z.shape[0])
        
        g3_inputs_dgp1 = Input(shape = (g3_x_s.shape[1], ))
        g3_deep_layer_dgp1 = Dense(128, activation = 'relu', weights = g3_dnn_weights, trainable = True)(g3_inputs_dgp1)
        g3_outputs_dgp1 = g3_gp_layer1(g3_deep_layer_dgp1)
        
        g3_deep_model1 = kgpModel(inputs = g3_inputs_dgp1, outputs = g3_outputs_dgp1)
        
        #compile model 
        g1_deep_model1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        g2_deep_model1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        g3_deep_model1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
                
        #fit model
        g1_deep_model1.fit(g1_x_s, g1_y_s, 
                        validation_data = (g1_xtest_all, g1_ytest_all), 
                        batch_size = 100, 
                        epochs = 10, 
                        callbacks = cb,
                        verbose = 1)
        g2_deep_model1.fit(g2_x_s, g2_y_s, 
                        validation_data = (g2_xtest_all, g2_ytest_all), 
                        batch_size = 100, 
                        epochs = 10, 
                        callbacks = cb,
                        verbose = 1)
        g3_deep_model1.fit(g3_x_s, g3_y_s, 
                        validation_data = (g3_xtest_all, g3_ytest_all), 
                        batch_size = 100, 
                        epochs = 10, 
                        callbacks = cb,
                        verbose = 1)
        
        # Get new dnn_weights and new z_modified 
        g1_x_s_activations = get_activations(g1_deep_model1, g1_x_s)
        g1_z_modified = g1_x_s_activations[-2] 
        g1_dnn_weights = g1_deep_model1.get_weights()

        g2_x_s_activations = get_activations(g2_deep_model1, g2_x_s)
        g2_z_modified = g2_x_s_activations[-2] 
        g2_dnn_weights = g2_deep_model1.get_weights()

        g3_x_s_activations = get_activations(g3_deep_model1, g3_x_s)
        g3_z_modified = g3_x_s_activations[-2] 
        g3_dnn_weights = g3_deep_model1.get_weights()        
        
        print('Group 1 - Deep Model 1:', g1_deep_model1.summary())
        print('Group 2 - Deep Model 1:', g2_deep_model1.summary())
        print('Group 3 - Deep Model 1:', g3_deep_model1.summary())
        
        # OPTIMIZE KGP LAYER ONLY 
        g1_gpml_model, g1_optimized_params = trainGP(g1_z_modified, g1_y_s, g1_dimensions, g1_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn)
        g2_gpml_model, g2_optimized_params = trainGP(g2_z_modified, g2_y_s, g2_dimensions, g2_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn)
        g3_gpml_model, g3_optimized_params = trainGP(g3_z_modified, g3_y_s, g3_dimensions, g3_hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn)
        
        ######################
        ##### FIT MODEL ######
        ######################
        
        print('----- FITTING MODEL -----')
        
        #prepare GP layer with optimized parameters
        # For group 1: 
        g1_gp_layer2 = GP(
                        hyp = g1_optimized_params,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g1_z.shape[0])
        g1_inputs_dgp2 = Input(shape = (g1_x_s.shape[1], ))
        g1_deep_layer_dgp2 = Dense(128, activation = 'relu', weights = g1_dnn_weights, trainable = False)(g1_inputs_dgp2)
        g1_outputs_dgp2 = g1_gp_layer2(g1_deep_layer_dgp2)
        
        g1_deep_model2 = kgpModel(inputs = g1_inputs_dgp2, outputs = g1_outputs_dgp2)
        # For group 2: 
        g2_gp_layer2 = GP(
                        hyp = g2_optimized_params,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g2_z.shape[0])
        g2_inputs_dgp2 = Input(shape = (g2_x_s.shape[1], ))
        g2_deep_layer_dgp2 = Dense(128, activation = 'relu', weights = g2_dnn_weights, trainable = False)(g2_inputs_dgp2)
        g2_outputs_dgp2 = g2_gp_layer2(g2_deep_layer_dgp2)
        
        g2_deep_model2 = kgpModel(inputs = g2_inputs_dgp2, outputs = g2_outputs_dgp2)        
        # For group 3: 
        g3_gp_layer2 = GP(
                        hyp = g3_optimized_params,
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = g3_z.shape[0])
        g3_inputs_dgp2 = Input(shape = (g3_x_s.shape[1], ))
        g3_deep_layer_dgp2 = Dense(128, activation = 'relu', weights = g3_dnn_weights, trainable = False)(g3_inputs_dgp2)
        g3_outputs_dgp2 = g3_gp_layer2(g3_deep_layer_dgp2)
        
        g3_deep_model2 = kgpModel(inputs = g3_inputs_dgp2, outputs = g3_outputs_dgp2)
        
        #compile model
        g1_deep_model2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        g2_deep_model2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        g3_deep_model2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        #fit model
        g1_deep_model2.fit(g1_x_s, g1_y_s,
                       validation_data = (g1_xtest_all, g1_ytest_all), 
                       batch_size = 100,
                       epochs = 10,
                       callbacks = cb,
                       verbose = 1)
        g2_deep_model2.fit(g2_x_s, g2_y_s,
                       validation_data = (g2_xtest_all, g2_ytest_all), 
                       batch_size = 100,
                       epochs = 10,
                       callbacks = cb,
                       verbose = 1)
        g3_deep_model2.fit(g3_x_s, g3_y_s,
                       validation_data = (g3_xtest_all, g3_ytest_all), 
                       batch_size = 100,
                       epochs = 10,
                       callbacks = cb,
                       verbose = 1)
        
        print('Group 1 - Deep Model 1:', g1_deep_model2.summary())
        print('Group 2 - Deep Model 1:', g2_deep_model2.summary())
        print('Group 3 - Deep Model 1:', g3_deep_model2.summary())
        
        #predict and compute error
        g1_sgp_train_m, g1_sgp_train_s = g1_deep_model2.predict(g1_x_s, g1_x_s, g1_y_s, return_var = True, verbose = 0)
        g1_sgp_test_m, g1_sgp_test_s = g1_deep_model2.predict(g1_xtest_all, g1_x_s, g1_y_s, return_var = True, verbose = 0)
        g2_sgp_train_m, g2_sgp_train_s = g2_deep_model2.predict(g2_x_s, g2_x_s, g2_y_s, return_var = True, verbose = 0)
        g2_sgp_test_m, g2_sgp_test_s = g2_deep_model2.predict(g2_xtest_all, g2_x_s, g2_y_s, return_var = True, verbose = 0)
        g3_sgp_train_m, g3_sgp_train_s = g3_deep_model2.predict(g3_x_s, g3_x_s, g3_y_s, return_var = True, verbose = 0)
        g3_sgp_test_m, g3_sgp_test_s = g3_deep_model2.predict(g3_xtest_all, g3_x_s, g3_y_s, return_var = True, verbose = 0)

        g1_sgp_train_error = mean_absolute_error(g1_sgp_train_m[0], g1_y_s)
        g1_sgp_test_error = mean_absolute_error(g1_sgp_test_m[0], g1_ytest_all)
        g2_sgp_train_error = mean_absolute_error(g2_sgp_train_m[0], g2_y_s)
        g2_sgp_test_error = mean_absolute_error(g2_sgp_test_m[0], g2_ytest_all)
        g3_sgp_train_error = mean_absolute_error(g3_sgp_train_m[0], g3_y_s)
        g3_sgp_test_error = mean_absolute_error(g3_sgp_test_m[0], g3_ytest_all)

        g1_sgp_fold_train_error.append(g1_sgp_train_error)
        g1_sgp_fold_test_error.append(g1_sgp_test_error)
        g2_sgp_fold_train_error.append(g2_sgp_train_error)
        g2_sgp_fold_test_error.append(g2_sgp_test_error)
        g3_sgp_fold_train_error.append(g3_sgp_train_error)
        g3_sgp_fold_test_error.append(g3_sgp_test_error)
        
        #get z for next iteration 
        g1_x_s_activations = get_activations(g1_deep_model2, g1_x_s)
        g1_z_modified = g1_x_s_activations[-2] 
        g2_x_s_activations = get_activations(g2_deep_model2, g2_x_s)
        g2_z_modified = g2_x_s_activations[-2] 
        g3_x_s_activations = get_activations(g3_deep_model2, g3_x_s)
        g3_z_modified = g3_x_s_activations[-2] 
        
        #get hyperparameters for next iteration 
        g1_hyp = g1_optimized_params
        print('parameters for next iteration', g1_hyp)
        g2_hyp = g2_optimized_params
        print('parameters for next iteration', g2_hyp)
        g3_hyp = g3_optimized_params
        print('parameters for next iteration', g3_hyp)
        
        #get weights for next iteration 
        g1_dnn_weights = g1_deep_model2.get_weights()
        g2_dnn_weights = g2_deep_model2.get_weights()
        g3_dnn_weights = g3_deep_model2.get_weights()
        
        #save weights 
        g1_M3_WEIGHTS_DIR = os.path.join(M3_FOLDER_DIR, 'g1_dnn_weights_f%s_loop%s.h5'%(fold_num, b+1))
        g1_deep_model2.save_weights(g1_M3_WEIGHTS_DIR)
        g2_M3_WEIGHTS_DIR = os.path.join(M3_FOLDER_DIR, 'g2_dnn_weights_f%s_loop%s.h5'%(fold_num, b+1))
        g2_deep_model2.save_weights(g2_M3_WEIGHTS_DIR)
        g3_M3_WEIGHTS_DIR = os.path.join(M3_FOLDER_DIR, 'g3_dnn_weights_f%s_loop%s.h5'%(fold_num, b+1))
        g3_deep_model2.save_weights(g3_M3_WEIGHTS_DIR)
        
        #clear session 
        K.clear_session()        
    
    # Save and add errors to dictionary, indexed by fold number 
    g1_all_train_error[fold_num] = g1_sgp_fold_train_error
    g1_all_test_error[fold_num] = g1_sgp_fold_test_error
    g2_all_train_error[fold_num] = g2_sgp_fold_train_error
    g2_all_test_error[fold_num] = g2_sgp_fold_test_error
    g3_all_train_error[fold_num] = g3_sgp_fold_train_error
    g3_all_test_error[fold_num] = g3_sgp_fold_test_error
    
    g1_SGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g1_fold_%s_sgp_train_error.csv'%(fold_num))
    g1_SGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g1_fold_%s_sgp_test_error.csv'%(fold_num))
    np.savetxt(g1_SGP_FOLD_TRAIN_ERROR_CSV_DIR, g1_sgp_fold_train_error, delimiter=",")
    np.savetxt(g1_SGP_FOLD_TEST_ERROR_CSV_DIR, g1_sgp_fold_test_error, delimiter=",")
    
    g2_SGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g2_fold_%s_sgp_train_error.csv'%(fold_num))
    g2_SGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g2_fold_%s_sgp_test_error.csv'%(fold_num))
    np.savetxt(g2_SGP_FOLD_TRAIN_ERROR_CSV_DIR, g2_sgp_fold_train_error, delimiter=",")
    np.savetxt(g2_SGP_FOLD_TEST_ERROR_CSV_DIR, g2_sgp_fold_test_error, delimiter=",")    
    
    g3_SGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g3_fold_%s_sgp_train_error.csv'%(fold_num))
    g3_SGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M3_FOLDER_DIR, 'g3_fold_%s_sgp_test_error.csv'%(fold_num))
    np.savetxt(g3_SGP_FOLD_TRAIN_ERROR_CSV_DIR, g3_sgp_fold_train_error, delimiter=",")
    np.savetxt(g3_SGP_FOLD_TEST_ERROR_CSV_DIR, g3_sgp_fold_test_error, delimiter=",")
    
    #TRAIN ADAPTATION AND TARGET MODELS
    
    print('----- TRAINING ADAPTATION AND TARGET MODELS -----')
    
    g1_error_all = {}
    g2_error_all = {}
    g3_error_all = {}
    
    #make folders 
#    GT_MEAN_FOLD_FOLDER = os.path.join(GT_MEAN_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(GT_MEAN_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
    
#    GT_MEAN_EXTRACTED_FOLD_FOLDER = os.path.join(GT_MEAN_EXTRACTED_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(GT_MEAN_EXTRACTED_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
    
#    ERROR_FOLD_FOLDER_DIR = os.path.join(ERROR_FOLDER_DIR, fold[i]) #name of folder  
#    pathlib.Path(ERROR_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #make folder 
    
    #output mean and variance predictions for source, adaptation, and target models 
    #iterate over test patients (group 1)
    for ID in g1_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g1_x_a_patient = g1_x_a[ID][:-1,:]
        g1_y_a_patient = g1_y_a[ID][:-1,:]
        g1_xtest = g1_x_a[ID]
        
        #predictions for baseline model 
        g1_m_b_patient = base_model.predict(g1_xtest)
        
        g1_predictions = call_deep_pgp(g1_deep_model2, g1_gp_layer2, g1_x_a_patient, g1_y_a_patient, g1_x_s, g1_y_s, g1_xtest)
        
        g1_g_t_patient = g1_g_t[ID]
        g1_m_s_patient = g1_predictions['source model mu']
        g1_m_a_patient = g1_predictions['adapted model mu']
        g1_m_t_patient = g1_predictions['target model mu']

        g1_x_rows, g1_x_cols = g1_g_t_patient.shape 
        g1_m_s_patient = np.reshape(g1_m_s_patient, (g1_x_rows, g1_y_a_patient.shape[1]))
        g1_m_a_patient = np.reshape(g1_m_a_patient, (g1_x_rows, g1_y_a_patient.shape[1]))
        g1_m_t_patient = np.reshape(g1_m_t_patient, (g1_x_rows, g1_y_a_patient.shape[1]))
        
        #compute mean absolute error
        g1_e_b = mean_absolute_error(g1_m_b_patient, g1_g_t_patient)
        g1_e_s = mean_absolute_error(g1_m_s_patient, g1_g_t_patient)
        g1_e_a = mean_absolute_error(g1_m_a_patient, g1_g_t_patient)
        g1_e_t = mean_absolute_error(g1_m_t_patient, g1_g_t_patient)

        g1_error_all[ID] = [g1_e_b, g1_e_s, g1_e_a, g1_e_t]

        #write ground truth and mu values to csv
        with open(g1_M3_GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g1_g_t_patient)):
                w.writerow([ID, g1_g_t_patient[i], g1_m_b_patient[i], g1_m_s_patient[i], g1_m_a_patient[i], g1_m_t_patient[i]])
    
    #write error values to csv
    with open(g1_M3_ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in g1_error_all.items():
            value = list(value)
            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])

    for ID in g2_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g2_x_a_patient = g2_x_a[ID][:-1,:]
        g2_y_a_patient = g2_y_a[ID][:-1,:]
        g2_xtest = g2_x_a[ID]
        
        #predictions for baseline model 
        g2_m_b_patient = base_model.predict(g2_xtest)
        
        g2_predictions = call_deep_pgp(g2_deep_model2, g2_gp_layer2, g2_x_a_patient, g2_y_a_patient, g2_x_s, g2_y_s, g2_xtest)
        
        g2_g_t_patient = g2_g_t[ID]
        g2_m_s_patient = g2_predictions['source model mu']
        g2_m_a_patient = g2_predictions['adapted model mu']
        g2_m_t_patient = g2_predictions['target model mu']

        g2_x_rows, g2_x_cols = g2_g_t_patient.shape 
        g2_m_s_patient = np.reshape(g2_m_s_patient, (g2_x_rows, g2_y_a_patient.shape[1]))
        g2_m_a_patient = np.reshape(g2_m_a_patient, (g2_x_rows, g2_y_a_patient.shape[1]))
        g2_m_t_patient = np.reshape(g2_m_t_patient, (g2_x_rows, g2_y_a_patient.shape[1]))
        
        #compute mean absolute error
        g2_e_b = mean_absolute_error(g2_m_b_patient, g2_g_t_patient)
        g2_e_s = mean_absolute_error(g2_m_s_patient, g2_g_t_patient)
        g2_e_a = mean_absolute_error(g2_m_a_patient, g2_g_t_patient)
        g2_e_t = mean_absolute_error(g2_m_t_patient, g2_g_t_patient)

        g2_error_all[ID] = [g2_e_b, g2_e_s, g2_e_a, g2_e_t]

        #write ground truth and mu values to csv
        with open(g2_M3_GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g2_g_t_patient)):
                w.writerow([ID, g2_g_t_patient[i], g2_m_b_patient[i], g2_m_s_patient[i], g2_m_a_patient[i], g2_m_t_patient[i]])
    
    #write error values to csv
    with open(g2_M3_ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in g2_error_all.items():
            value = list(value)
            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])

    for ID in g3_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g3_x_a_patient = g3_x_a[ID][:-1,:]
        g3_y_a_patient = g3_y_a[ID][:-1,:]
        g3_xtest = g3_x_a[ID]
        
        #predictions for baseline model 
        g3_m_b_patient = base_model.predict(g3_xtest)
        
        g3_predictions = call_deep_pgp(g3_deep_model2, g3_gp_layer2, g3_x_a_patient, g3_y_a_patient, g3_x_s, g3_y_s, g3_xtest)
        
        g3_g_t_patient = g3_g_t[ID]
        g3_m_s_patient = g3_predictions['source model mu']
        g3_m_a_patient = g3_predictions['adapted model mu']
        g3_m_t_patient = g3_predictions['target model mu']

        g3_x_rows, g3_x_cols = g3_g_t_patient.shape 
        g3_m_s_patient = np.reshape(g3_m_s_patient, (g3_x_rows, g3_y_a_patient.shape[1]))
        g3_m_a_patient = np.reshape(g3_m_a_patient, (g3_x_rows, g3_y_a_patient.shape[1]))
        g3_m_t_patient = np.reshape(g3_m_t_patient, (g3_x_rows, g3_y_a_patient.shape[1]))
        
        #compute mean absolute error
        g3_e_b = mean_absolute_error(g3_m_b_patient, g3_g_t_patient)
        g3_e_s = mean_absolute_error(g3_m_s_patient, g3_g_t_patient)
        g3_e_a = mean_absolute_error(g3_m_a_patient, g3_g_t_patient)
        g3_e_t = mean_absolute_error(g3_m_t_patient, g3_g_t_patient)

        g3_error_all[ID] = [g3_e_b, g3_e_s, g3_e_a, g3_e_t]

        #write ground truth and mu values to csv
        with open(g3_M3_GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g3_g_t_patient)):
                w.writerow([ID, g3_g_t_patient[i], g3_m_b_patient[i], g3_m_s_patient[i], g3_m_a_patient[i], g3_m_t_patient[i]])
    
    #write error values to csv
    with open(g3_M3_ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in g3_error_all.items():
            value = list(value)
            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])
        
#write average error values to csv 
g1_column_sums = None 

with open (g1_M3_ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (g1_M3_ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)
    
g2_column_sums = None 

with open (g2_M3_ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (g2_M3_ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)

g3_column_sums = None 

with open (g3_M3_ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (g3_M3_ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)





## Loop for 10 folds
##for i in range(0, 1):
#for i in range(0, 10): #0 to 9
#    
#    # Extract necessary data from data_all 
#    x_s, y_s, xtest_all, ytest_all, x_a, y_a, g_t, g_t_all, tst_ind, tr_ind_source = extract_data(i, ID_all, data_all)
#    
#    ######################
#    #### BASE MODEL ######
#    ######################
#    
#    print('----- BUILDING BASE MODEL -----')
#    
#    base_weights, z = call_base_model(x_s, y_s, xtest_all, ytest_all)
#        
#    ######################
#    # OPTIMIZE WITH GPML #
#    ######################
#    
#    print('----- OPTIMIZING WITH GPML -----')
#    
#    #TODO: start: call_gpml 
#    # Calculate initial parameters
#    max_x_s = np.amax(z, axis = 0)
#    min_x_s = np.amin(z, axis = 0)
#    initial_lik = np.log(np.sqrt(0.1*np.var(y_s)))
#    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(np.var(y_s)))]])
#    print('Initial Parameters:', initial_lik, initial_cov)
#
#    # Input features for GP
#    dimensions = z.shape[1]
#    initial_hyp = {'lik': initial_lik, 'mean': [], 'cov': initial_cov}
#    train_opt = {}
#    inf_method = 'infExact'
#    mean_fxn = 'meanZero'
#    cov_fxn = 'covSEiso'
#    lik_fxn = 'likGauss'
#    dlik_fxn = 'dlikExact'
#    
#    # Build deep GP model 
#    # Clone models and weights 
#    #TODO: move building of deep GP model outside of for loop 
#    
#    #train model until error minimizes 
#    loop = 1
#    all_error = []
#    avg_error_diff = 0 
#    
#    #while loop < 4 or avg_error_diff > -0.005: 
#    while loop < 2: #will add error condition later... 
#        
#        print('LOOP COUNT:', loop)
#        
#        if loop == 1:
#            z_modified = z 
#            hyp = initial_hyp 
#            
#        gpml_model, optimized_params = trainGP(z_modified, y_s, dimensions, hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn) 
#        
#        ######################
#        ##### FIT MODEL ######
#        ######################
#        
#        print('----- FITTING MODEL -----')
#        
#        #prepare GP layer with optimized parameters
#        gp_layer = GP(
#                        hyp = optimized_params,
#                        inf = 'infExact',
#                        dlik = 'dlikExact',
#                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
#                        mean = 'meanZero',
#                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
#                        cov = 'covSEiso',
#                        batch_size = 100,
#                        nb_train_samples = z.shape[0])
#        
#        #build model with GP layer
#        #TODO: Allow for variable number of DNNs 
#        #possibly make method with number of DNN input 
#        
#        num_dnn = 3
#        print('Number of DNN Layers:', num_dnn)
#        
#        inputs_dgp = Input(shape = (x_s.shape[1], ))
#        deep_layer_dgp = Dense(128, activation = 'relu')(inputs_dgp)
#        outputs_dgp = gp_layer(deep_layer_dgp)
##        inputs_dgp = Input(shape = (x_s.shape[1], ))
##        deep_layer_dgp = Dense(128, activation = 'relu')(inputs_dgp)
##        DNN1_dgp = Dense(128, activation = 'relu')(deep_layer_dgp)
##        DNN2_dgp = Dense(128, activation = 'relu')(DNN1_dgp)
##        DNN3_dgp = Dense(128, activation = 'relu')(DNN2_dgp)
##        outputs_dgp = gp_layer(DNN3_dgp)
#        
#        deep_model = kgpModel(inputs = inputs_dgp, outputs = outputs_dgp)
#        
#        # set appropriate weights to deep model 
#        if num_dnn < 3: 
#            mod_base_weights = base_weights[:num_dnn*2-6]
#            print(num_dnn*2-6)
#        else:
#            mod_base_weights = base_weights[:-2]
#            for dnn in range(num_dnn-2):
#                mod_base_weights += base_weights[-4:-2]
#        
#        deep_model.set_weights(mod_base_weights)
#        
#        deep_model = kgpModel(inputs = deep_model.layers[0], outputs = deep_model.layers[-1])
#        
#        deep_weights = deep_model.get_weights() 
#        
#        
#        print('Check Model Weights')
#        for weight in range(len(mod_base_weights)):
##            print('mod base weight')
##            print(mod_base_weights[weight][:20])
##            print('deep model weight')
##            print(deep_weights[weight][:20])
#            print('Equal?')
#            print(mod_base_weights[weight] == deep_weights[weight])
#        
#        print(deep_model.summary())
#        
#        # quit()
#        
#        # set weights for deep layer 
#        
#        # do for loop to set weights for DNN layers 
#        # condition: if last layer, set weight to 128x1 weight 
#        
##        print(deep_model.summary())
##        deep_model.set_weights(base_weights)
##        print(deep_model.summary())
##        print(base_weights)
##        print(deep_model.get_weights())
#        #setting dense layer weights to additional DNN layers 
##        deep_model.layers[-2].set_weights(dense_weights)
##        deep_model.layers[-3].set_weights(dense_weights)
#        
##        #set weights to intermediate DNN layer 
##        deep_model.layers[-2].set_weights(dense_weights)
#        
#        #set weights to new deep gp model 
#        # deep_model.layers[-4].set_weights(base_weights[-4])
##        deep_model.layers[-3].set_weights(base_weights[0])
##        deep_model.layers[-2].set_weights(base_weights[-2])
#        
#        #base_weights[0] = weights for input layer 
#        #base_weights[1] = weights for deep layer 
#        #base_weights[2] = weights for dense layer 
#        
#        #compile model
#        deep_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
#        
#        #fit model
#        deep_model.fit(x_s, y_s,
#                       validation_data = (xtest_all, ytest_all), 
#                       batch_size = 100,
#                       epochs = 10,
#                       callbacks = cb,
#                       verbose = 1)
#
#        #predict and compute error
#        y_predict, var_predict = deep_model.predict(xtest_all, x_s, y_s, return_var = True, verbose = 0)
#        y_predict = y_predict[0]
#        error = mean_absolute_error(y_predict, ytest_all)
#        all_error.append(error)
#        
#        #get z for next iteration 
#        x_s_activations = get_activations(deep_model, x_s)
#        z_modified = x_s_activations[-2] 
#        
#        #get hyperparameters for next iteration 
#        hyp = optimized_params
#        print('parameters for next iteration', hyp)
#        
#        loop += 1
#    
#    #TRAIN ADAPTATION AND TARGET MODELS
#    
#    print('----- TRAINING ADAPTATION AND TARGET MODELS -----')
#    
#    error_all = {}
#    
#    #make folders 
##    GT_MEAN_FOLD_FOLDER = os.path.join(GT_MEAN_FOLDER_DIR, fold[i]) #name of folder  
##    pathlib.Path(GT_MEAN_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
#    
##    GT_MEAN_EXTRACTED_FOLD_FOLDER = os.path.join(GT_MEAN_EXTRACTED_FOLDER_DIR, fold[i]) #name of folder  
##    pathlib.Path(GT_MEAN_EXTRACTED_FOLD_FOLDER).mkdir(parents=True, exist_ok=True) #make folder 
#    
##    ERROR_FOLD_FOLDER_DIR = os.path.join(ERROR_FOLDER_DIR, fold[i]) #name of folder  
##    pathlib.Path(ERROR_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #make folder 
#    
#    #output mean and variance predictions for source, adaptation, and target models 
#    #iterate over test patients 
#    for ID in tst_ind: 
#        
#        print('----- TEST PATIENT: %s -----'%(ID))
#        
#        x_a_patient = x_a[ID][:-1,:]
#        y_a_patient = y_a[ID][:-1,:]
#        xtest = x_a[ID]
#        
#        #predictions for baseline model 
#        m_b_patient = base_model.predict(xtest)
#        
#        predictions = call_deep_pgp(deep_model, gp_layer, x_a_patient, y_a_patient, x_s, y_s, xtest)
#        
#        g_t_patient = g_t[ID]
#        m_s_patient = predictions['source model mu']
#        m_a_patient = predictions['adapted model mu']
#        m_t_patient = predictions['target model mu']
#
#        x_rows, x_cols = g_t_patient.shape 
#        m_s_patient = np.reshape(m_s_patient, (x_rows, y_a_patient.shape[1]))
#        m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
#        m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
#        
#        #compute mean absolute error
#        e_b = mean_absolute_error(m_b_patient, g_t_patient)
#        e_s = mean_absolute_error(m_s_patient, g_t_patient)
#        e_a = mean_absolute_error(m_a_patient, g_t_patient)
#        e_t = mean_absolute_error(m_t_patient, g_t_patient)
#
#        error_all[ID] = [e_b, e_s, e_a, e_t]
#
#        #write ground truth and mu values to csv
#        with open(GT_MEAN_DIR, 'a') as mean_csv_file:
#            w = csv.writer(mean_csv_file)
#            for i in range(len(g_t_patient)):
#                w.writerow([ID, g_t_patient[i], m_b_patient[i], m_s_patient[i], m_a_patient[i], m_t_patient[i]])
#    
#    #write error values to csv
#    with open(ERROR_DIR, 'a') as error_csv_file:
#        w = csv.writer(error_csv_file)
#        for key, value in error_all.items():
#            value = list(value)
#            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])
#        
##write average error values to csv 
#column_sums = None 
#
#with open (ERROR_DIR) as error_csv_file:
#    all_lines = error_csv_file.readlines()
#    lines = all_lines[1:]
#    rows_of_numbers = [map(float, line.split(',')) for line in lines]
#    sums = map(sum, zip(*rows_of_numbers))
#    averages = [sum_item / len(lines) for sum_item in sums]
#    averages[0] = 'average error'
#    
#with open (ERROR_DIR, 'a') as error_csv_file:
#    w = csv.writer(error_csv_file)
#    w.writerow(averages)