## Model 2 

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

MAT_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'm2_mat')
pathlib.Path(MAT_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

# Create CSV files 
M2_GT_MEAN_DIR = os.path.join(M2_FOLDER_DIR, 'gt_mean.csv')
M2_ERROR_DIR = os.path.join(M2_FOLDER_DIR, 'mse.csv')
M2_HYP_DIR = os.path.join(M2_FOLDER_DIR, 'hyp.csv')
#M2_WEIGHTS_DIR = os.path.join(M2_FOLDER_DIR, 'dnn_weights.h5')
#M2_ACTIV_DIR = os.path.join(M2_FOLDER_DIR, 'activ.csv')


all_train_error = {}
all_test_error = {}

# Loop for 10 folds, plot error 
for i in range(0,1): #0 to 9
    
    fold_num = i+1
    
    print('FOLD:', fold_num)
    
    MAT_FOLD_FOLDER_DIR = os.path.join(MAT_FOLDER_DIR, 'fold_%s'%(i+1))
    pathlib.Path(MAT_FOLD_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
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
    max_x_s = np.amax(x_s, axis = 0)
    min_x_s = np.amin(x_s, axis = 0)
    var_y_s = np.sum(y_s**2)/(len(y_s)-1) - (len(y_s))*np.mean(y_s)**2/(len(y_s)-1) 
    initial_lik = np.log(np.sqrt(0.1*var_y_s))
    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(var_y_s))]])
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
    
#    # Build and train deep GP model 
    sgp_fold_train_error = []
    sgp_fold_test_error = []
#    pgp_fold_train_error = [] # added 
#    pgp_fold_test_error = [] # added 
#    tgp_fold_train_error = [] # added 
#    tgp_fold_test_error = [] # added 
        
    loop = 25 
    
    for b in range(loop):
#    for b in range(100):
        print('LOOP COUNT:', b+1)
        
        MAT_LOOP_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'loop_%s'%(b+1))
        pathlib.Path(MAT_LOOP_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
        
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
                        epochs = 5, 
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
                       epochs = 5,
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
        
#        # added 
#        #predict pgp results and compute error 
#        all_patient_pgp_fold_train_error = []
#        all_patient_pgp_fold_test_error = []
#        all_patient_tgp_fold_train_error = []
#        all_patient_tgp_fold_test_error = []
        
        MAT_LOOP_TRAINING_FOLDER_DIR = os.path.join(MAT_LOOP_FOLDER_DIR, 'training')
        pathlib.Path(MAT_LOOP_TRAINING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    

        MAT_LOOP_TESTING_FOLDER_DIR = os.path.join(MAT_LOOP_FOLDER_DIR, 'testing')
        pathlib.Path(MAT_LOOP_TESTING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        
        
        for ID in tr_ind_source: 
            
            print('----- TRAINING PATIENT: %s -----'%(ID))
            
            m_s_patient, s_s_patient, y_a_patient  = save_to_mat(MAT_LOOP_TRAINING_FOLDER_DIR, ID, x_s, y_s, x_dict_s, y_dict_s, y_dict_s, deep_model2, sgp_train_m, sgp_train_s, optimized_params=None, gp_layer=gp_layer2, kgp_model=True)
            
#            x_a_patient = x_dict_s[ID][:-1,:]
#            y_a_patient = y_dict_s[ID][:-1,:]
#            xtest = x_dict_s[ID]
#            
#            ls = np.exp(optimized_params['cov'][0])
#            mul = [x_a_patient.shape[1]]
#            var = np.exp(2*optimized_params['cov'][1])
#            sn2 = np.exp(2*optimized_params['lik'])
#            
#            g_t_patient = g_t[ID]
#            m_s_patient, s_s_patient = predictGP(x_s, y_s, xtest, gpml_model)
#            
#            predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, xtest)
#            
#            g_t_patient = y_dict_s[ID]
#            m_a_patient = predictions['adapted model mu']
#            m_t_patient = predictions['target model mu']
#    
#            x_rows, x_cols = g_t_patient.shape 
#            m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
#            m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
#            
#            #compute mean absolute error
#            e_a = mean_absolute_error(m_a_patient, g_t_patient)
#            e_t = mean_absolute_error(m_t_patient, g_t_patient)
#    
#            #append to list 
#            all_patient_pgp_fold_train_error.append(e_a)
#            all_patient_tgp_fold_train_error.append(e_t)
        
        for ID in tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            m_s_patient, s_s_patient, y_a_patient  = save_to_mat(MAT_LOOP_TESTING_FOLDER_DIR, ID, x_s, y_s, x_a, y_a, g_t, deep_model2, sgp_test_m, sgp_test_s, optimized_params=None, gp_layer=gp_layer2, kgp_model=True)
            
#            x_a_patient = x_a[ID][:-1,:]
#            y_a_patient = y_a[ID][:-1,:]
#            xtest = x_a[ID]
#            
#            predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, xtest)
#            
#            g_t_patient = g_t[ID]
#            m_a_patient = predictions['adapted model mu']
#            m_t_patient = predictions['target model mu']
#    
#            x_rows, x_cols = g_t_patient.shape 
#            m_a_patient = np.reshape(m_a_patient, (x_rows, y_a_patient.shape[1]))
#            m_t_patient = np.reshape(m_t_patient, (x_rows, y_a_patient.shape[1]))
#            
#            #compute mean absolute error
#            e_a = mean_absolute_error(m_a_patient, g_t_patient)
#            e_t = mean_absolute_error(m_t_patient, g_t_patient)
#    
#            #append to list 
#            all_patient_pgp_fold_test_error.append(e_a)
#            all_patient_tgp_fold_test_error.append(e_t)
            
#        pgp_fold_train_error.append(sum(all_patient_pgp_fold_train_error)/len(all_patient_pgp_fold_train_error))
#        pgp_fold_test_error.append(sum(all_patient_pgp_fold_test_error)/len(all_patient_pgp_fold_test_error))
#        tgp_fold_train_error.append(sum(all_patient_tgp_fold_train_error)/len(all_patient_tgp_fold_train_error))
#        tgp_fold_test_error.append(sum(all_patient_tgp_fold_test_error)/len(all_patient_tgp_fold_test_error))
        
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
        
        if b != loop-1: 
            #clear session 
            K.clear_session()
    
    # Save and add errors to dictionary, indexed by fold number 
    all_train_error[fold_num] = sgp_fold_train_error
    all_test_error[fold_num] = sgp_fold_test_error
    
    SGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_sgp_train_error.csv'%(fold_num))
    SGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_sgp_test_error.csv'%(fold_num))
    np.savetxt(SGP_FOLD_TRAIN_ERROR_CSV_DIR, sgp_fold_train_error, delimiter=",")
    np.savetxt(SGP_FOLD_TEST_ERROR_CSV_DIR, sgp_fold_test_error, delimiter=",")
    
#    # added 
#    PGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_pgp_train_error.csv'%(fold_num))
#    PGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_pgp_test_error.csv'%(fold_num))
#    np.savetxt(PGP_FOLD_TRAIN_ERROR_CSV_DIR, pgp_fold_train_error, delimiter=",")
#    np.savetxt(PGP_FOLD_TEST_ERROR_CSV_DIR, pgp_fold_test_error, delimiter=",")    
#
#    TGP_FOLD_TRAIN_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_tgp_train_error.csv'%(fold_num))
#    TGP_FOLD_TEST_ERROR_CSV_DIR = os.path.join(M2_FOLDER_DIR, 'fold_%s_tgp_test_error.csv'%(fold_num))
#    np.savetxt(TGP_FOLD_TRAIN_ERROR_CSV_DIR, tgp_fold_train_error, delimiter=",")
#    np.savetxt(TGP_FOLD_TEST_ERROR_CSV_DIR, tgp_fold_test_error, delimiter=",")    
#    # added 
    
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
    
    MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'final')
    pathlib.Path(MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    for ID in tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        m_s_patient, s_s_patient, y_a_patient = save_to_mat_final(MAT_FINAL_FOLDER_DIR, ID, x_s, y_s, x_a, y_a, g_t, deep_model2, gp_layer2, sgp_test_m, sgp_test_s, base_weights)
        
#        x_a_patient = x_a[ID][:-1,:]
#        y_a_patient = y_a[ID][:-1,:]
#        xtest = x_a[ID]
#        
#        #rebuild baseline model 
#        base_model = rebuild_base_model(x_s, y_s, base_weights, deep_layer_dims = 128)
#        
#        #predictions for baseline model 
#        m_b_patient = base_model.predict(xtest)
#        
#        predictions = call_deep_pgp(deep_model2, gp_layer2, x_a_patient, y_a_patient, x_s, y_s, xtest)
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
#        with open(M2_GT_MEAN_DIR, 'a') as mean_csv_file:
#            w = csv.writer(mean_csv_file)
#            for i in range(len(g_t_patient)):
#                w.writerow([ID, g_t_patient[i], m_b_patient[i], m_s_patient[i], m_a_patient[i], m_t_patient[i]])
#    
#    #write error values to csv
#    with open(M2_ERROR_DIR, 'a') as error_csv_file:
#        w = csv.writer(error_csv_file)
#        for key, value in error_all.items():
#            value = list(value)
#            w.writerow([key, value[0][0], value[1][0], value[2][0], value[3][0]])
#
##write average error values to csv 
#column_sums = None 
#
#with open (M2_ERROR_DIR) as error_csv_file:
#    all_lines = error_csv_file.readlines()
#    lines = all_lines[1:]
#    rows_of_numbers = [map(float, line.split(',')) for line in lines]
#    sums = map(sum, zip(*rows_of_numbers))
#    averages = [sum_item / len(lines) for sum_item in sums]
#    averages[0] = 'average error'
#    
#with open (M2_ERROR_DIR, 'a') as error_csv_file:
#    w = csv.writer(error_csv_file)
#    w.writerow(averages)
