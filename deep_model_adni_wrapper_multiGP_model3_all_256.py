## Model 3 
'''
Trains DNN layer with all source data  
'''

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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

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

MAT_FOLDER_DIR = os.path.join(RESULTS_FOLDER_DIR, 'm3_mat')
pathlib.Path(MAT_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 

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

g1_ID_all = list(classif_data[:,0][g1_inds])
g2_ID_all = list(classif_data[:,0][g2_inds])
g3_ID_all = list(classif_data[:,0][g3_inds])

g1_all_train_error = {}
g1_all_test_error = {}
g2_all_train_error = {}
g2_all_test_error = {}
g3_all_train_error = {}
g3_all_test_error = {}

# Loop for 4 folds, plot error 
for i in range(0,1): #0 to 3
#for i in range(0,1): #0 to 3
    
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
    #### BASE MODEL ######
    ######################
    
    print('----- BUILDING BASE MODEL -----')
    
    x_s = np.vstack((g1_x_s, g2_x_s, g3_x_s))
    y_s = np.vstack((g1_y_s, g2_y_s, g3_y_s))
    xtest_all = np.vstack((g1_xtest_all, g2_xtest_all, g3_xtest_all))
    ytest_all = np.vstack((g1_ytest_all, g2_ytest_all, g3_ytest_all))
    
    base_weights, g1_z, g2_z, g3_z, base_model = call_base_model_m3_all(x_s, y_s, xtest_all, ytest_all, g1_x_s, g2_x_s, g3_x_s, deep_layer_dims = 256) 
    
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    # For group 1: 
    g1_max_x_s = np.amax(g1_z, axis = 0)
    g1_min_x_s = np.amin(g1_z, axis = 0)
    g1_var_y_s = np.sum(g1_y_s**2)/(len(g1_y_s)-1) - (len(g1_y_s))*np.mean(g1_y_s)**2/(len(g1_y_s)-1) 
    g1_initial_lik = np.log(np.sqrt(0.1*g1_var_y_s))
    g1_initial_cov = np.array([[np.log(np.median(g1_max_x_s - g1_min_x_s))], [np.log(np.sqrt(g1_var_y_s))]])
    print('Initial Parameters:', g1_initial_lik, g1_initial_cov)
    # For group 2: 
    g2_max_x_s = np.amax(g2_z, axis = 0)
    g2_min_x_s = np.amin(g2_z, axis = 0)
    g2_var_y_s = np.sum(g2_y_s**2)/(len(g2_y_s)-1) - (len(g2_y_s))*np.mean(g2_y_s)**2/(len(g2_y_s)-1) 
    g2_initial_lik = np.log(np.sqrt(0.1*g2_var_y_s))
    g2_initial_cov = np.array([[np.log(np.median(g2_max_x_s - g2_min_x_s))], [np.log(np.sqrt(g2_var_y_s))]])
    print('Initial Parameters:', g2_initial_lik, g2_initial_cov)
    # For group 3: 
    g3_max_x_s = np.amax(g3_z, axis = 0)
    g3_min_x_s = np.amin(g3_z, axis = 0)
    g3_var_y_s = np.sum(g3_y_s**2)/(len(g3_y_s)-1) - (len(g3_y_s))*np.mean(g3_y_s)**2/(len(g3_y_s)-1) 
    g3_initial_lik = np.log(np.sqrt(0.1*g3_var_y_s))
    g3_initial_cov = np.array([[np.log(np.median(g3_max_x_s - g3_min_x_s))], [np.log(np.sqrt(g3_var_y_s))]])
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
    
    loop_count = 5
    
    for b in range(loop_count):
        print('LOOP COUNT:', b+1)
        
        g1_MAT_LOOP_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g1_loop_%s'%(b+1))
        pathlib.Path(g1_MAT_LOOP_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
        
        g2_MAT_LOOP_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g2_loop_%s'%(b+1))
        pathlib.Path(g2_MAT_LOOP_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
        
        g3_MAT_LOOP_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g3_loop_%s'%(b+1))
        pathlib.Path(g3_MAT_LOOP_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
        
        if b == 0:
            g1_z_modified = g1_z
            g1_hyp = g1_initial_hyp 
            g2_z_modified = g2_z
            g2_hyp = g2_initial_hyp 
            g3_z_modified = g3_z
            g3_hyp = g3_initial_hyp 
            # Get weights corresponding to DNN layer 
            dnn_weights = base_weights[:2] 
            
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
        # Randomly choose order to train DNN + KGP model (note: shuffle(x) mutates list)
        x = [0, 1, 2]
        shuffle(x)
        # x = [2, 0, 1]
        
        x_dict_s1, y_dict_s1, x_s1, y_s1, xtest_all1, ytest_all1, x_a1, y_a1, g_t1, g_t_all1, tst_inds1, tr_inds1 = group_data[x[0]]
        x_dict_s2, y_dict_s2, x_s2, y_s2, xtest_all2, ytest_all2, x_a2, y_a2, g_t2, g_t_all2, tst_inds2, tr_inds2 = group_data[x[1]]
        x_dict_s3, y_dict_s3, x_s3, y_s3, xtest_all3, ytest_all3, x_a3, y_a3, g_t3, g_t_all3, tst_inds3, tr_inds3 = group_data[x[2]]
        
        group_hyp = (g1_hyp, g2_hyp, g3_hyp)
        group_z = (g1_z, g2_z, g3_z)
        
        #prepare GP layer with optimized parameters 
        # For random set 1: 
        gp_layer1_1 = GP(
                        hyp = group_hyp[x[0]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s.shape[0])
        
        inputs_dgp1_1 = Input(shape = (x_s.shape[1], ))
        deep_layer_dgp1_1 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_1)
        outputs_dgp1_1 = gp_layer1_1(deep_layer_dgp1_1)
        
        deep_model1_1 = kgpModel(inputs = inputs_dgp1_1, outputs = outputs_dgp1_1)        
        
        #compile model 
        deep_model1_1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
                
        #fit model
        deep_model1_1.fit(x_s, y_s, 
                        validation_data = (xtest_all, ytest_all), 
                        batch_size = 100, 
                        epochs = 5, 
                        callbacks = cb,
                        verbose = 1)
        
        # Get new dnn_weights  
        dnn_weights = deep_model1_1.get_weights()
        
        # For random set 2: 
        gp_layer1_2 = GP(
                        hyp = group_hyp[x[1]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s.shape[0])
        
        inputs_dgp1_2 = Input(shape = (x_s.shape[1], ))
        deep_layer_dgp1_2 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_2)
        outputs_dgp1_2 = gp_layer1_2(deep_layer_dgp1_2)
        
        deep_model1_2 = kgpModel(inputs = inputs_dgp1_2, outputs = outputs_dgp1_2)       
        
        #compile model 
        deep_model1_2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
                
        #fit model
        deep_model1_2.fit(x_s, y_s, 
                        validation_data = (xtest_all, ytest_all), 
                        batch_size = 100, 
                        epochs = 5, 
                        callbacks = cb,
                        verbose = 1)
        
        # Get new dnn_weights  
        dnn_weights = deep_model1_2.get_weights()
        
        # For random set 3: 
        gp_layer1_3 = GP(
                        hyp = group_hyp[x[2]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s.shape[0])
        
        inputs_dgp1_3 = Input(shape = (x_s.shape[1], ))
        deep_layer_dgp1_3 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_3)
        outputs_dgp1_3 = gp_layer1_3(deep_layer_dgp1_3)
        
        deep_model1_3 = kgpModel(inputs = inputs_dgp1_3, outputs = outputs_dgp1_3)
        
        #compile model 
        deep_model1_3.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
                
        #fit model
        deep_model1_3.fit(x_s, y_s, 
                        validation_data = (xtest_all, ytest_all), 
                        batch_size = 100, 
                        epochs = 5, 
                        callbacks = cb,
                        verbose = 1)
        
        # Get new dnn_weights  
        dnn_weights = deep_model1_3.get_weights()
        
        # Rebuild all models with new dnn_weights 
        # model 1 
        gp_layer1_1 = GP(
                        hyp = group_hyp[x[0]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s1.shape[0])
        inputs_dgp1_1 = Input(shape = (x_s1.shape[1], ))
        deep_layer_dgp1_1 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_1)
        outputs_dgp1_1 = gp_layer1_1(deep_layer_dgp1_1)
        deep_model1_1 = kgpModel(inputs = inputs_dgp1_1, outputs = outputs_dgp1_1)        
        deep_model1_1.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        # model 2 
        gp_layer1_2 = GP(
                        hyp = group_hyp[x[1]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s2.shape[0])
        inputs_dgp1_2 = Input(shape = (x_s2.shape[1], ))
        deep_layer_dgp1_2 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_2)
        outputs_dgp1_2 = gp_layer1_2(deep_layer_dgp1_2)
        deep_model1_2 = kgpModel(inputs = inputs_dgp1_2, outputs = outputs_dgp1_2)        
        deep_model1_2.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        # model 3
        gp_layer1_3 = GP(
                        hyp = group_hyp[x[2]],
                        inf = 'infExact',
                        dlik = 'dlikExact',
                #        opt = {'cg_maxit': 100, 'cg_tol': 1e-6}, #doesn't affect result
                        mean = 'meanZero',
                #        grid_kwargs={'eq': 1, 'k': 100.}, #doesn't affect result
                        cov = 'covSEiso',
                        batch_size = 100,
                        nb_train_samples = x_s3.shape[0])
        inputs_dgp1_3 = Input(shape = (x_s3.shape[1], ))
        deep_layer_dgp1_3 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = True)(inputs_dgp1_3)
        outputs_dgp1_3 = gp_layer1_3(deep_layer_dgp1_3)
        deep_model1_3 = kgpModel(inputs = inputs_dgp1_3, outputs = outputs_dgp1_3)        
        deep_model1_3.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        # Get new z_modified 
        x_s_activations1 = get_activations(deep_model1_1, x_s1)
        z_modified1 = x_s_activations1[-2] 
        
        x_s_activations2 = get_activations(deep_model1_2, x_s2)
        z_modified2 = x_s_activations2[-2]  
        
        x_s_activations3 = get_activations(deep_model1_3, x_s3)
        z_modified3 = x_s_activations3[-2]  
        
        group_deep_model1 = (deep_model1_1, deep_model1_2, deep_model1_3)
        group_z_modified = (z_modified1, z_modified2, z_modified3)
        
        g1_ind = x.index(0)
        g2_ind = x.index(1)
        g3_ind = x.index(2)
        
        g1_deep_model1 = group_deep_model1[g1_ind]
        g2_deep_model1 = group_deep_model1[g2_ind]
        g3_deep_model1 = group_deep_model1[g3_ind]
        
        g1_z_modified = group_z_modified[g1_ind]
        g2_z_modified = group_z_modified[g2_ind]
        g3_z_modified = group_z_modified[g3_ind]
        
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
        g1_deep_layer_dgp2 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = False)(g1_inputs_dgp2)
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
        g2_deep_layer_dgp2 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = False)(g2_inputs_dgp2)
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
        g3_deep_layer_dgp2 = Dense(256, activation = 'relu', weights = dnn_weights, trainable = False)(g3_inputs_dgp2)
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
                       epochs = 5,
                       callbacks = cb,
                       verbose = 1)
        g2_deep_model2.fit(g2_x_s, g2_y_s,
                       validation_data = (g2_xtest_all, g2_ytest_all), 
                       batch_size = 100,
                       epochs = 5,
                       callbacks = cb,
                       verbose = 1)
        g3_deep_model2.fit(g3_x_s, g3_y_s,
                       validation_data = (g3_xtest_all, g3_ytest_all), 
                       batch_size = 100,
                       epochs = 5,
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
        
        g1_MAT_LOOP_TRAINING_FOLDER_DIR = os.path.join(g1_MAT_LOOP_FOLDER_DIR, 'training')
        pathlib.Path(g1_MAT_LOOP_TRAINING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        g1_MAT_LOOP_TESTING_FOLDER_DIR = os.path.join(g1_MAT_LOOP_FOLDER_DIR, 'testing')
        pathlib.Path(g1_MAT_LOOP_TESTING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        g2_MAT_LOOP_TRAINING_FOLDER_DIR = os.path.join(g2_MAT_LOOP_FOLDER_DIR, 'training')
        pathlib.Path(g2_MAT_LOOP_TRAINING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        g2_MAT_LOOP_TESTING_FOLDER_DIR = os.path.join(g2_MAT_LOOP_FOLDER_DIR, 'testing')
        pathlib.Path(g2_MAT_LOOP_TESTING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        g3_MAT_LOOP_TRAINING_FOLDER_DIR = os.path.join(g3_MAT_LOOP_FOLDER_DIR, 'training')
        pathlib.Path(g3_MAT_LOOP_TRAINING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    
        
        g3_MAT_LOOP_TESTING_FOLDER_DIR = os.path.join(g3_MAT_LOOP_FOLDER_DIR, 'testing')
        pathlib.Path(g3_MAT_LOOP_TESTING_FOLDER_DIR).mkdir(parents=True, exist_ok=True)    

        for ID in g1_fold_tr_ind: 
            
            print('----- TRAINING PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g1_MAT_LOOP_TRAINING_FOLDER_DIR, ID, g1_x_s, g1_y_s, g1_x_dict_s, g1_y_dict_s, g1_y_dict_s, g1_deep_model2, g1_sgp_train_m, g1_sgp_train_s, g1_tr_inds, optimized_params=None, gp_layer=g1_gp_layer2, kgp_model=True, tst_pat=False)
                
        for ID in g1_fold_tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g1_MAT_LOOP_TESTING_FOLDER_DIR, ID, g1_x_s, g1_y_s, g1_x_a, g1_y_a, g1_g_t, g1_deep_model2, g1_sgp_test_m, g1_sgp_test_s, g1_tst_inds, optimized_params=None, gp_layer=g1_gp_layer2, kgp_model=True, tst_pat=True)

        for ID in g2_fold_tr_ind: 
            
            print('----- TRAINING PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g2_MAT_LOOP_TRAINING_FOLDER_DIR, ID, g2_x_s, g2_y_s, g2_x_dict_s, g2_y_dict_s, g2_y_dict_s, g2_deep_model2, g2_sgp_train_m, g2_sgp_train_s, g2_tr_inds, optimized_params=None, gp_layer=g2_gp_layer2, kgp_model=True, tst_pat=False)
                
        for ID in g2_fold_tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g2_MAT_LOOP_TESTING_FOLDER_DIR, ID, g2_x_s, g2_y_s, g2_x_a, g2_y_a, g2_g_t, g2_deep_model2, g2_sgp_test_m, g2_sgp_test_s, g2_tst_inds, optimized_params=None, gp_layer=g2_gp_layer2, kgp_model=True, tst_pat=True)
            
        for ID in g3_fold_tr_ind: 
            
            print('----- TRAINING PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g3_MAT_LOOP_TRAINING_FOLDER_DIR, ID, g3_x_s, g3_y_s, g3_x_dict_s, g3_y_dict_s, g3_y_dict_s, g3_deep_model2, g3_sgp_train_m, g3_sgp_train_s, g3_tr_inds, optimized_params=None, gp_layer=g3_gp_layer2, kgp_model=True, tst_pat=False)
                
        for ID in g3_fold_tst_ind: 
            
            print('----- TEST PATIENT: %s -----'%(ID))
            
            _ = save_to_mat(g3_MAT_LOOP_TESTING_FOLDER_DIR, ID, g3_x_s, g3_y_s, g3_x_a, g3_y_a, g3_g_t, g3_deep_model2, g3_sgp_test_m, g3_sgp_test_s, g3_tst_inds, optimized_params=None, gp_layer=g3_gp_layer2, kgp_model=True, tst_pat=True)
        
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
        
        if b != loop_count-1: 
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
    
    g1_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g1_final')
    pathlib.Path(g1_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    g2_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g2_final')
    pathlib.Path(g2_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    g3_MAT_FINAL_FOLDER_DIR = os.path.join(MAT_FOLD_FOLDER_DIR, 'g3_final')
    pathlib.Path(g3_MAT_FINAL_FOLDER_DIR).mkdir(parents=True, exist_ok=True) 
    
    #output mean and variance predictions for source, adaptation, and target models 
    #iterate over test patients (group 1)
    for ID in g1_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g1_m_s_patient, g1_s_s_patient, g1_y_a_patient = save_to_mat_final(g1_MAT_FINAL_FOLDER_DIR, ID, g1_x_s, g1_y_s, g1_x_a, g1_y_a, g1_g_t, g1_deep_model2, g1_gp_layer2, g1_sgp_test_m, g1_sgp_test_s, g1_tst_inds, base_weights, deep_layer_dims=256, tst_pat=True)
    
    #iterate over test patients (group 2)
    for ID in g2_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g2_m_s_patient, g2_s_s_patient, g2_y_a_patient = save_to_mat_final(g2_MAT_FINAL_FOLDER_DIR, ID, g2_x_s, g2_y_s, g2_x_a, g2_y_a, g2_g_t, g2_deep_model2, g2_gp_layer2, g2_sgp_test_m, g2_sgp_test_s, g2_tst_inds, base_weights, deep_layer_dims=256, tst_pat=True)
    
    #iterate over test patients (group 3)
    for ID in g3_fold_tst_ind: 
        
        print('----- TEST PATIENT: %s -----'%(ID))
        
        g3_m_s_patient, g3_s_s_patient, g3_y_a_patient = save_to_mat_final(g3_MAT_FINAL_FOLDER_DIR, ID, g3_x_s, g3_y_s, g3_x_a, g3_y_a, g3_g_t, g3_deep_model2, g3_gp_layer2, g3_sgp_test_m, g3_sgp_test_s, g3_tst_inds, base_weights, deep_layer_dims=256, tst_pat=True)
