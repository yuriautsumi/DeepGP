

############################
## Wrapper for Deep Model ##
############################

#import methods
from call_gpml import *
from get_activation import *
from compute_error import *
from call_deep_pgp import * 

import os
import csv
import numpy as np
import itertools 
import pathlib

#set to use GPU... '0' or '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

######################
##### PREP DATA ######
######################

print('----- PREPARING DATA -----')

#define directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_DIR = os.path.join(CURRENT_DIR, 'adni_data_all_norm_MMSE.csv') #data from Oggi

ID_DIR = os.path.join(CURRENT_DIR, 'Patient_RIDs_more_than_10_Visits_Less_Than_82_5_Perc_Missing_NoHeaders.csv')

#GT_MEAN_FOLDER_DIR = os.path.join(CURRENT_DIR, 'gt_mean')
#pathlib.Path(GT_MEAN_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #makes folder
#
#GT_MEAN_EXTRACTED_FOLDER_DIR = os.path.join(CURRENT_DIR, 'gt_mean_extracted')
#pathlib.Path(GT_MEAN_EXTRACTED_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #makes folder
#
#ERROR_FOLDER_DIR = os.path.join(CURRENT_DIR, 'mean_squared_error')
#pathlib.Path(ERROR_FOLDER_DIR).mkdir(parents=True, exist_ok=True) #makes folder

#create csv files 
gt_mean_csv_name = 'ground_truth_mean.csv'
GT_MEAN_DIR = os.path.join(CURRENT_DIR, gt_mean_csv_name)

gt_mean_extracted_csv_name = 'ground_truth_mean_extracted.csv'
GT_MEAN_EXTRACTED_DIR = os.path.join(CURRENT_DIR, gt_mean_extracted_csv_name)

error_csv_name = 'mean_squared_error.csv'
ERROR_DIR = os.path.join(CURRENT_DIR, error_csv_name)

#write csv headers 
with open(GT_MEAN_DIR, 'w') as mean_csv_file:
    w = csv.writer(mean_csv_file)
    w.writerow(['ID', 'ground truth', 'base model mu', 'source model mu', 'adapted model mu', 'target model mu'])

with open(GT_MEAN_EXTRACTED_DIR, 'w') as mean_extracted_csv_file:
    w = csv.writer(mean_extracted_csv_file)
    w.writerow(['ID', 'ground truth', 'base model mu', 'source model mu', 'adapted model mu', 'target model mu'])

with open(ERROR_DIR, 'w') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(['ID', 'base model error', 'source model error', 'adapted model error', 'target model error'])

#create list of IDs
with open(ID_DIR, 'r') as f:
    reader = csv.reader(f)
    ID_all = list(reader)
    ID_all = list(itertools.chain.from_iterable(ID_all)) #flatten list
    ID_all = list(map(int, ID_all)) #cast each element as int

#loop for 10 folds
#for i in range(0, 1):
for i in range(0, 10): #0 to 9
    tst_ind = ID_all[i*10:i*10+10]
    tr_ind_source = np.setdiff1d(ID_all, tst_ind)
    
    #create X_all, Y_all, value_all
    with open(CSV_DATA_DIR, 'r', encoding = 'utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter = " ", quotechar = '|')
        reader = list(reader)[0:]
        X_all = {}
        Y_all = {}
        value_all = {}
        for line in reader:
            line = line[0].split(',')
            line = list(map(float, line))
            ID = int(line[0])
            x_line = list(map(float, line[3:-2])) 
            y_line = [line[-2]]
            value_line = [line[-1]]
            if ID in ID_all: #check if ID is in ID_all
                if ID in X_all:
                    X_all[ID] = np.vstack((X_all[ID], x_line))
                    Y_all[ID] = np.vstack((Y_all[ID], y_line))
                    value_all[ID] = np.vstack((value_all[ID], value_line))
                else:
                    X_all[ID] = x_line
                    Y_all[ID] = y_line
                    value_all[ID] = value_line
    
#    #make each value a list of floats 
#    X_all = {k: v.tolist() for k, v in X_all.items()}
#    Y_all = {k: v.tolist() for k, v in Y_all.items()}
#    value_all = {k: v.tolist() for k, v in value_all.items()}
    
    #create x_s, y_s
    x_dict_s = {key:value for key, value in X_all.items() if key in tr_ind_source}
    y_dict_s = {key:value for key, value in Y_all.items() if key in tr_ind_source}
    
    x_list_s = tuple(x_dict_s.values())
    y_list_s = tuple(y_dict_s.values())

    x_s = np.vstack(x_list_s)
    y_s = np.vstack(y_list_s)
    
    #create x_a, y_a 
    x_a = {key:value for key, value in X_all.items() if key in tst_ind}
    y_a = {key:value for key, value in Y_all.items() if key in tst_ind}
    
    #ground truth 
    g_t = y_a 
    
    #create g_t_all 
    g_list_t = tuple(g_t.values())
    g_t_all = np.vstack(g_list_t)
    g_t_all = list(g_t_all.flatten())
    
    #create xtest_all, ytest_all
    x_dict_test_all = {key:value for key, value in X_all.items() if key in tst_ind}
    y_dict_test_all = {key:value for key, value in Y_all.items() if key in tst_ind}
    
    x_list_test_all = tuple(x_dict_test_all.values())
    y_list_test_all = tuple(y_dict_test_all.values())
    
    xtest_all = np.vstack(x_list_test_all)
    ytest_all = np.vstack(y_list_test_all)
    
    #create indices_all 
    indices_loop = 0
    indices_all = []
    add_value = 0 
    
    for ID in tst_ind: 
        print('ID:', ID)
        values = value_all[ID]
        values_array = np.array(values)
        values_array = np.insert(values_array, 0, 0)
        values_array = values_array[:-1]
        values_array = np.reshape(values_array, (len(g_t[ID]),1))
        indices = np.where(values_array == 0)
        indices = np.ndarray.tolist(indices[0])
        add_indices = list(map(lambda i:i + add_value, indices))
        indices_all.extend(add_indices)
        add_value = add_value + len(g_t[ID])
        indices_loop+=1 
    
    ######################
    #### BASE MODEL ######
    ######################
    
    print('----- BUILDING BASE MODEL -----')
    
    from keras.optimizers import Adadelta 
    from keras.layers import Input, Dense
    from keras.models import Model as kerasModel
    from keras.callbacks import EarlyStopping
    from kgp.models import Model as kgpModel
    from kgp.layers import GP 
    
    #build model 
    inputs = Input(shape = (x_s.shape[1], ))
    deep_layer = Dense(128, activation = 'relu')(inputs)
    outputs = Dense(1, activation = 'linear')(deep_layer) #last layer - units equal output dimension 
    
    base_model = kerasModel(inputs, outputs)
    
    base_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
         
    #fit on training data
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
    
    #get modified input --> z
    x_s_activations = get_activations(base_model, x_s)
    z = x_s_activations[-2] 
        
    ######################
    # OPTIMIZE WITH GPML #
    ######################
    
    print('----- OPTIMIZING WITH GPML -----')
    
    #calculate initial parameters
    max_x_s = np.amax(z, axis = 0)
    min_x_s = np.amin(z, axis = 0)
    initial_lik = np.log(np.sqrt(0.1*np.var(y_s)))
    initial_cov = np.array([[np.log(np.median(max_x_s - min_x_s))], [np.log(np.sqrt(np.var(y_s)))]])
    
    #input features for GP
    dimensions = z.shape[1]
    initial_hyp = {'lik': initial_lik, 'mean': [], 'cov': initial_cov}
    train_opt = {}
    inf_method = 'infExact'
    mean_fxn = 'meanZero'
    cov_fxn = 'covSEiso'
    lik_fxn = 'likGauss'
    dlik_fxn = 'dlikExact'
    
    #train model until error minimizes 
    loop = 1
    all_error = []
    avg_error_diff = 0 
    
    #while loop < 4 or avg_error_diff > -0.005: 
    while loop < 3: #will add error condition later... 
        
        print('LOOP COUNT:', loop)
        
        if loop == 1:
            z_modified = z 
            hyp = initial_hyp 
            
        gpml_model, optimized_params = trainGP(z_modified, y_s, dimensions, hyp, train_opt, inf_method, mean_fxn, cov_fxn, lik_fxn, dlik_fxn) 
        
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
                        batch_size = 100,
                        nb_train_samples = z.shape[0])
        
        #build model with GP layer
        outputs = gp_layer(deep_layer)
        deep_model = kgpModel(inputs = inputs, outputs = outputs)
        
        #compile model
        deep_model.compile(optimizer = Adadelta(), loss = 'mse', metrics = ['accuracy'])
        
        #fit model
        deep_model.fit(x_s, y_s,
                       validation_data = (xtest_all, ytest_all), 
                       batch_size = 100,
                       epochs = 5,
                       callbacks = cb,
                       verbose = 1)
        
        #predict and compute error
        y_predict, var_predict = deep_model.predict(xtest_all, x_s, y_s, return_var = True, verbose = 0)
        y_predict = list(y_predict[0].flatten())
        g_t_extracted = list(map(lambda i:g_t_all[i], indices_all))
        y_predict_extracted = list(map(lambda i:y_predict[i], indices_all))   
        error = mean_absolute_error(y_predict_extracted, g_t_extracted)
        all_error.append(error)
        
        #get z for next iteration 
        x_s_activations = get_activations(deep_model, x_s)
        z_modified = x_s_activations[-2] 
        
        #get hyperparameters for next iteration 
        hyp = optimized_params
        print('parameters for next iteration', hyp)
        
        loop += 1
    
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
        m_b_patient = m_b_patient.flatten().tolist()
        
        predictions = call_deep_pgp(deep_model, gp_layer, x_a_patient, y_a_patient, x_s, y_s, xtest)
        
        g_t_patient = g_t[ID]
        g_t_patient = list(g_t_patient.flatten())
        m_s_patient = predictions['source model mu']
        m_a_patient = predictions['adapted model mu']
        m_t_patient = predictions['target model mu']

        values = value_all[ID]
        values_array = np.array(values)
        values_array = np.insert(values_array, 0, 0)
        values_array = values_array[:-1]
        values_array = np.reshape(values_array, (len(g_t_patient),1))
        indices = np.where(values_array == 0)
        indices = np.ndarray.tolist(indices[0])
        
        #extract only values with valid data
        g_t_extracted = list(map(lambda i:g_t_patient[i], indices))
        m_b_extracted = list(map(lambda i:m_b_patient[i], indices))
        m_s_extracted = list(map(lambda i:m_s_patient[i], indices))
        m_a_extracted = list(map(lambda i:m_a_patient[i], indices))
        m_t_extracted = list(map(lambda i:m_t_patient[i], indices))

        #compute mean absolute error
        e_b = mean_absolute_error(m_b_extracted, g_t_extracted)
        e_s = mean_absolute_error(m_s_extracted, g_t_extracted)
        e_a = mean_absolute_error(m_a_extracted, g_t_extracted)
        e_t = mean_absolute_error(m_t_extracted, g_t_extracted)

        error_all[ID] = [e_b, e_s, e_a, e_t]

        #write ground truth and mu values to csv
        with open(GT_MEAN_DIR, 'a') as mean_csv_file:
            w = csv.writer(mean_csv_file)
            for i in range(len(g_t_patient)):
                w.writerow([ID, g_t_patient[i], m_b_patient[i], m_s_patient[i], m_a_patient[i], m_t_patient[i]])

        #write ground truth and mu values to csv
        with open(GT_MEAN_EXTRACTED_DIR, 'a') as mean_extracted_csv_file:
            w = csv.writer(mean_extracted_csv_file)
            for i in range(len(g_t_extracted)):
                w.writerow([ID, g_t_extracted[i], m_b_extracted[i], m_s_extracted[i], m_a_extracted[i], m_t_extracted[i]])
    
    #write error values to csv
    with open(ERROR_DIR, 'a') as error_csv_file:
        w = csv.writer(error_csv_file)
        for key, value in error_all.items():
            value = list(value)
            w.writerow([key, value[0], value[1], value[2], value[3]])
        
#write average error values to csv 
column_sums = None 

with open (ERROR_DIR) as error_csv_file:
    all_lines = error_csv_file.readlines()
    lines = all_lines[1:]
    rows_of_numbers = [map(float, line.split(',')) for line in lines]
    sums = map(sum, zip(*rows_of_numbers))
    averages = [sum_item / len(lines) for sum_item in sums]
    averages[0] = 'average error'
    
with open (ERROR_DIR, 'a') as error_csv_file:
    w = csv.writer(error_csv_file)
    w.writerow(averages)