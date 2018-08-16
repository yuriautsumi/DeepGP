

#######################################
## Deep Model Adaptation of call_pgp ## 
#######################################

from call_pgp import * 
from get_activation import * 

from kgp.models import Model as kgpModel 
from kgp.layers import GP 

import numpy as np 

def call_deep_source(deep_model, gp_layer, x_s, y_s, x_a, xtest):
    """
    Input: trained deep kgp model, GP layer, source data, adaptation input, test input 
    Output: m_s, s_s, GP parameters, modified input values 
    """
    #predict mean and variance 
    m_s_array, s_s_array = deep_model.predict(xtest, x_s, y_s, return_var = True, verbose = 0)
    
    m_s = m_s_array[0].flatten().tolist()
    s_s = s_s_array[0].flatten().tolist()
    
    #get modified input values 
    x_s_activations = get_activations(deep_model, x_s)
    x_s_modified = x_s_activations[-2]
    
    x_a_activations = get_activations(deep_model, x_a)
    x_a_modified = x_a_activations[-2]
    
    xtest_activations = get_activations(deep_model, xtest)
    xtest_modified = xtest_activations[-2]
    
    #get GP parameters 
    hyp = gp_layer.hyp 
    ls = np.exp(hyp['cov'][0])
    mul = [x_s_modified.shape[1]] 
    var = np.exp(2*hyp['cov'][1])
    sn2 = np.exp(2*hyp['lik'])
    
    return m_s, s_s, ls, mul, var, sn2, x_s_modified, x_a_modified, xtest_modified 

def call_deep_pgp(deep_model, gp_layer, x_a, y_a, x_s, y_s, xtest):
    """
    Input: trained deep kgp model, GP layer, source data, adaptation data, test input data 
    Output: m_s, s_s, m_a, s_a, m_t, s_t, g_t, GP parameters 
    """
    
    #SOURCE MODEL 
    print('----- CALL SOURCE MODEL -----')
    m_s, s_s, ls, mul, var, sn2, x_s_modified, x_a_modified, xtest_modified = call_deep_source(deep_model, gp_layer, x_s, y_s, x_a, xtest)    
    
    #ADAPTATION MODEL 
    print('----- CALL ADAPTATION MODEL -----')
    m_a, s_a = call_adapt(x_a_modified, y_a, x_s_modified, y_s, xtest_modified, m_s, s_s, ls, mul, var, sn2)
    
    #TARGET MODEL
    print('----- CALL TARGET MODEL -----')
    m_t, s_t = call_target(x_a_modified, y_a, x_s_modified, y_s, xtest_modified, m_s, s_s, ls, mul, var, sn2)
    
    #outputs 
    m_a = list(m_a.flatten())
    s_a = list(s_a.flatten())

    m_t = list(m_t.flatten())
    s_t = list(s_t.flatten())

    values = {'source model mu': m_s, 'source model sigma': s_s, 'adapted model mu': m_a, 'adapted model sigma': s_a, 'target model mu': m_t, 'target model sigma': s_t, 'param - kernel ls': ls, 'param - kernel var': var, 'param - likelihood var': sn2}
    
    return values 