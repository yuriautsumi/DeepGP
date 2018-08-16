

###################################
## GPML Implementation in Python ##
###################################

from kgp.backend.gpml import GPML

def trainGP(x_s, y_s, input_dim, hyp, opt, inf, mean, cov, lik, dlik, iters=5):
    """
    Creates instance of GPML class, configures GP, trains model.
    Return: model and model parameters
    
    Input: source data, test data, model configuration variables
    Output: dict of model parameters
    """
    #create instance of GPML class
    model = GPML(engine = 'octave', engine_kwargs = None, gpml_path = None)

    #configure GP model
    model.configure(input_dim, hyp, opt, inf, mean, cov, lik, dlik, verbose = 1)  

    #train GP model
    params = model.train(n_iter = iters, X_tr = x_s, y_tr = y_s, verbose = 1) 
#    params = model.train(n_iter = 100, X_tr = x_s, y_tr = y_s, verbose = 1) 
    
    return model, params


def predictGP(x_s, y_s, xtest, model):
    """
    Return: mean and variance based on trained source model
    
    Input: source data, test data, trained source model
    Output: mean, variance
    """
    # predict mean and variance
    m_s, s_s = model.predict(X = xtest, X_tr = x_s, y_tr = y_s, return_var = True, verbose = 1)

    return m_s, s_s 