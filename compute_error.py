

#####################
## Computing Error ##
#####################

import numpy as np

def mean_absolute_error(pred, labels):
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    diff = np.subtract(pred, labels)
    abs_diff = np.fabs(diff)
    return sum(abs_diff)/len(labels)