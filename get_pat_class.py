

# Group 1: CN 
# Group 2: MCI 
# Group 3: CN → MCI 
# Group 4: Anything → AD 

import os 
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_DIR = os.path.join(CURRENT_DIR, 'classified_patient_data')
CSV_DIR = os.path.join(CURRENT_DIR, 'adni_dx_100_l1_l4.csv')
CLASSIF_CSV_DIR = os.path.join(CURRENT_DIR, 'patient_classification.csv')

#create array of csv data 

data = np.genfromtxt(CSV_DIR, dtype=float, delimiter=',') 

#get # of patients who stay CN (1), stay MCI (2), convert from CN --> MCI (3), or convert to AD (4)

#verification arrays 
cn = np.full((21,1), 1)
mci = np.full((21,1), 2)

classif = np.array([0.,0.])

for p in range(100):
    p_data = data[p*21:p*21+21, 2:3]
    ID = data[p*21,0]
    
    cn_bool = (p_data == cn).all()
    mci_bool = (p_data == mci).all()
    ad_bool = 3 in p_data 
    
    if cn_bool: 
        classif = np.vstack((classif, [ID, 1]))
    elif ad_bool: 
        classif = np.vstack((classif, [ID, 3]))
    else:
        classif = np.vstack((classif, [ID, 2]))

np.savetxt(CLASSIF_CSV_DIR, classif[1:,:], delimiter = ',')
