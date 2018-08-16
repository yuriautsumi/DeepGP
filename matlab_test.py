#from call_pgp import * 
#
#import numpy as np 
#
#x_a = np.array([[1.5, 1.7, 1.8], [2.7,2.4,2.9],[3.6,3.7,3.9]])
#y_a = np.array([[5.2],[3.4],[1.7]])
#x_s = np.array([[1.3,1.6,1.3],[5.3,5.2,5.7],[3.4,3.6,3.3],[2.4,2.3,2.6],[4.3,4.3,4.6]])
#y_s = np.array([[5.4],[0.2],[1.9],[3.8],[2.9]])
#xtest = np.array([[1.5, 1.7, 1.8], [2.7,2.4,2.9],[3.6,3.7,3.9], [5.2,5.3,5.6]])
#m_s = np.array([[5.0],[3.5],[1.6],[0.4]])
#s_s = np.array([[0.2],[0.15],[0.3],[0.23]])
#
#ls = 75.2
#mul = 3 
#var = 36.1
#sn2 = 23.7
#
#for i in range(20): 
#    m_adapt, s_adapt = call_adapt(x_a, y_a, x_s, y_s, xtest, m_s, s_s, np.array([ls]), [mul], np.array([var]), np.array([sn2]))



import matlab.engine
eng = matlab.engine.start_matlab()
ret = eng.triarea(1.0,5.0)
print(ret)
#

#import time 
#
#start1 = time.time()
#import matlab.engine 
#eng = matlab.engine.start_matlab()
#end1 = time.time()
#
#print('time to start engine', end1-start1)
#
#start2 = time.time()
#for i in range(10): 
#    eng.compute_final_results(0, nargout=0)
#end2 = time.time()
#
#print('time to run script', end2-start2)