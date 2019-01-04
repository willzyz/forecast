import cPickle, matplotlib.pyplot as plt, numpy as np 

def compute_mape(x, y):
    ## function to compute the Mean Average Percentage Error (MAPE)
    ## across two vectors x, y; x, y are asserted to be same-sized
    ## vertical vectors 
    
    assert(len(x.shape) < 2 or x.shape[1] <= 1) 
    assert(len(y.shape) < 2 or y.shape[1] <= 1) 
    assert(x.shape[0] == y.shape[0]) 
    
    return 1.0 / x.shape[0] * np.sum(np.abs(x - y) / np.abs(x)) 

def load_data_build_4wa_baseline_supply():
    ## function loads numpy matrix D of supply hours
    ## from .p pickle file and convolve with mean kernel
    ## to produce a 4 week average baseline
    
    ## return:
    ## D -- 42 x 168 matrix, where each row is hourly data for a week
    ##
    ## M -- 44 x 168 matrix, contains forecasts starting from week 5, till week 44
    ##
    
    D = cPickle.load(open('save_crino_supplyhour_data_cp.p', 'rb')) 
    
    M = np.zeros([D.shape[0] + 2, D.shape[1]]) 
    
    k = 0.25 * np.asarray([1.0, 1.0, 1.0, 1.0]) 
    
    for i in range(D.shape[1]): 
        d = D[:, i] 
        c = np.convolve(d, k, 'valid') 
        M[5:, i] = c 
    
    return D, M
