## Build baseline using linear regression 
## benchmark against arima models 

import cPickle, matplotlib.pyplot as plt, numpy as np, pandas as pd 
from statsmodels.regression.mixed_linear_model import MixedLM 

def fit_func(rdf):
    md = MixedLM.from_formula("supply_hours ~ 1 + delta_weeks", 
                              groups='block_dow', 
                              re_formula='1 + delta_weeks', 
                              data=rdf.fillna({'supply_hours': 0.}) 
                             ) 
    
    mdf = md.fit() 
    index = mdf.random_effects.keys() 
    
    data = { 
        'supply_hours': (mdf.params['Intercept'] + [mdf.random_effects[i]['Intercept'] for i in index]), 
        'block_dow' : index 
    } 
    
    result = pd.DataFrame(data).set_index('block_dow') 
    return result 

D = cPickle.load(open('save_crino_supplyhour_data_cp.p', 'rb')) 

OrigData = D; 

M = np.zeros([D.shape[0] + 2, D.shape[1]]) 

k = 0.25 * np.asarray([1.0, 1.0, 1.0, 1.0]) 

for i in range(D.shape[1]): 
    d = D[:, i] 
    c = np.convolve(d, k, 'valid') 
    M[5:, i] = c 

D = np.reshape(D, [D.size, ]) 

block_how_list = [i % (24 * 7) for i in range(D.size)] 

d = {'week_index' : [ int(np.floor(i / (24 * 7))) for i in range(D.size)], 
     'supply_hours' : D, 
     'block_how' : block_how_list, 
     'block_dow' : [int(np.floor(i / 24)) for i in block_how_list], 
     'block_hod' : [i % 24 for i in block_how_list] 
     } 

DF = pd.DataFrame(data = d) 
num_weeks_train_regressor = 5 

for next_wk in range(10, 15): 
    df = DF.loc[np.logical_and(DF['week_index'] < next_wk, DF['week_index'] >= next_wk - num_weeks_train_regressor)] 
    rdf = df.copy() 
    rdf['delta_weeks'] = rdf['week_index'] - next_wk 
    predict_df = rdf.groupby(['block_hod']).apply(fit_func).reset_index()
    sorted_pdf = predict_df.sort_values(['block_dow', 'block_hod'], ascending=[True, True])
    
    plt.figure(); plt.plot(sorted_pdf['supply_hours'].as_matrix(), 'm'); 
    plt.plot(M[next_wk + 1, :], 'c'); plt.grid(True); 
    plt.plot(OrigData[next_wk + 1, :], 'b'); plt.show() 
