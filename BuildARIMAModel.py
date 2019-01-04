#!/usr/bin/env python 
# coding: utf-8 

# ## 
# 
# Build ARIMA model with python package 

# In[1]:

import cPickle, matplotlib.pyplot as plt, numpy as np 
from statsmodels.tsa.arima_model import ARIMA 
from matplotlib import pyplot as plt 
from crino_utils import * 

D, M = load_data_build_4wa_baseline_supply() 

## define the orders of ARIMA model 
arima_order = (3, 1, 1) 

full_pred_list = [] 
full_target_list = [] 
full_base_pred_list = [] 

## start & end weeks in the year for evaluation 
start_wk = 15 
end_wk = 35 
start_hour_wk = 0 
n_hours_wk = 168 

model_cnt = 0 
for ih in range(start_hour_wk, n_hours_wk): 
    print('working on hour of week: ' + str(ih)) 
    ## choose data to be the 
    X = D[:, ih] 
    for next_wk in range(start_wk, end_wk + 1): 
        forecast_wk = next_wk + 1 
        train, tgt = X[0:next_wk], X[forecast_wk] # forecast two week away 
        history_0 = [x for x in train] 
        try:
            ## two-step forecast
            h_mean = np.mean(np.asarray(history_0)) 
            h_std = np.std(np.asarray(history_0))
            norm_history_0 = (history_0 - h_mean) / h_std 
            
            model_0 = ARIMA(norm_history_0, order=arima_order) 
            model_fit_0 = model_0.fit(disp=0) 
            output_0 = model_fit_0.forecast() 
            norm_pred_0 = output_0[0][0] 
            
            pred_0 = norm_pred_0 * h_std + h_mean 
            
            history_1 = history_0 + [pred_0] 
            
            h_mean = np.mean(np.asarray(history_1)) 
            h_std = np.std(np.asarray(history_1))
            norm_history_1 = (history_1 - h_mean) / h_std             
            
            model_1 = ARIMA(norm_history_1, order=arima_order) 
            model_fit_1 = model_1.fit(disp=0) 
            output_1 = model_fit_1.forecast() 
            
            norm_pred_1 = output_1[0][0]
            
            pred_1 = norm_pred_1 * h_std + h_mean
            
            model_cnt = model_cnt + 1
        except: 
            pred_1 = M[forecast_wk, ih] 
        if np.isnan(pred_1): 
            pred_1 = M[forecast_wk, ih] 
        
        fwa = M[forecast_wk, ih] 
        full_target_list.append(tgt) 
        full_pred_list.append(pred_1) 
        full_base_pred_list.append(fwa) 

base_error = compute_mape(np.asarray(full_target_list), np.asarray(full_base_pred_list)) 
print('Base 4-week-ave Test MAPE: %.3f' % base_error) 

arima_error = compute_mape(np.asarray(full_target_list), np.asarray(full_pred_list)) 
print('ARIMA Model Test MAPE: %.3f' % arima_error) 

print('percentage of arima predicted values in pred_list: ' + str(1.0 * model_cnt / ((end_wk + 1 - start_wk) * (n_hours_wk - start_hour_wk)))) 
print(np.sum(np.isnan(full_pred_list))) 
