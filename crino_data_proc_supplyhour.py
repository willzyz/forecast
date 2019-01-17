#!/usr/bin/env python
# coding: utf-8

# # Week-away Forecasting for Supply and Demand 
# 
# ## Objective 
# 
# Build capabilities for forecasting week-away sequential data and time-series. The forecast should accurate compared to benchmarks such as 30-minute average on historical data (date-arrange TBD) and warn the system if it encounters anomalies. 
# 
# ## Key Results 
# 
# - Measure the mean absolute percentage error (MAPE) 
# - Forecast with several models and observe the difference compared to 30min average with these models: 
#     --logistic regression 
#     --neural nets 
#     --KDE 
# - Deal with undampened demand with session and eye-ball modeling, observe difference across dampened/undampened 
# - Create dashboards that can visualize forecasts 
# 

# ## Problem Breakdown 
# 
# Brainstorm 
# - Support different geo-fences 
# - BAF/undampened modeling, take in as feature to predict the result 
#         - we can predict BAF 
#         - we want all forecasts to be undampened, meaning assume there is no BAF 
# - Session and eye-ball modeling 
# - Fine-grained Hexagonal data 
# - Anomally detection 
# 

# ## Todo Steps 
# 
# - load and clean up data 
# - build first version of model 
# - validate and test versus baseline 
# 

# In[1]:


### --- data loading processing cleaning etc --- 

import numpy as np, pandas as pd, datetime, matplotlib.pyplot as plt, cPickle 

# load up all files into panda frames 

filenames = [#'SF_supply_eff_rel_super_data_2018-01-01_to_2018-01-08.csv', \
             #'SF_supply_eff_rel_super_data_2018-01-08_to_2018-01-15.csv', \
             'SF_supply_eff_rel_super_data_2018-01-22_to_2018-01-29.csv', \
             'SF_supply_eff_rel_super_data_2018-01-29_to_2018-02-05.csv', \
             'SF_supply_eff_rel_super_data_2018-02-05_to_2018-02-12.csv', \
             'SF_supply_eff_rel_super_data_2018-02-12_to_2018-02-19.csv', \
             'SF_supply_eff_rel_super_data_2018-02-19_to_2018-02-26.csv', \
             'SF_supply_eff_rel_super_data_2018-02-26_to_2018-03-05.csv', \
             'SF_supply_eff_rel_super_data_2018-03-05_to_2018-03-12.csv', \
             'SF_supply_eff_rel_super_data_2018-03-12_to_2018-03-19.csv', \
             'SF_supply_eff_rel_super_data_2018-03-19_to_2018-03-26.csv', \
             'SF_supply_eff_rel_super_data_2018-03-26_to_2018-04-02.csv', \
             'SF_supply_eff_rel_super_data_2018-04-02_to_2018-04-09.csv', \
             'SF_supply_eff_rel_super_data_2018-04-09_to_2018-04-16.csv', \
             'SF_supply_eff_rel_super_data_2018-04-16_to_2018-04-23.csv', \
             'SF_supply_eff_rel_super_data_2018-04-23_to_2018-04-30.csv', \
             'SF_supply_eff_rel_super_data_2018-04-30_to_2018-05-07.csv', \
             'SF_supply_eff_rel_super_data_2018-05-07_to_2018-05-14.csv', \
             'SF_supply_eff_rel_super_data_2018-05-14_to_2018-05-21.csv', \
             'SF_supply_eff_rel_super_data_2018-05-21_to_2018-05-28.csv', \
             'SF_supply_eff_rel_super_data_2018-05-28_to_2018-06-04.csv', \
             'SF_supply_eff_rel_super_data_2018-06-04_to_2018-06-11.csv', \
             'SF_supply_eff_rel_super_data_2018-06-11_to_2018-06-18.csv', \
             'SF_supply_eff_rel_super_data_2018-06-18_to_2018-06-25.csv', \
             'SF_supply_eff_rel_super_data_2018-06-25_to_2018-07-02.csv', \
             'SF_supply_eff_rel_super_data_2018-07-02_to_2018-07-09.csv', \
             'SF_supply_eff_rel_super_data_2018-07-09_to_2018-07-16.csv', \
             'SF_supply_eff_rel_super_data_2018-07-16_to_2018-07-23.csv', \
             'SF_supply_eff_rel_super_data_2018-07-23_to_2018-07-30.csv', \
             'SF_supply_eff_rel_super_data_2018-07-30_to_2018-08-06.csv', \
             'SF_supply_eff_rel_super_data_2018-08-06_to_2018-08-13.csv', \
             'SF_supply_eff_rel_super_data_2018-08-13_to_2018-08-20.csv', \
             'SF_supply_eff_rel_super_data_2018-08-20_to_2018-08-27.csv', \
             'SF_supply_eff_rel_super_data_2018-08-27_to_2018-09-03.csv', \
             'SF_supply_eff_rel_super_data_2018-09-03_to_2018-09-10.csv', \
             'SF_supply_eff_rel_super_data_2018-09-10_to_2018-09-17.csv', \
             'SF_supply_eff_rel_super_data_2018-09-17_to_2018-09-24.csv', \
             'SF_supply_eff_rel_super_data_2018-09-24_to_2018-10-01.csv', \
             'SF_supply_eff_rel_super_data_2018-10-01_to_2018-10-08.csv', \
             'SF_supply_eff_rel_super_data_2018-10-08_to_2018-10-15.csv', \
             'SF_supply_eff_rel_super_data_2018-10-15_to_2018-10-22.csv', \
             'SF_supply_eff_rel_super_data_2018-10-22_to_2018-10-29.csv', \
             'SF_supply_eff_rel_super_data_2018-10-29_to_2018-11-05.csv', \
             'SF_supply_eff_rel_super_data_2018-11-05_to_2018-11-12.csv']

## Union all supply-hour data 
data = pd.read_csv(filenames[0]) 
print(data.loc[0]) 
for i in range(len(filenames) - 1): 
    print('processing file: ' + str(i + 1)) 
    filename = filenames[i + 1] 
    if i == 0: #i > 2: 
        d = pd.read_csv(filename) 
    else: 
        d = pd.read_csv(filename, skiprows=[0]) 
    print(d.loc[0]) 
    data = pd.concat([data, d], axis=0) 
print(len(data.columns)) 

## group by hour add the supply hours, and filter dedicated drivers 
cols = ['Index', 'Time', 'OrdinalTime', 'SupplyHours'] 
dtstr = '2018-01-22 03:00:00' 
ts_data = pd.DataFrame(columns=cols) 

count = 0 
while dtstr != '2018-11-12 03:00:00': 
    if count % 100 == 0: 
        print(count)
    yr = int(dtstr.split('-')[0])
    mth = int(dtstr.split('-')[1])
    dy = int(dtstr.split('-')[2].split(' ')[0])
    
    hr = int(dtstr.split(':')[0].split(' ')[1])
    min = int(dtstr.split(':')[1])
    sec = int(dtstr.split(':')[2])
    
    x = datetime.datetime(yr, mth, dy, hr, min, sec)     
    y = x + datetime.timedelta(hours=1) 
    dtstr = y.isoformat(' ') 
    
    daydata = data.loc[data['hour_of']==dtstr] 
    
    SH = daydata['supply_hours'] 
    sh = np.sum(SH.as_matrix()[~np.isnan(SH.as_matrix())]) 
    
    ## aggregate supply days 
    d = pd.DataFrame(np.asarray([[count, dtstr, y, sh]]), columns=cols, index = [count]) 
    count = count + 1 
    ts_data = ts_data.append(d) 

plt.plot(ts_data['Index'].as_matrix(), ts_data['SupplyHours'].as_matrix()) 
plt.show() 
#print(ts_data.loc[ts_data['Index'] == 338]) 

print('Pringing out Weekly patterns:') 
for i in range(10): 
    print('Week: ' + str(i + 1))
    plt.figure 
    plt.plot(ts_data.loc[i * 168:i * 168 + 168 - 1]['Index'].as_matrix(), ts_data.loc[i * 168:i * 168 + 168 - 1]['SupplyHours'].as_matrix()) 
    plt.grid() 
    plt.show() 

## process daily patterns 
a = 7055.0 / 24 
D = ts_data.loc[:]['SupplyHours'].as_matrix() 
D = np.reshape(D, [-1, 168]) 
plt.imshow(D.astype(np.float32))
plt.show() 

cPickle.dump(D, open('save_crino_supplyhour_data_cp.p', 'wb')) 
