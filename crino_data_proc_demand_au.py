import numpy as np, pandas as pd, cPickle as pkl 

M = pkl.load(open('crino_demand_au_data.p', 'r')) 

D = M['ddata_au'] 

D = D.sort_values(by=['rtp_forecast_demand_active_users_sf.tshour']) 

data = D['rtp_forecast_demand_active_users_sf.active_users'].as_matrix() 

print(data.shape) 
print(D['rtp_forecast_demand_active_users_sf.tshour'])





