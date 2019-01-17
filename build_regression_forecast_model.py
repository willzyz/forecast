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

## back-log: code from Crino 

"""Predict a metric using a mixed effects linear model. 

Currently just the historical average.

Parameters 
---------- 
se_df : pandas.DataFrame 
    supply efficiency model training data 
metric : str 
    column name of the metric to forecast 
recommendation_week : str 
    %Y-%m-%d for the Monday of the week to forecast for 
min_value : float 
    minimum value guardrail 
kwargs : dict 
    keyword arguments 

Returns
-------
dataframe of predictions

"""
"""
def predict_metric(se_df, metric, recommendation_week, min_value=0., **kwargs):
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        raw_df = se_df.copy()

        raw_df['delta_weeks'] = (pd.to_datetime(raw_df['week_of']) - pd.to_datetime(recommendation_week)).dt.days / 7

        def fit_func(group):
            try:
                md = MixedLM.from_formula("{} ~ 1 + delta_weeks".format(metric),
                                          groups='block_dow',
                                          re_formula='1 + delta_weeks',
                                          data=group.fillna({metric: 0.})
                                          )

                mdf = md.fit()

                index = mdf.random_effects.keys()
                data = {
                    metric: (mdf.params['Intercept'] + [mdf.random_effects[i]['Intercept'] for i in index]),
                    'block_dow': index,
                }
                return pd.DataFrame(data).set_index('block_dow')

            except np.linalg.linalg.LinAlgError as err:
                logging.warning(err)
                return group.groupby('block_dow')[metric].mean().reset_index()

        prediction_df = raw_df.groupby(['block_hod', 'zone']).apply(fit_func).reset_index()
    except ValueError:
        warnings.warn("when trying to forecast {}, mixed effects didn't work, using historical mean".format(metric))
        prediction_df = raw_df.groupby(['block_hod', 'zone', 'block_dow'])[metric].mean().reset_index()

    check_df_shape_validity(prediction_df, '{} prediction df'.format(metric))

    if prediction_df[metric].min() < min_value:
        prediction_df[metric] = np.maximum(prediction_df[metric], min_value)
        warnings.warn("{0} prediction generated value less than {1}".format(metric, min_value))
    return prediction_df
""" 
