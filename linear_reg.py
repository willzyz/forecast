def predict_metric(se_df, metric, recommendation_week, min_value=0., **kwargs):
17a83f0a52b1 · D1438409	
    """Predict a metric using a mixed effects linear model.

    Currently just the historical average.

    Parameters
    ----------
    se_df : pandas.DataFrame
        supply efficiency model training data
    metric : str
        column name of the metric to forecast
9b50c5aedf52 · D1691991	
    recommendation_week : str
17a83f0a52b1 · D1438409	
        %Y-%m-%d for the Monday of the week to forecast for
    min_value : float
        minimum value guardrail
    kwargs : dict
        keyword arguments

    Returns
    -------
    dataframe of predictions

    """
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM

        raw_df = se_df.copy()

9b50c5aedf52 · D1691991	
        raw_df['delta_weeks'] = (pd.to_datetime(raw_df['week_of']) - pd.to_datetime(recommendation_week)).dt.days / 7
17a83f0a52b1 · D1438409	

        def fit_func(group):
ea0c134be68e · D1540393	
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
dca42ed79c75 · D2335295	
                return group.groupby('block_dow')[metric].mean().reset_index()
ea0c134be68e · D1540393	

17a83f0a52b1 · D1438409	
        prediction_df = raw_df.groupby(['block_hod', 'zone']).apply(fit_func).reset_index()
    except ValueError:
        warnings.warn("when trying to forecast {}, mixed effects didn't work, using historical mean".format(metric))
1f2b4a695abe · D2340907	
        prediction_df = raw_df.groupby(['block_hod', 'zone', 'block_dow'])[metric].mean().reset_index()
17a83f0a52b1 · D1438409	

dca42ed79c75 · D2335295	
    check_df_shape_validity(prediction_df, '{} prediction df'.format(metric))

17a83f0a52b1 · D1438409	
    if prediction_df[metric].min() < min_value:
        prediction_df[metric] = np.maximum(prediction_df[metric], min_value)
        warnings.warn("{0} prediction generated value less than {1}".format(metric, min_value))
    return prediction_df
