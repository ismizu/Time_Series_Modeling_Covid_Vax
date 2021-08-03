import pandas as pd
import numpy as np
import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

import pickle
import requests


#---------- Data Upkeep Functions ----------#

def vax_data_update():
    
    '''
    Retrieve updated vax data
    Concat new updates to original dataframe
    Save new dataframe
    '''
    
    #Retrieve api data
    api_url = 'https://data.cdc.gov/resource/unsk-b7fc.json'
    results = requests.get(api_url).json()
    
    #Open pickled data
    infile = open('pickled_data/general_data/vax_df_clean.pickle','rb')
    vax_df_clean = pickle.load(infile)
    infile.close()
    
    #Append new data to list
    parsed_data = []
    for result in results:
        
        #Retrieve last date on current vax dataset
        last_date = str(vax_df_clean.sort_values('date', ascending = False).iloc[0]['date'])[:10]
        
        #Save data if date new, stop when false
        if result['date'][:10] > last_date:
            data = (result['date'][:10], result['location'],
                    result['administered_dose1_pop_pct'],
                    result['series_complete_pop_pct'])
            parsed_data.append(data)
        else:
            break
    
    #Convert list to dataframe
    new_data = pd.DataFrame(parsed_data,
                 columns = ['date', 'location',
                            'administered_dose1_pop_pct',
                            'series_complete_pop_pct'
                           ]
                )
    
    #Change dtypes
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data = new_data.astype({'administered_dose1_pop_pct': 'float',
                                'series_complete_pop_pct': 'float'})
    
    #Concat to old dataframe
    vax_df_clean = pd.concat([new_data, vax_df_clean])
    
    #Save results
    pickle_out = open('pickled_data/general_data/vax_df_clean.pickle','wb')
    pickle.dump(vax_df_clean, pickle_out)
    pickle_out.close()
    
    #Return new dataframe
    return vax_df_clean

def table_cleaner(state_abbreviated,
                  hospitalizations_df,
                  deaths_df,
                  hospitalizations_outliers,
                  death_outliers,
                  vax_df_clean):
    '''
    Clean and concat desired data into Prophet's format
    Returns deaths/hospitalizations w/vax data
    Returns outliers in Prophet's holiday format
    '''
    
    #Import pickled dictionary of state name/abbreviation
    load_states_dict = open('pickled_data/general_data/states.pickle','rb')
    states = pickle.load(load_states_dict)
    load_states_dict.close()
    
    #Set state name per abbreviation
    full_state = states[state_abbreviated]
    
    #----- Hopitalizations/Fatalities Data -----#
    
    #Retrieve state's hospitalization data
    hosp_df = hospitalizations_df[hospitalizations_df['location_name'] == full_state].sort_values('date')[['date', 'value']]
    #Reformat column names for Prophet
    hosp_df.rename(columns = {'value': 'hospitalizations',
                              'date': 'ds'},
                   inplace = True
                  )
    #Adjust to datetime & set index to date
    hosp_df['ds'] = pd.to_datetime(hosp_df['ds'])
    hosp_df.set_index(keys = 'ds', drop = True, inplace = True)
    
    #Retrieve state's fatality data
    death_df = deaths_df[deaths_df['location_name'] == full_state].sort_values('date')[['date', 'value']]
    #Reformat column names for Prophet
    death_df.rename(columns = {'value': 'deaths',
                               'date': 'ds'},
                   inplace = True
                  )
    #Adjust to datetime & set index to date
    death_df['ds'] = pd.to_datetime(death_df['ds'])
    death_df.set_index(keys = 'ds', drop = True, inplace = True)
    
    #Concat hospitalization/fatality dataframes
    death_hosp = pd.concat([hosp_df, death_df], join = 'outer', axis = 1)
    #Fill early hospital data with 0
    death_hosp['hospitalizations'] = death_hosp['hospitalizations'].fillna(0)
    #Resample as weekly data
    death_hosp = death_hosp.resample('W').sum()
    
    #----- Vax Data -----#
    
    #Retrieve state's vax data
    state_vax = vax_df_clean[vax_df_clean['location'] == state_abbreviated]
    #Reformat column names for Prophet 
    state_vax.rename(columns = {'date': 'ds'},
                    inplace = True
                   )
    #Adjust to datetime & set index to date
    state_vax['ds'] = pd.to_datetime(state_vax['ds'])
    state_vax.set_index(keys = 'ds', drop = True, inplace = True)
    #Resample as weekly data
    state_vax = state_vax.resample('W').mean()
    
    #Concat hospitalization/fatality and vax dataframes
    final_df = pd.concat([death_hosp, state_vax], join = 'outer', axis = 1)
    
    #Fill missing early data with 0's
    columns = ['administered_dose1_pop_pct', 'series_complete_pop_pct']
    final_df[columns] = final_df[columns].fillna(0)
    
    #Drop data missing due to update imbalance
    final_df.dropna(inplace = True)
    final_df.reset_index(inplace = True)
    
    #----- Outlier Data -----#
    
    #Retrieve state's hospitalization outliers
    hosp_out = list(set(hospitalizations_outliers[hospitalizations_outliers['location_abbreviation'] == state_abbreviated]['date']))
    #Format for Prophet
    hosp_out_df = pd.DataFrame({
        'holiday': 'hosp_outliers',
        'ds': pd.to_datetime(hosp_out)    
    })
    #Retrive state's fatality outliers
    deaths_out = list(set(death_outliers[death_outliers['location_abbreviation'] == state_abbreviated]['date']))
    #Format for Prophet
    deaths_out_df = pd.DataFrame({
        'holiday': 'deaths_outliers',
        'ds': pd.to_datetime(hosp_out)    
    })
    #Concat holidays
    outliers = pd.concat([hosp_out_df, deaths_out_df])
    
    
    #Return final_df, outliers
    pickle_final_df = open(f'pickled_data/state_data_pickled/{state_abbreviated}_final_df.pickle','wb')
    pickle.dump(final_df, pickle_final_df)
    pickle_final_df.close()

    pickle_holidays = open(f'pickled_data/state_data_pickled/{state_abbreviated}_outliers.pickle','wb')
    pickle.dump(outliers, pickle_holidays)
    pickle_holidays.close()


#---------- Modeling Functions ----------#

def make_model(state_abbreviation):
    
    '''
    Run predictions for vax rates
    Use vax predictions for hospitalization/fatality models
    '''
    
    #----- Initial Data Load -----#
    
    #State separated data
    load_state_df = open(f'pickled_data/state_data_pickled/{state_abbreviation}_final_df.pickle','rb')
    the_state_df = pickle.load(load_state_df)
    load_state_df.close()
    
    #State outliers
    load_state_outliers = open(f'pickled_data/state_data_pickled/{state_abbreviation}_outliers.pickle','rb')
    state_outliers = pickle.load(load_state_outliers)
    load_state_outliers.close()
    
    #Vax grid searched parameters
    load_dose_one_param = open(f'pickled_data/model_params/{state_abbreviation}_dose_one_param.pickle','rb')
    dose_one_param = pickle.load(load_dose_one_param)
    load_dose_one_param.close()
    load_series_complete_param = open(f'pickled_data/model_params/{state_abbreviation}_series_complete_param.pickle','rb')
    series_complete_param = pickle.load(load_series_complete_param)
    load_series_complete_param.close()
    
    #Separate Regressors to run predictions on
    #Rename to Prophet naming conventions
    dose_one_df = the_state_df[['ds', 'administered_dose1_pop_pct']].rename(columns = {'administered_dose1_pop_pct': 'y'})
    series_complete_df = the_state_df[['ds', 'series_complete_pop_pct']].rename(columns = {'series_complete_pop_pct': 'y'})
    
    #----- Regressor Predictions -----#
    
    #Instantiate model for dose one vax
    dose_one_model = Prophet(n_changepoints = 30,
                             changepoint_prior_scale = dose_one_param)

    dose_one_model.fit(dose_one_df[dose_one_df['y'] > 0])
    
    #Make future dataframe and create predictions 1 month forward
    dose_one_future = dose_one_model.make_future_dataframe(periods = 4, freq = 'W')
    dose_one_forecast = dose_one_model.predict(dose_one_future)

    #Instantiate model for complete vax
    series_complete_model = Prophet(n_changepoints = 30,
                                    changepoint_prior_scale = series_complete_param)

    series_complete_model.fit(series_complete_df[series_complete_df['y'] > 0])
    
    #Make future dataframe and create predictions 1 month forward
    series_complete_future = series_complete_model.make_future_dataframe(periods = 4, freq = 'W')
    series_complete_forecast = series_complete_model.predict(series_complete_future)

    #Concat predictions
    vax_forecast = pd.concat(
        [dose_one_forecast[['ds', 'yhat']][-4:].\
         set_index(keys = 'ds', drop = True).\
         rename(columns = {'yhat': 'administered_dose1_pop_pct'}),

         series_complete_forecast[['ds', 'yhat']][-4:].\
         set_index(keys = 'ds', drop = True).\
         rename(columns = {'yhat': 'series_complete_pop_pct'})],
        axis = 1
    ).reset_index()

    #Set predictions index to follow full dataframe
    start_index = the_state_df.tail(1).index.values[0] + 1
    vax_forecast.index = pd.RangeIndex(start_index, start_index + 4)
    
    #Concat predictions to end of full dataframe
    the_state_df = pd.concat([the_state_df, vax_forecast])
    
    #Export update dataframe
    pickled_new_state_df = open(f'pickled_data/state_vax_pred_pickled/{state_abbreviation}_vax_pred_df.pickle','wb')
    pickle.dump(the_state_df, pickled_new_state_df)
    pickled_new_state_df.close()
    
    #----- Hospitalization/Fatality Models -----#
    
    #Load in grid searched parameters
    load_deaths_params = open(f'pickled_data/model_params/{state_abbreviation}_deaths_params.pickle','rb')
    deaths_params = pickle.load(load_deaths_params)
    load_deaths_params.close()

    load_hosp_params = open(f'pickled_data/model_params/{state_abbreviation}_hosp_params.pickle','rb')
    hosp_params = pickle.load(load_hosp_params)
    load_hosp_params.close()
    
    #Separate targets
    #Rename to Prophet naming conventions
    state_deaths = the_state_df.drop(columns = 'hospitalizations').rename(columns = {'deaths': 'y'})
    state_hospitalizations = the_state_df.drop(columns = 'deaths').rename(columns = {'hospitalizations': 'y'})
    
    #Covid Fatalities Model
    deaths_model = Prophet(n_changepoints = 30,
                           changepoint_range = deaths_params['changepoint_range'],
                           changepoint_prior_scale = deaths_params['changepoint_prior_scale'],
                           holidays = state_outliers[state_outliers['holiday'] == 'deaths_outliers'])
    
    #Add regressors w/ their vax predictions
    deaths_model.add_regressor('administered_dose1_pop_pct')
    deaths_model.add_regressor('series_complete_pop_pct')
    
    #Fit model on current data
    deaths_model.fit(state_deaths[:-4])
    
    #Covid Hospitalizations Model
    hosp_model = Prophet(n_changepoints = 30,
                         changepoint_range = hosp_params['changepoint_range'],
                         changepoint_prior_scale = hosp_params['changepoint_prior_scale'],
                         holidays = state_outliers[state_outliers['holiday'] == 'hosp_outliers'])
    
    #Add regressors w/ their vax predictions
    hosp_model.add_regressor('administered_dose1_pop_pct')
    hosp_model.add_regressor('series_complete_pop_pct')
    
    #Fit model on current data
    hosp_model.fit(state_hospitalizations[:-4])
    
    #----- Export models -----#
    
    #Vax models for metrics
    pickle_dose_one_model = open(f'pickled_data/models_pickled/{state_abbreviation}_dose_one_model.pickle','wb')
    pickle.dump(dose_one_model, pickle_dose_one_model)
    pickle_dose_one_model.close()

    pickle_series_complete_model = open(f'pickled_data/models_pickled/{state_abbreviation}_series_complete_model.pickle','wb')
    pickle.dump(series_complete_model, pickle_series_complete_model)
    pickle_series_complete_model.close() 
    
    #Fatality/Hospitaliztion models for later use
    pickle_deaths_model = open(f'pickled_data/models_pickled/{state_abbreviation}_deaths_model.pickle','wb')
    pickle.dump(deaths_model, pickle_deaths_model)
    pickle_deaths_model.close()

    pickle_hosp_model = open(f'pickled_data/models_pickled/{state_abbreviation}_hosp_model.pickle','wb')
    pickle.dump(hosp_model, pickle_hosp_model)
    pickle_hosp_model.close()

def visualization_predictions(state_abbreviation, multiplier):
    
    '''
    Create predictions based on vax multiplier
    '''
    
    #----- Data Load In ----- #
    load_deaths_model = open(f'pickled_data/models_pickled/{state_abbreviation}_deaths_model.pickle','rb')
    deaths_model = pickle.load(load_deaths_model)
    load_deaths_model.close()
    
    load_hosp_model = open(f'pickled_data/models_pickled/{state_abbreviation}_hosp_model.pickle','rb')
    hosp_model = pickle.load(load_hosp_model)
    load_hosp_model.close()
    
    load_vax_pred = open(f'pickled_data/state_vax_pred_pickled/{state_abbreviation}_vax_pred_df.pickle','rb')
    vax_pred_df = pickle.load(load_vax_pred)
    load_vax_pred.close()
    
    #Separate deaths/hospitalizations and rename columns
    state_deaths = vax_pred_df.drop(columns = 'hospitalizations').rename(columns = {'deaths': 'y'})
    state_hospitalizations = vax_pred_df.drop(columns = 'deaths').rename(columns = {'hospitalizations': 'y'})
    
    #----- Fatality Predictions -----#
    
    #Create future dataframe for predictions
    deaths_future = deaths_model.make_future_dataframe(periods = 4, freq = 'W')
    
    #Add regressor columns to future dataframe
    columns = ['administered_dose1_pop_pct', 'series_complete_pop_pct']
    #Multiplier to allow value alterations
    deaths_future[columns] = state_deaths[columns] * multiplier
    #Cap max vaccination rate at 100%
    deaths_future[columns[0]].clip(upper = 100., inplace = True)
    deaths_future[columns[1]].clip(upper = 100., inplace = True)
    
    #Create predictions, cap lower limit as 0
    deaths_forecast = deaths_model.predict(deaths_future)
    deaths_forecast['yhat'].clip(lower = 0, inplace = True)
   
    #----- Hospitalizations Predictions -----#

    #Create future dataframe for predictions
    hosp_future = hosp_model.make_future_dataframe(periods = 4, freq = 'W')
    
    #Add regressor columns to future dataframe
    #Multiplier to allow value alterations
    hosp_future[columns] = state_deaths[columns] * multiplier
    #Cap max vaccination rate at 100%
    hosp_future[columns[0]].clip(upper = 100., inplace = True)
    hosp_future[columns[1]].clip(upper = 100., inplace = True)
    
    #Create predictions, cap lower limit as 0
    hosp_forecast = hosp_model.predict(hosp_future) 
    hosp_forecast['yhat'].clip(lower = 0, inplace = True)        
    
    #Save base forecasts
    if multiplier == 1.:
        pickle_deaths_forecast = open(f'pickled_data/forecasts/{state_abbreviation}_deaths_forecast.pickle','wb')
        pickle.dump(deaths_forecast, pickle_deaths_forecast)
        pickle_deaths_forecast.close()

        pickle_hosp_forecast = open(f'pickled_data/forecasts/{state_abbreviation}_hosp_forecast.pickle','wb')
        pickle.dump(hosp_forecast, pickle_hosp_forecast)
        pickle_hosp_forecast.close()
    
    return deaths_forecast[['ds', 'yhat']], hosp_forecast[['ds', 'yhat']], deaths_future


#---------- Parameter Tuning Functions ----------#

def score_check(state_abbreviation, save_df):
    
    '''
    Run a cross validation for each model
    User indicates dataframe to save results to
    '''
    
    #----- Initial Data Load -----#
    load_dose_one_model = open(f'pickled_data/models_pickled/{state_abbreviation}_dose_one_model.pickle','rb')
    dose_one_model = pickle.load(load_dose_one_model)
    load_dose_one_model.close()

    load_series_complete_model = open(f'pickled_data/models_pickled/{state_abbreviation}_series_complete_model.pickle','rb')
    series_complete_model = pickle.load(load_series_complete_model)
    load_series_complete_model.close()
    
    load_deaths_model = open(f'pickled_data/models_pickled/{state_abbreviation}_deaths_model.pickle','rb')
    deaths_model = pickle.load(load_deaths_model)
    load_deaths_model.close()

    load_hosp_model = open(f'pickled_data/models_pickled/{state_abbreviation}_hosp_model.pickle','rb')
    hosp_model = pickle.load(load_hosp_model)
    load_hosp_model.close()
    
    #----- Vax Cross Validation -----#
    dose_one_cv = cross_validation(dose_one_model,
                                   horizon = '4 W',
                                   disable_tqdm = True)
    
    dose_one_mape = performance_metrics(dose_one_cv[dose_one_cv['y'] > 0 ]).iloc[-1]['mape']
    
    
    series_complete_cv = cross_validation(series_complete_model,
                                          horizon = '4 W',
                                          disable_tqdm = True)
    
    series_complete_mape = performance_metrics(series_complete_cv[series_complete_cv['y'] > 0 ]).iloc[-1]['mape']
    
    #----- Hospitalization/Fatality Cross Validation -----#
    deaths_model_cv = cross_validation(deaths_model,
                                       horizon = '4 W',
                                       period = '12 W',
                                       disable_tqdm = True)
    
    deaths_model_mape = performance_metrics(deaths_model_cv[deaths_model_cv['y'] > 0 ]).iloc[-1]['mape']
    
    
    hosp_model_cv = cross_validation(hosp_model,
                                     horizon = '4 W',
                                     period = '12 W',
                                     disable_tqdm = True)
    
    hosp_model_mape = performance_metrics(hosp_model_cv[hosp_model_cv['y'] > 0 ]).iloc[-1]['mape']
    
    #Return append to dataframe
    return save_df.append({'state': state_abbreviation,
                           'dose_one_score': round(dose_one_mape, 2),
                           'series_complete_score': round(series_complete_mape, 2),
                           'fatality_score': round(deaths_model_mape, 2),
                           'hosp_score': round(hosp_model_mape, 2)
                          },
                          ignore_index = True
                         )

def vax_params(state_abbreviation):
    
    '''
    Run a grid search to retrieve vax models' best settings
    '''
    
    #----- Initial Data Load -----#
    
    load_state_df = open(f'pickled_data/state_data_pickled/{state_abbreviation}_final_df.pickle','rb')
    the_state_df = pickle.load(load_state_df)
    load_state_df.close()

    load_state_outliers = open(f'pickled_data/state_data_pickled/{state_abbreviation}_outliers.pickle','rb')
    state_outliers = pickle.load(load_state_outliers)
    load_state_outliers.close()
    
    #Separate Regressors to run predictions on
    #Rename to Prophet naming conventions
    dose_one_df = the_state_df[['ds', 'administered_dose1_pop_pct']].rename(columns = {'administered_dose1_pop_pct': 'y'})
    series_complete_df = the_state_df[['ds', 'series_complete_pop_pct']].rename(columns = {'series_complete_pop_pct': 'y'})
    
    
    #----- Grid Searches -----#
    
    #Create parameters
    dose_one_param_grid = {  
        'changepoint_prior_scale': [.1, .5, 1, 3],
        'n_changepoints': [30]
    }
    dose_one_all_params = [dict(zip(dose_one_param_grid.keys(), v)) for v in itertools.product(*dose_one_param_grid.values())]
    dose_one_rmses = []
    
    #Check performance for all iterations of parameters
    for params in dose_one_all_params:
        dose_one_m = Prophet(**params).fit(dose_one_df[dose_one_df['y'] > 0])
        dose_one_df_cv = cross_validation(dose_one_m, horizon='4 W', disable_tqdm = True)
        dose_one_df_p = performance_metrics(dose_one_df_cv, rolling_window = 1)
        dose_one_rmses.append(dose_one_df_p['rmse'].values[0])
    
    #Retrieve best parameters
    dose_one_best_params = dose_one_all_params[np.argmin(dose_one_rmses)]
    
    #Create parameters
    series_complete_param_grid = {  
        'changepoint_prior_scale': [.1, .5, 1, 3],
        'n_changepoints': [30]
    }
    series_complete_all_params = [dict(zip(series_complete_param_grid.keys(), v)) for v in itertools.product(*series_complete_param_grid.values())]
    series_complete_rmses = []
    
    #Check performance for all iterations of parameters
    for params in series_complete_all_params:
        series_complete_m = Prophet(**params).fit(series_complete_df[series_complete_df['y'] > 0])
        series_complete_df_cv = cross_validation(series_complete_m, horizon='4 W', disable_tqdm = True)
        series_complete_df_p = performance_metrics(series_complete_df_cv, rolling_window = 1)
        series_complete_rmses.append(series_complete_df_p['rmse'].values[0])
    
    #Retrieve best parameters
    series_complete_best_params = series_complete_all_params[np.argmin(series_complete_rmses)]
    
    #Save parameters
    pickle_dose_one_param = open(f'pickled_data/model_params/{state_abbreviation}_dose_one_param.pickle','wb')
    pickle.dump(dose_one_best_params['changepoint_prior_scale'], pickle_dose_one_param)
    pickle_dose_one_param.close()

    pickle_series_complete_param = open(f'pickled_data/model_params/{state_abbreviation}_series_complete_param.pickle','wb')
    pickle.dump(series_complete_best_params['changepoint_prior_scale'], pickle_series_complete_param)
    pickle_series_complete_param.close()


def main_params(state_abbreviation):
    
    '''
    Run a gridsearch to retrieve hospitalization/fatality models' best settings
    '''
    
    #----- Initial Data Load -----#
    
    load_state_df = open(f'pickled_data/state_data_pickled/{state_abbreviation}_final_df.pickle','rb')
    the_state_df = pickle.load(load_state_df)
    load_state_df.close()

    load_state_outliers = open(f'pickled_data/state_data_pickled/{state_abbreviation}_outliers.pickle','rb')
    state_outliers = pickle.load(load_state_outliers)
    load_state_outliers.close()
    
    #Separate targets
    #Rename to Prophet naming conventions
    state_deaths = the_state_df.drop(columns = 'hospitalizations').rename(columns = {'deaths': 'y'})
    state_hospitalizations = the_state_df.drop(columns = 'deaths').rename(columns = {'hospitalizations': 'y'})
    
    #Create parameters
    deaths_param_grid = {  
        'changepoint_prior_scale': [.1, .5, 1, 3],
        'changepoint_range': [.85, .90, .95, 1],
        'n_changepoints': [30],
        'holidays': [state_outliers[state_outliers['holiday'] == 'deaths_outliers']]
    }
    deaths_all_params = [dict(zip(deaths_param_grid.keys(), v)) for v in itertools.product(*deaths_param_grid.values())]
    deaths_rmses = []
    
    #Check performance for all iterations of parameters
    for params in deaths_all_params:
        deaths_m = Prophet(**params)
        
        deaths_m.add_regressor('administered_dose1_pop_pct')
        deaths_m.add_regressor('series_complete_pop_pct')
        
        deaths_m.fit(state_deaths)
        
        deaths_df_cv = cross_validation(deaths_m,
                                        horizon= '4 W',
                                        period = '12 W',
                                        disable_tqdm = True)
        deaths_df_p = performance_metrics(deaths_df_cv[deaths_df_cv['y'] > 0], rolling_window = 1)
        deaths_rmses.append(deaths_df_p['rmse'].values[0])
    
    #Retrieve best parameters
    deaths_best_params = deaths_all_params[np.argmin(deaths_rmses)]
    
    #Create parameters
    hosp_param_grid = {  
        'changepoint_prior_scale': [.1, .5, 1, 3, 4, 5],
        'changepoint_range': [.85, .90, .95, 1],
        'n_changepoints': [30],
        'holidays': [state_outliers[state_outliers['holiday'] == 'hosp_outliers']]
    }
    hosp_all_params = [dict(zip(hosp_param_grid.keys(), v)) for v in itertools.product(*hosp_param_grid.values())]
    hosp_rmses = []
    
    #Check performance for all iterations of parameters
    for params in hosp_all_params:
        hosp_m = Prophet(**params)
        
        hosp_m.add_regressor('administered_dose1_pop_pct')
        hosp_m.add_regressor('series_complete_pop_pct')
        
        hosp_m.fit(state_hospitalizations)
        
        hosp_df_cv = cross_validation(hosp_m,
                                      horizon= '4 W',
                                      period = '12 W',
                                      disable_tqdm = True)
        hosp_df_p = performance_metrics(hosp_df_cv[hosp_df_cv['y'] > 0], rolling_window = 1)
        hosp_rmses.append(hosp_df_p['rmse'].values[0])
    
    #Retrieve best parameters
    hosp_best_params = hosp_all_params[np.argmin(hosp_rmses)]
    
    #Save parameters
    pickle_deaths_params = open(f'pickled_data/model_params/{state_abbreviation}_deaths_params.pickle','wb')
    pickle.dump(deaths_best_params, pickle_deaths_params)
    pickle_deaths_params.close()
    
    pickle_hosp_params = open(f'pickled_data/model_params/{state_abbreviation}_hosp_params.pickle','wb')
    pickle.dump(hosp_best_params, pickle_hosp_params)
    pickle_hosp_params.close()


#---------- Visualizations Functions ----------#

def make_fig(state_abbreviation):
    
    '''
    Create/save figure
    '''
        
    # Create figure
    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes = True,
                        specs=[[{'type': 'scatter'}],
                               [{'type': 'scatter'}]
                              ],
                        row_heights = [10,
                                       20],
                        subplot_titles=('Vaccination Rate',
                                        'Weekly Hospitalizations & Fatalities')
                       )


    # Add traces, one for each slider step
    for step in np.arange(0, 2, 0.05):

        death_forecast, hosp_forecast, vax_df = visualization_predictions(state_abbreviation, step)

        fig.add_traces(
            [
            go.Scatter(
                visible = False,
                line = dict(color = '#7dfffc', width=6),
                name = 'Dose 1 Pop. %',
                x = vax_df['ds'],
                y = vax_df['administered_dose1_pop_pct']
            ),
            go.Scatter(
                visible = False,
                line = dict(color = '#7dff80', width=6),
                name = 'Series Complete Pop. %',
                x = vax_df['ds'],
                y = vax_df['series_complete_pop_pct']
            )
            ],
            rows = [1, 1], cols = [1, 1]
        )

        fig.add_traces(
            [
            go.Scatter(
                visible = False,
                line = dict(color = '#ff2828', width=6),
                name = 'Fatalities',
                x = death_forecast['ds'],
                y = death_forecast['yhat']
            ),
            go.Scatter(
                visible = False,
                line = dict(color = '#ff9428', width=6),
                name = 'Hospitalizations',
                x = hosp_forecast['ds'],
                y = hosp_forecast['yhat']
            )
            ],
            rows = [2, 2], cols = [1, 1]
        )

    # Make middle trace visible
    middle_trace = int(len(fig.data)/2)
    for i in range(middle_trace, middle_trace + 4):
        fig.data[i].visible = True

    # Create and add slider
    steps = []
    for i in range(0, len(fig.data), 8):
        step = dict(
            method = 'update',
            args = [{'visible': [False] * len(fig.data)},
                    {'title': f'{state_abbreviation} Vaccine Rate Multipler at: ' + str(i/80)}],
        )
        step['args'][0]['visible'][i] = True
        
        # Set multiple traces to visible, unless reaching end of traces
        try:
            step['args'][0]['visible'][i+3] = True
        except:
            continue
        try:
            step['args'][0]['visible'][i+2] = True
        except:
            continue
        try:
            step['args'][0]['visible'][i+1] = True
        except:
            continue
        steps.append(step)

    sliders = [dict(
        active = 10,
        currentvalue = {'prefix': 'Vaccination Rate Multiplier: '},
        pad = {'t': 50},
        steps = steps
    )]

    fig.update_layout(
        autosize = True,
        sliders = sliders,
        yaxis_range = [0, 100],
        hovermode = 'x'
    )
    
    #Save/export graph
    pickle_graph = open(f'pickled_data/graphs_pickled/{state_abbreviation}_graph.pickle', 'wb')
    pickle.dump(fig, pickle_graph)
    pickle_graph.close()


def show_fig(state_abbreviation):
    
    '''
    Read in figure and show
    '''
    
    load_graph = open(f'pickled_data/graphs_pickled/{state_abbreviation}_graph.pickle','rb')
    fig = pickle.load(load_graph)
    load_graph.close()

    fig.show()

def initial_fig(state_abbreviation):
    
    '''
    Read in initial figures and show
    '''
    
    load_graph = open(f'pickled_data/initial_graphs_pickled/{state_abbreviation}_graph.pickle','rb')
    fig = pickle.load(load_graph)
    load_graph.close()

    fig.show()

#---------- Manual Tuning Functions ----------#

def check_(state_abbreviation):
    load_deaths_params = open(f'pickled_data/model_params/{state_abbreviation}_deaths_params.pickle','rb')
    deaths_params = pickle.load(load_deaths_params)
    load_deaths_params.close()

    load_hosp_params = open(f'pickled_data/model_params/{state_abbreviation}_hosp_params.pickle','rb')
    hosp_params = pickle.load(load_hosp_params)
    load_hosp_params.close()
    return print(f'''{state_abbreviation} Deaths: Rnge: {deaths_params['changepoint_range']} Prior: {deaths_params['changepoint_prior_scale']}
{state_abbreviation} Hosp: Rnge: {hosp_params['changepoint_range']} Prior: {hosp_params['changepoint_prior_scale']}''')

def manual_tune(state_abbreviation,
                death_changepoint_range,
                death_changepoint_prior_scale,
                hosp_changepoint_range,
                hosp_changepoint_prior_scale):
    
    
    deaths_params = {'changepoint_range': None,
                     'changepoint_prior_scale': None}
    

    
    hosp_params = {'changepoint_range': None,
                   'changepoint_prior_scale': None}
    
    
    deaths_params['changepoint_range'] = death_changepoint_range
    deaths_params['changepoint_prior_scale'] = death_changepoint_prior_scale
    hosp_params['changepoint_range'] = hosp_changepoint_range
    hosp_params['changepoint_prior_scale'] = hosp_changepoint_prior_scale
    
    pickle_deaths_params = open(f'pickled_data/model_params/{state_abbreviation}_deaths_params.pickle','wb')
    pickle.dump(deaths_params, pickle_deaths_params)
    pickle_deaths_params.close()
    
    pickle_hosp_params = open(f'pickled_data/model_params/{state_abbreviation}_hosp_params.pickle','wb')
    pickle.dump(hosp_params, pickle_hosp_params)
    pickle_hosp_params.close()


