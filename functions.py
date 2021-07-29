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
    death_hosp = death_hosp.resample('W').mean()
    
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