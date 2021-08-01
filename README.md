# Visualizing COVID Hospitalizations and Fatalities based on Vaccination Rate

The purpose of this project is two fold:
1. Predict the hospitalizations and fatalities due to COVID-19 for the next four weeks
    - Providing insight for healthcare facilities, vaccination clinics
2. Provide a sliding-scale to alter vaccination rates, visualizing the results
    - Visualize what a higher or lower vaccination rate could have led to in terms of hospitalizations and fatalities

![pexels-artem-podrez-5878514.jpg](https://github.com/ismizu/Time_Series_Modeling_Covid_Vax/blob/main/images/pexels-artem-podrez-5878514.jpg)

##### Image by [Artem Podrez](https://www.pexels.com/@artempodrez) from [Pexels](https://www.pexels.com/)

## Repository Structure

- The base, starting data can be found in the /data folder and was obtained from the [CDC](https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-Jurisdi/unsk-b7fc)

- Data updates are retrieved from [Reich Labs](https://reichlab.io/) and their [COVID-19 github repository](https://github.com/reichlab/covid19-forecast-hub#ensemble-model).

- All images used throughout the project can be found in the /images folder

- /pickled_data contains all updated dataframes as well as any saved checkpoints used throughout the project such as state-divided data, models, model parameters, and graphs

### Functions
This project largely relies upon the placement of functions within loops. The loops run through each individual state, repeating the function for each one. These functions can be found within the functions.py file.

### Project Overview
An overview of the project can be found at [this slide deck](https://docs.google.com/presentation/d/1z3zzTOvnFKVS_X-35wsUMyr73-Xmh90ihaLCswBJGPs/edit?usp=sharing).

### Contributor
Project by [Isana Mizuma](https://github.com/ismizu)

# Maintaining Data

One of the top items that this project addresses is updateability. The COVID-19 pandemic is an ongoing event. As such, a static prediction made at the time of project completion would soon lose efficacy.

With this in mind, the individual pieces that amalgamate into the final product were all made to run with minimal user input. From data cleaning, to preparing models, to creating each figure; the individual pieces were designed to process each state's outcome through the use of scripts. Thus, when the data is updated, each piece can rapidly be re-run to create the final product.

This updated dataframe is later assigned to the variable "vax_df_clean" for use in the full data cleaning function.

Below, the following data is retrieved from Reich Lab at UMass-Amherst.
- COVID-19 hospitalizations and fatalities
- Outliers identified by Reich Lab to help the models identify them
> Data is obtained outside of the function as the function is intended to run through a loop and separate data by state. Placing the calls within the function would cause it to re-read the data every time, considerably increasing execution time.

A loop is run to allow the data to be separated/updated on a per-state basis.

![data_clean_loop.png](https://github.com/ismizu/Time_Series_Modeling_Covid_Vax/blob/main/images/data_clean_loop.png)

![data_per_state_example.png](https://github.com/ismizu/Time_Series_Modeling_Covid_Vax/blob/main/images/data_per_state_example.png)

# Modeling

Following the creation of state-dependent dataframes, a function is used to create the models. The steps are as follows:
1. Retrieve state-dependent data
2. Run future vaccination predictions
3. Create models for hospitalization/fatality predictions
4. Save and export the models
>Vaccination numbers are used as exogenous variables to predict hospitalizations and fatalities. As such, their future values are needed in order to make future predictions on hospitalizations and fatalities. Therefore, an additional machine learning model is required to create those predictions.

With the created models, visualizations are then created.

The visualizations allow for alterations of vaccinations rates and show the potential changes in hospitalizations and fatalities should the vaccination rate reach, or fall, to such a level.

![readme_nj_example.png](https://github.com/ismizu/Time_Series_Modeling_Covid_Vax/blob/main/images/readme_nj_example.png)

The above is an example of the function's output.

With everything ready, the figure creation function is run for all states.

# Final Statements

Overall, the use of scripts was successful in creating a final product that could be repeated on a per-state level.

The project's first use was for COVID hospitalizations and fatalities predictions one months out. Unfortunately, in order to fulfill the project's second usage, visualizing the effect of vaccination rates, the accuracy of these predictions had to be negatively altered.

However, I viewed the trade-off in a positive light. The first usage, predictions of rates, is still fulfilled. Despite a lowered rate of accuracy, the ability to visualize what might happen, and what could have been, is something I view as more invaluable. Thus, I view the trade-off as a worthy loss.
