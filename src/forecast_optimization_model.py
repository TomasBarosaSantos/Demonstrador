import numpy as np
import pandas as pd
import pyomo
from pyomo.environ import *
import plotly.express as px
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import dash
import datetime
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import pickle
from sklearn import metrics
from datetime import timedelta
import base64

date_format ='%d/%m/%Y %H:%M'

#Import the CVS files
df_1 = pd.read_csv('policia_municipal_24.csv')
df_1['Date']= pd.to_datetime(df_1['Date'], format=date_format)
df_1 = df_1.set_index('Date', drop = True)
df_1= df_1.resample("h").mean() #Resampling to hourly and doing the mean

df_2 = pd.read_csv('biblioteca_municipal_de_belem_24.csv')
df_2['Date']= pd.to_datetime(df_2['Date'], format=date_format)
df_2 = df_2.set_index('Date', drop = True)
df_2= df_2.resample("h").mean() #Resampling to hourly and doing the mean

df_3 = pd.read_csv('assembleia_municipal_24.csv')
df_3['Date']= pd.to_datetime(df_3['Date'], format=date_format)
df_3 = df_3.set_index('Date', drop = True)
df_3= df_3.resample("h").mean() #Resampling to hourly and doing the mean

df_consumption = pd.merge(df_1,df_2, on='Date')
df_consumption = pd.merge(df_consumption,df_3, on='Date')

#Import the CVS Price Forecast Input data
price_format = '%Y-%m-%d %H:%M:%S'
df_PriceForescast = pd.read_csv('__PriceForecast_InputData.csv')
df_PriceForescast['Date'] = pd.to_datetime(df_PriceForescast['Date'])#, format=price_format)
df_PriceForescast = df_PriceForescast.set_index('Date', drop = True)
df_PriceForescast = df_PriceForescast['2023-01-01':] # Forecast data from


# Import Prediction Model
with open('XGB_model_3.pkl','rb') as file:
    XGB_model_3 = pickle.load(file)


#Import the CSS file
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 38.7295,
	"longitude": -9.1649,
	"hourly": ["temperature_2m", "direct_normal_irradiance"],
	"start_date": "2023-01-01",
	"end_date": "2024-04-14"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_direct_normal_irradiance = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = False),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = False),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["Temperature"] = hourly_temperature_2m
hourly_data["Irradiance (W/m2)"] = hourly_direct_normal_irradiance

hourly_dataframe = pd.DataFrame(data = hourly_data)
hourly_dataframe = hourly_dataframe.set_index('date', drop = True)
df_Weather = hourly_dataframe
############################################################

##Merging consumpiton with Weather Data
df_Weather.index.names = ['Date']
df_data = pd.merge(df_consumption, df_Weather, on='Date')

#Definition of functions

def prediction(df, date_input):
        
        # Define forecast period: 72h ahead
        date_start = datetime.datetime.strptime(date_input, '%Y-%m-%d')
        date_start = date_start + timedelta(days=1)
        #date_start = pd.to_datetime(date_input) + pd.DateOffset(days=1)
        date_end = pd.to_datetime(date_input) + timedelta(days=3)
        date_start = date_start.strftime('%Y-%m-%d') #time2str
        date_end = date_end.strftime('%Y-%m-%d') #time2str 

        # Limit data only for specific period
        df_forecast = df.copy()
        df_forecast = df_forecast[date_start:date_end]

        # Define values of Y (target) and X (predictors)
        Z = df_forecast.values
        Y_real = Z[:,0]
        X_real = Z[:,1:]

        # Predicting electricity price values: guarantee they are positive and round to 2 decimal
        def pred_positive(Y_pred):
            Y_pred = np.maximum(Y_pred, 0)
            Y_pred = np.round(Y_pred, 2)
            return Y_pred

        # Run the model - obtain forecasted prices
        # Final model is Model_3 w/ trained data from 2021 to Q1-2024 (equilibrium beteween results and historical data)
        Y_pred = XGB_model_3.predict(X_real)
        Y_pred = pred_positive(Y_pred)

        # Compare prices: prediction vs real
        df_forecast = df_forecast.rename(columns={'PT Price': 'Y_real'})
        df_forecast['Y_pred'] = Y_pred
        df_forecast['Y_pred_inferior'] = Y_pred*0.9
        df_forecast['Y_pred_superior'] = Y_pred*1.1

        # Calculate metrics
        MAE=metrics.mean_absolute_error(Y_real,Y_pred)  
        MSE=metrics.mean_squared_error(Y_real,Y_pred)  
        RMSE=np.sqrt(metrics.mean_squared_error(Y_real,Y_pred))
        metrics_string = f'Forecasted Metrics: \nMAE: {MAE:.3f} || MSE: {MSE:.3f} || RMSE: {RMSE:.3f}.'

        return df_forecast, metrics_string

#PV production
def FE_PV(df,num_panels):

    Pot_Panels = 500
    coef_Pp = -0.36

    df['Irradiance (W/m2)'] = df['Irradiance (W/m2)'].astype('float64')
    df['PV Power'] = (df['Irradiance (W/m2)']/1000) * Pot_Panels * (1+ coef_Pp*(df['Temperature']-25)) * num_panels/1000

    return df

#Obtain results from non optimal solution
def results(df, value_drop):
    
    df['GRID'] = df[value_drop] - df['PV Power']
    df_1 = df[df['GRID'] <0 ]
    df = df[df['GRID'] >0 ]   
    values = sum(df_1['GRID'] * df_1['PV Sell']) 
    values = values + sum(df['GRID'] * df['PT Price (€/kWh)'])
    return values

#Optimization model
def optimization_model(df, building,size_batt):

    model = ConcreteModel()
    opt = SolverFactory('glpk')

    #Define the parameters for the optimization problem
    model.Period = RangeSet(0, len(df)-1)
    model.Price = Param(model.Period, initialize=list(df['PT Price (€/kWh)']), within=Any,doc='Spot Price')
    model.Demand = Param(model.Period, initialize=list(df[building]), doc='Demand')
    model.PV = Param(model.Period, initialize=list(df['PV Power']), doc='PV Power')
    model.SellPV = Param(model.Period,initialize=list(df['PV Sell']))

    #Battery Characteristics
    BATTERY_CAPACITY = size_batt
    INICIAL_CHARGE = BATTERY_CAPACITY * 0.5
    MAX_BATTERY_CAPACITY = BATTERY_CAPACITY * 0.8
    MIN_BATTERY_CAPACITY = BATTERY_CAPACITY * 0.1
    #C-Rate is 0.5, meaning that it takes 2 hours at max power rate to charge/discharge completly
    MAX_CHARGE = - size_batt * 0.5
    MAX_DISCHARGE = size_batt * 0.5

    #Decision Variables
    model.Grid = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='Energy from Grid')
    model.BATT = Var(model.Period,bounds=(MAX_CHARGE,MAX_DISCHARGE), domain=Reals, doc='Energy Exchanged with BATT')
    model.SOC = Var(model.Period,bounds=(MIN_BATTERY_CAPACITY, MAX_BATTERY_CAPACITY), domain= Reals, doc='State of Battery')
    model.PVEx = Var(model.Period, bounds=(0.0,None), domain=NonNegativeReals)

    #Objective Fuction
    #Minimize the costs of energy 
    def obj_rule(model):
        return sum((model.Grid[t] * model.Price[t]) - (model.PVEx[t] * model.SellPV[t]) for t in model.Period)
    model.obj = Objective(rule=obj_rule, sense=minimize)

    ##Constraints
    #
    model.C1 = ConstraintList()
    model.C2 = ConstraintList()
    model.C3 = ConstraintList()

    for t in model.Period:
        
        model.C1.add(model.Grid[t] + model.BATT[t] - model.PVEx[t] == model.Demand[t] - model.PV[t])   
            

    def SOC_Storage(model,t):
        if t == 0:
            return (model.SOC[t] == INICIAL_CHARGE - model.BATT[t])
        else:
            return (model.SOC[t] == model.SOC[t-1] - model.BATT[t])
        
    model.C2 = Constraint(model.Period, rule= SOC_Storage) 

    for t in model.Period:

        model.C3.add(model.PVEx[t] <= model.PV[t])   

    #Solve the problem
    results = opt.solve(model) 
    results_optimal = format(value(model.obj), ".2f")

    #Putting together a graph
    optimal_Grid = [value(model.Grid[key]) for key in model.Grid]
    optimal_SOC = [value(model.SOC[key]) for key in model.SOC]
    optimal_BATT = [value(model.BATT[key]) for key in model.BATT]
    optimal_PV = [value(model.PVEx[key]) for key in model.BATT]
    demand = df[building]

    generation = df['PV Power']
    df_Optimal = pd.DataFrame({"Demand (kWh)":demand,"Grid Consumption (kWh)":optimal_Grid,"SOC (kWh)":optimal_SOC,"Exchange with Battery (kWh)":optimal_BATT, "Sold Energy to Grid (kWh)":optimal_PV, "PV Generation (kWh)": generation})

    return df_Optimal , results_optimal

# Definition of Fuctions

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

# Metric errors for the final five models tested with XGBoost in the period of Q1-2024
metrics_table = pd.DataFrame({'Model': ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5'],
                              'Training Period': ['2019-2023', '2021-2023', '2021-Q1_2024', '2023-Q1_2024', 'Q1_2024'],
                              'MAE': [12.54, 10.79, 02.63, 01.48, 00.83],
                              'RMSE': [16.77, 14.12, 4.16, 2.81, 2.50],
                              'cvRMSE': ['38.0%', '32.0%', '9.4%', '6.4%', '5.7%']})


###Start of application
app = Dash(__name__, external_stylesheets=external_stylesheets)

#Application main layout
app.layout = html.Div([
    html.H2('Energy Services Project 3 - Market Price Prediction and Optimization'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Market Price Prediction', value='tab-1'),
        dcc.Tab(label='Optimization', value='tab-2'),
        dcc.Tab(label='Prediction Project Details', value='tab-3'),
        dcc.Tab(label='Optimization Project Details', value='tab-4'),
        ]),
    html.Div(id='tabs-content'),
])

@app.callback(
        dash.dependencies.Output('tabs-content', 'children'),
        dash.dependencies.Input('tabs', 'value'),
        )

def render_content(tab):

    if tab == 'tab-1':
        return html.Div([
                html.Div([
                    html.H2('Market Price Prediction vs Real Value'),
                    html.H4('Date:'),
                    dcc.DatePickerSingle(
                        id='my-date-picker-single',
                        min_date_allowed=datetime.date(2023, 1, 1),
                        max_date_allowed=datetime.date(2024, 4, 11),
                        initial_visible_month=datetime.date(2024, 1, 1),
                        date=datetime.date(2024,1,1),
                        #display_format ='MMM Do, YYYY'
                                        ),
                    
                        ]),
                
                html.Div([dcc.Graph(id='Price_Line')]),
                html.Div(className= 'six columns',id='metrics'),
                
        ])
    
    if tab == 'tab-2':
        return html.Div([
                html.H2('Battery Optimization for a Building'),
                html.H3('Select a building, the number of solar panels and battery capacity and check the results!'),
                
                dcc.Dropdown(
                id='dropdown',
                options=[
                {'label': 'Assembleia Municipal', 'value': 'AM Power (kW)'},
                {'label': 'Biblioteca Municipal de Belém', 'value': 'BL Power (kW)'},
                {'label': 'Polícia Municipal', 'value': 'PM Power (kW)'},
                        ],
                value='AM Power (kW)'
                ),

                html.Div([
                    html.H4('Date:'),
                    dcc.DatePickerSingle(
                        id='my-date-picker-single_1',
                        min_date_allowed=datetime.date(2023, 1, 1),
                        max_date_allowed=datetime.date(2024, 4, 11),
                        initial_visible_month=datetime.date(2024, 4, 1),
                        date=datetime.date(2024,4,11),
                        #display_format ='MMM Do, YYYY'
                                        ),
                        ]),

                html.Div([
                    html.Div([
                    html.H4('Number of Solar Panels (500 Watt Panels)'),
                    daq.NumericInput(
                    min=0,
                    max=100,
                    id='my-numeric-input-1',
                    value=2
                    ),], className="six columns"),
                    html.Div([
                    html.H4('Size of Battery (kWh)'),
                    daq.NumericInput(
                    min=0,
                    max=100,
                    id='my-numeric-input-2',
                    value=8
                    ),],className="six columns"),
                ], className= 'row'),
                
        dbc.Row(
            [
                dbc.Col(
                    html.Div([
                        dcc.Graph(
                                id='Line'
                                ),
                            ],)),

            ],
            ),
        html.H3('Results:'),
        html.Div(className= 'six columns', id='results_base'),
        html.Div(className= 'six columns',id='results_optimal'),
    
        ])
    if tab == 'tab-3':
        return html.Div([
            html.H3('Project Development Details'),
            html.H3(''),
            html.H5('Electricity Price Forecasting Model'),
            html.H5(''),
            html.H5('By: João Victor Costa IST1108317'),
            html.H5(''),
            html.H6('The forecast of electricity wholesale market prices is a challenge task due to its complexity and dependence of several variables that could affect the clearing price, such as demand, generation technology mix, renewable production, storage systems (e.g. hydro pumped), power plant availability, commodities prices (e.g. natural gas, coal, CO2), transfer capacity and energy exchange between regions or countries, etc. For time series forecasting is necessary develop models that can deal with complex and nonlinear dependencies, multivalent inputs, and multi-step forecasting. '),
            html.H6('Nevertheless, the comprehension of electricity price formation and ability to predict it can allow market participants optimize their assets in other to maximize profits or reduce costs. In this project, the price forecasting supports prosumers which have installed PV and batteries, and their goal is to optimize the storage system to reduce the energy costs of the building.'),
            html.H6('It was used open access data retrieved from ENTSO-E API and MIBGAS website to develop models for Portuguese electricity spot prices to the horizon of the next 72h. The models were initially tested with data from 2019 to 2023, them also include the first quarter of 2024. From ENTSO-E was extracted data of historical day-ahead prices from Portugal, Spain, and France; forecasted load of Portugal and Spain; generation aggregated and consumption of Portugal and Spain; generation by technology of Portugal and Spain; water reservoirs levels of Portugal and Spain; the transfer capacity and power exchange between the country boundaries (Portugal, Spain, France). From MIBGAS historical files was retrieved the natural gas price for the day-ahead (PVB D+1 contracts). '),
            html.H6('The raw data was analysed to identify the behaviour or tendency of the variables through the time and removed the few outliers by linear interpolations. We highlight the high volatility of the hourly prices during the period of study, varying from 0€ to more than 600€, and its high correlation with the natural gas price levels (Figure 1). Other interesting tendency is the impressive increase of solar power generation throughout the years, which affect directly in the price formation (Figure 2).'),
            html.Div([    
                html.Div([
                    html.Img(src=b64_image('_Fig01.png'), width="700"),
                    html.H6('Fig.01 - Electricity and Natural Gas price evolution.')
                ], style={'textAlign': 'center'}),
                html.Div([
                    html.Img(src=b64_image('_Fig02.png'), width="700"),
                    html.H6('Fig.02 - Solar Generation evolution.')
                ], style={'textAlign': 'center'}),
            ],
                style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}
            ),
            html.H6(''),
            html.H6('Regarding the distribution of the Portuguese electricity prices was plot a histogram between the period from January 2019 and March 2024, where is identified several occurrences at zero price level and other two peaks, the first near 50€ and a second smaller near 100€, as showed in the Figure 3. If it is plot a new histogram excluding prices from years 2019 and 2020, the peak at 50€ just disappear, as show in the Figure 4. It is important to note these behaviours for design the forecast algorithm, because training the model with different periods of time, more closer or distant for the current time, could lead to different outputs due to the data with which have been learning.'),
            html.Div([    
                html.Div([
                    html.Img(src=b64_image('_Fig03.png'), width="700"),
                    html.H6('Fig.03 - Electricity price histogram (Jan/2019-Mar/2024).')
                ], style={'textAlign': 'center'}),
                html.Div([
                    html.Img(src=b64_image('_Fig04.png'), width="700"),
                    html.H6('Fig.03 - Electricity price histogram (Jan/2021-Mar/2024).')
                ], style={'textAlign': 'center'}),
            ],
                style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}
            ),
            html.H6('A first correlation between all variables shows that currently electricity price is strongly correlated with the previous prices of electricity and the natural gas price level. Other variables that present quite positive correlation are the generation from natural gas, and the transfer capacity between Spain and Portugal. On the other hand, the water reservoir levels present significant negative correlation with the spot price, as show in the Figure 5.'),
            html.Div([
                html.Img(src=b64_image('_Fig05.png'), width="1200"),
                html.H6('Fig.05 - Variables Correlation.')
                ], style={'textAlign': 'center'}),
            html.H6(''),
            html.H6('Analysis of the features importance:'),
            html.Div([
                html.Img(src=b64_image('_Fig06.png'), width="700"),
                html.H6('Fig.06 - Feature analysis.')
                ], style={'textAlign': 'center'}),
            html.H6('Then it was tested different models, such as Linear Regression (LR), Random Forest (RF), Bootstrapping (BT), and XGBoost (XGB); for different feature combination sets (X0, X1, … X10) at the period from 2019 to 2023. The Figure 7 shows the performance of the models based on their cvRMSE. From this comparative analysis, the model which presented better results was the XGB with the following variables: previous 24h electricity prices of Portugal, Spain and France, natural gas price for the day-head, forecasted loads(PT and ES), generation aggregation and consumption (PT and ES), fossil gas generation (PT and ES), hydro generation (PT and ES), hydro pumped generation (PT and ES), wind generation (PT and ES), solar generation (PT and ES), nuclear generation (ES), water reservoirs (PT and ES), transfer capacity (PT, ES, FR), and calendar variables (hour, weekday, month, quarter, and holidays).  '),
            html.Div([
                html.Img(src=b64_image('_Fig07.png'), width="700"),
                html.H6('Fig.07 - Performance of different forecast models (data from 2019 to 2023).')
                ], style={'textAlign': 'center'}),
            html.H6('Then the selected forecast model was trained with different time periods and finally test with more resent price data from the first quarter of 2024, the result of these analysis is the table below. Based on this outcome was defined the Model 3 as the final model for the Portuguese electricity price forecast, considering the equilibrium between performance and range of observed data.'),
            html.Div([
                generate_table(metrics_table),
                ], style={'textAlign': 'center'}),
            html.Div([
                html.Img(src=b64_image('_Fig08.png'), width="700"),
                html.H6('Fig.08 - Energy Price Forecast: Model 3 (XGBoost 2021-Q1_2024).')
                ], style={'textAlign': 'center'}),
        ])
    
    if tab == 'tab-4':
        return html.Div([
            html.H3('Project Development Details'),
            html.H3(''),
            html.H5('Optimization Model'),
            html.H5(''),
            html.H5('By: Tomás Barosa Santos IST1110849'),
            html.H5(''),
            html.H6('In this project the objective was to develop an optimization model that manages the energy consumption of buildings.'),
            html.H6('The objective of the model is to minimize costs related to energy consumption of a building.'),
            html.H6('The building has a photovoltaic production system and batteries. Making this a multisource system.'),
            html.H6('The user of the interface can choose the building to optimize for, having 3 available buildings with different consumption profiles. Can also choose the number of photovoltaic panels installed and the capacity of the batteries.'),
            html.H6('Finally, the user can choose the starting date to optimize for, the model will optimize the system for 3 days or 72 hours.'),
            html.H6('The formulation of the optimization problem can be seen in the following figure:'),
            html.Div([
                html.Img(src=b64_image('formulation.png'), width="700"),
                html.H6('Fig.1 - Optimization Problem Formulation.')
                ], style={'textAlign': 'center'}),
            html.H6(''),
            html.H6('The optimization algorithm chosen has GLPK, a Linear Programming that uses a revised simplex method.'),
            html.H6('Reguraly used for multisource energy problems.'),
            html.H6(''),
            html.H5('Additional Information:'),
            html.H6('The price of energy being bougth from the grid is equal to Market Price plus 20% markup.'),
            html.H6('The price of energy sold to the grid is equal to the Market Price.'),
            html.H6('The cost without optimization doesnt take into consideration the battery, just the energy bought and sold to the grid.'),
            
        ])


@app.callback(
        
    dash.dependencies.Output('Price_Line', 'figure'),
    dash.dependencies.Output('metrics', 'children'),
        
    dash.dependencies.Input('my-date-picker-single', 'date'),

)

def dateselecter(date):
    
    df_forecast , metrics = prediction(df_PriceForescast,date)

    fig = px.line(df_forecast, x=df_forecast.index, y=['Y_pred', 'Y_real'], 
              labels={'x': 'Date', 'value': 'PT Price [€/MWh]'},
              title='Electricity Price Forecast',
              line_shape='linear')
    fig.update_traces(line=dict(color='red'), selector=dict(name='Y_real'))
    fig.update_traces(line=dict(dash='dash'), selector=dict(name='Y_real'))
    fig.update_traces(line=dict(color='darkblue'), selector=dict(name='Y_pred'))
    fig.update_layout(legend=dict(font=dict(size=12)),  
                    xaxis=dict(title='Date'),  
                    yaxis=dict(title='PT Price [€/MWh]'), 
                    height=600, width=1000,  
                    showlegend=True,  
                    yaxis_tickformat='.2f',  
                    template='plotly_white') 

    return fig, metrics

@app.callback(

    dash.dependencies.Output('Line', 'figure'),
    dash.dependencies.Output('results_base', 'children'),
    dash.dependencies.Output('results_optimal', 'children'),

    dash.dependencies.Input('my-date-picker-single_1', 'date'),
    dash.dependencies.Input('dropdown', 'value'),
    dash.dependencies.Input('my-numeric-input-1', 'value'),
    dash.dependencies.Input('my-numeric-input-2', 'value'),
)



def update_Optimization(date,value_drop,value_panels,size_batt):

    df_forecast, metrics = prediction(df_PriceForescast,date)
    df_data_1 = pd.merge(df_data, df_forecast, on='Date')

    #Buy price is market price plus 5% profit for retailer
    df_data_1['PT Price (€/kWh)'] = (df_data_1['Y_pred']/1000) *1.05

    #Sale price will be market price
    df_data_1['PV Sell'] = df_data_1['PT Price (€/kWh)']

    df_data_2 = FE_PV(df_data_1, value_panels)
    df_optimal , results_optimal = optimization_model(df_data_2, value_drop,size_batt)

    figure = px.line(df_optimal)
    
    results_base = format(results(df_data_2, value_drop),".2f")

    str_base = f'Costs without Optimization:{results_base}'
    str_optimal = f'Costs with Optimization:{results_optimal}'


    return figure, str_base, str_optimal


if __name__ == '__main__':
    app.run(debug=True)