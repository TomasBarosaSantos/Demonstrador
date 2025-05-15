import numpy as np
import pandas as pd
from pyomo.environ import *
import plotly.express as px
import pandas as pd
import dash
import datetime
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import pickle as pkl
from sklearn import metrics
from datetime import timedelta
import base64
import OptimizationModel
import NPV_Computing as NPV
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import optimization_metrics as optm
import dumb_PV_Batt as rule_based
import graph_script as graph
from io import BytesIO


#Import the forecasting model
model = pkl.load(open('src/XGBoostBP.pkl', 'rb'))

svr_model = pkl.load(open('src/SVR_Model.pkl', 'rb'))

#Import the data for the optimization
df_Opti_Data = pd.read_csv('src/Data_Demo.csv')
df_Opti_Data.set_index('Datetime', inplace=True, drop=True)

#Import the data for forecasting
df_Forecast_Data = pd.read_csv('src/dataPowerForecast.csv', index_col=0, parse_dates=True)
df_Forecast_Data.index.name = 'Datetime'

#Import solar forecasting data
dataset_PV_Pred = pd.read_csv('src/Dataset_PV_Pred.csv')
#dataset_PV_Pred = dataset_PV_Pred.rename(columns={'Date': 'Datetime'})
#dataset_PV_Pred = dataset_PV_Pred.set_index('Datetime', drop=True)
Hour_Parameters = pd.read_csv('src/Parametros_Horarios_SVR+SA_PV_NorteShopping(Hist(12)+Temp(Hist+Prev)).csv')
Hour_Parameters.columns.values[0] = 'Hour'

#Import the CSS file
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

value1 = 1.78e+13
value2 = 3.62e+12

#Create dataframe for the Solar Generation metrics
df_Gen = pd.DataFrame({'Metric': ['MAE [kW]', 'MSE [kW\u00b2]', 'RMSE [kW]', 'SSE [kW\u00b2]', 'CV_RMSE [%]', 'WAPE [%]'],
                           'Train' : [21.2, 3129.79, 55.94, "{:e}".format(value1), 36.28, 13.75],
                           'Test' : [21.33, 2753.07, 52.47, "{:e}".format(value2), 32.49, 13.21]})

df_Train_Test = pd.DataFrame({'Step': ['Train', 'Test'],'Percentage of data [%]': [70, 30], 'Time Period': ['17-01-2022 to 31-05-2023', '01-06-2023 to 31-12-2023']})

#df_Train_Test_PD = pd.DataFrame({'Step': ['Train', 'Test'],'Percentage of data [%]': [80, 20], 'Time Period': ['17-01-2022 to 31-05-2023', '01-06-2023 to 31-12-2023']})

df_consm = pd.DataFrame({'Model': ['XGBoost', 'Random Forest', 'Artificial Neural Network', 'Linear Regression', 'Support Vector Regression'],
                         'R\u00b2': [0.9811, 0.9556, 0.9774, 0.9676, 0.9536],
                         'RMSE [kW]': [82.77, 126.81, 90.55, 108.41, 129.66],
                         'MSE [kW\u00b2]': [6850.90, 16081.27, 8199.67, 11753.46, 16810.93],
                         'MAE [kW]': [46.91, 79.58, 45.48, 66.10, 75.77],
                         'MAPE [%]': [6.19, 10.37, 5.68, 9.36, 8.89],
                         'Total Training Time [s]': [4.53, 0.03, 40.66, 80.26, 10.35],
                         'Total Testing Time [s]': [0.018, 0.001, 0.018, 0.009, 0.732],

})

Predicted_Values = []

for h in range(0, 24):
    # Select rows where 'Hora' is equal to h
    Inputs_hora = dataset_PV_Pred.loc[dataset_PV_Pred['Hora'] == h].iloc[:, 2:]
    Outputs_hora = dataset_PV_Pred.loc[dataset_PV_Pred['Hora'] == h].iloc[:, 1]
    # Set SVR model parameters for the current Hora
    svr_model.set_params(
        C=Hour_Parameters.loc[Hour_Parameters['Hour'] == h, 'C'].values[0],
        gamma=Hour_Parameters.loc[Hour_Parameters['Hour'] == h, 'gamma'].values[0],
        epsilon=Hour_Parameters.loc[Hour_Parameters['Hour'] == h, 'epsilon'].values[0],
        kernel='rbf'
    )

    # Prepare the input data for prediction
    X = Inputs_hora[['Lag51', 'Lag50', 'Lag49', 'Lag48', 'Lag27', 'Lag26', 'Lag25', 'Lag24', 'Lag3', 'Lag2', 'Lag1', 'Temperatura_24H', 'Temperatura_Prev']]
    X = X.values.reshape(-1, 13)  # Reshape to 2D array for prediction
    Y = Outputs_hora
    # Fit the SVR model to the data
    svr_model.fit(X, Y) 
    # Predict using the SVR model
    Y_pred = svr_model.predict(X)
    # Append the predicted values to the list
    Predicted_Values.append(Y_pred)

Previsoes = np.array(Predicted_Values).T.reshape(-1, 1)
    
Previsoes = pd.DataFrame(Previsoes, columns=['Forecasted'])

for i in range(len(Previsoes)):
    if Previsoes.loc[i,'Forecasted'] < 0:
        Previsoes.loc[i,'Forecasted'] = 0
    else:
        Previsoes.loc[i,'Forecasted'] = Previsoes.loc[i,'Forecasted']

Previsoes['Real'] = dataset_PV_Pred['Real'].values
Previsoes['Datetime'] = dataset_PV_Pred['Date'].values


##Function Definitions
#

#Function that creates a table
def generate_table_head(dataframe, max_rows=10):
    return html.Table([
        html.Thead([
            html.Tr([html.Th(col) for col in dataframe.columns])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ],style={'width': '50%', 'height': '50%','margin-left': 'auto', 'margin-right': 'auto', 'text-align': 'center','border-collapse': 'collapse'})

#Function that creates a table
def generate_table_wHead(dataframe, max_rows=10):
    return html.Table([
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ],style={'width': '35%', 'height': '35%','margin-left': 'auto', 'margin-right': 'auto', 'text-align': 'center','border-collapse': 'collapse'})

#Function that opens a image file and converts it to base64
def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

#Function that formats the date 
def date_transform(date):
    date_format ='%Y-%m-%d %H:%M:%S+00:00'
    date = date.strftime(date_format)   
    return date

#Function that creates a dataframe with the data for the selected date
def dataframe_date(df,date):

    end_date = pd.to_datetime(date) + timedelta(days=3)
    end_date = date_transform(end_date)

    df = df.reset_index()
    
    mask_data = (df['Datetime'] >= date) & (df['Datetime'] <= end_date)
    df = df.loc[mask_data]

    df = df.set_index('Datetime', drop=True)

    return df

#Function that reformats the dataframe
def dataframe_reformat(df):
    df = df.reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime', drop=True)
    return df


#Function that computes the metrics for the Optimization Algorithm
def optimization_metrics(df_NoBatt,df_Batt,pv_power,batt_capacity):

    all_costs = (df_Opti_Data['Cost (Eur/kW)'] + df_Opti_Data['Access Tariff (Eur/kWh)'] + df_Opti_Data['PHP Tariff (Eur/kWh)'])
    Cost_NoBatt = df_NoBatt['Grid Electric Consumption [kW]']*all_costs
    Cost_NoBatt = Cost_NoBatt.dropna()

    Price_Energy_Grid_NoBatt = sum(Cost_NoBatt)
    Battery_Cycles = round((sum(df_Batt['Electrical Energy Charged [kW]']) + sum(df_Batt['Electrical Energy Discharged [kW]']) ) / (2* batt_capacity),2)

    Cost_Batt = df_Batt['Grid Electric Consumption [kW]']*all_costs
    Cost_Batt = Cost_Batt.dropna()

    Price_Energy_Grid = sum(Cost_Batt)
    Savings = round(Price_Energy_Grid_NoBatt - Price_Energy_Grid,7)

    column_names = ['Savings', 'Battery Cycles']
    df_Results = pd.DataFrame(columns= column_names)
    df_Results.loc[0] = [str(round(Savings,2))+' €', str(Battery_Cycles)+ ' Cycles']
    df_Results = df_Results.T
    df_Results = df_Results.reset_index()
    
    return df_Results


def run_forecasting(df_Forecast_Data,date):
    end_date = pd.to_datetime(date) + timedelta(days=3)
    end_date = end_date.strftime('%Y-%m-%d')

    X_Real = df_Forecast_Data.drop(columns=['Power Demand'])
    Y_Real = df_Forecast_Data['Power Demand']

    scaler = RobustScaler()
    X_Real = scaler.fit_transform(X_Real)

    Y_Pred = model.predict(X_Real)

    data_graph = pd.DataFrame({'Real': Y_Real, 'Forecasted': Y_Pred})

    data_graph = data_graph.reset_index()
        
    mask_data = (data_graph['Datetime'] >= date) & (data_graph['Datetime'] <= end_date)
    data_graph = data_graph.loc[mask_data]

    data_graph = data_graph.set_index('Datetime', drop=True)

    return data_graph

def run_forecasting_PV(Previsoes,date):

    end_date = pd.to_datetime(date) + timedelta(days=3)
    end_date = end_date.strftime('%Y-%m-%d')
    
    mask_data = (Previsoes['Datetime'] >= date) & (Previsoes['Datetime'] <= end_date)
    Previsoes = Previsoes.loc[mask_data]
    Previsoes = Previsoes.set_index('Datetime', drop=True)

    return Previsoes


def generate_fig_forecasting(data_graph):

    figure = go.Figure()

    figure.add_trace(go.Scatter(x=data_graph.index, y=data_graph['Real'], mode='lines', name='Real', ))
    figure.add_trace(go.Scatter(x=data_graph.index, y=data_graph['Forecasted'], line = dict(color='red', width=2, dash='dash'), name='Forecasted'))
    figure.update_layout(xaxis_title='Datetime', yaxis_title='Power Demand [kW]')
    figure.update_layout(legend=dict(
        yanchor="bottom",
        y=-0.37,
        xanchor="center",
        x=0.50,
        orientation="h",

    ),legend_title_text="")

    return figure


def generate_fig_solar(data_solar):

    figure = go.Figure()

    figure.add_trace(go.Scatter(x=data_solar.index, y=data_solar['Real'], mode='lines', name='Real', ))
    figure.add_trace(go.Scatter(x=data_solar.index, y=data_solar['Forecasted'], line = dict(color='red', width=2, dash='dash'), name='Forecasted'))
    figure.update_layout(xaxis_title='Datetime', yaxis_title='PV Generation [kW]')
    figure.update_layout(legend=dict(
        yanchor="bottom",
        y=-0.37,
        xanchor="center",
        x=0.50,
        orientation="h",

    ),legend_title_text="")

    return figure

def combine_data(df, data_graph, data_solar):

    df = df.iloc[0:72]
    data_graph = data_graph.iloc[0:72]
    df['Electric Power (kW)'] = data_graph['Forecasted'].values
    df['Thermal Power (kW)'] = np.zeros(len(df))

    df['Irradiance (W/m2)'] = data_solar['Forecasted'].values
    df['Irradiance (W/m2)'] = (df['Irradiance (W/m2)']/3000)/(0.193*2*0.98) * 1000 

    df.dropna(inplace=True)

    return df

def forecasting_metrics(Y_Real, Y_Pred):
    #Calculate errors
    errors = Y_Real - Y_Pred
    mae = round(np.mean(np.abs(errors)),2)
    mse = round(np.mean(errors**2),2)
    rmse = round(np.sqrt(mse),2)
    mape = round(np.mean(np.abs(errors / Y_Real)) * 100,2)
    r2 = round(r2_score(Y_Real,Y_Pred),2)

    df_forecasting_metrics = pd.DataFrame(index=["Row 1"], data={'MAE [kW]':mae,'RMSE [kW]':rmse,'MAPE [%]':mape,'R\u00b2':r2})

    return df_forecasting_metrics

def forecasting_metrics_PV(Y_Real, Y_Pred):
    #Calculate errors
    errors = Y_Real - Y_Pred
    mae = round(np.mean(np.abs(errors)),2)
    mse = round(np.mean(errors**2),2)
    rmse = round(np.sqrt(mse),2)
    r2 = round(r2_score(Y_Real,Y_Pred),2)
    WAPE = round(sum(np.abs(Y_Pred-Y_Real))/sum(Y_Real),2)
    SSE = round(sum(errors**2),2)

    df_forecasting_metrics = pd.DataFrame(index=["Row 1"], data={'MAE [kW]':mae,'RMSE [kW]':rmse,'WAPE':WAPE,'R\u00b2':r2, 'SSE [kW\u00b2]':SSE})

    return df_forecasting_metrics


##Start of application
#
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

#Application main layout
app.layout = html.Div([
    
    html.Img(
        src=b64_image('src/Branding ATE/Logos ATE/Logos ATE_PT/logo cor_pt-01.png'),
        style={'width': '22%', 'height': '22%'}
        ),

    html.Img(
        src=b64_image('src/Branding ATE/Barra Assinaturas PRR/Barra PRR.png'),
        style={'width': '50%', 'height': '50%'}
        ),

    html.H2('Building Energy Management System Optimisation Tool in Smart Buildings'),

    html.H3('Choose a date to run the algorithms:'),
    dcc.DatePickerSingle(
                        id='my-date-picker-single',
                        min_date_allowed=datetime.date(2022, 1, 1),
                        max_date_allowed=datetime.date(2023, 12, 31),
                        initial_visible_month=datetime.date(2022, 4, 10),
                        date=datetime.date(2022, 4, 10),
                        display_format ='DD/MM/YYYY',
                                        ),
    
    dcc.Tabs(id='tabs', value='tab-1', 
            children=[
                dcc.Tab(label='Power Demand and Generation Forecasting Algorithms', value='tab-1'),
                dcc.Tab(label='Energy Optimization Algorithm', value='tab-2'),
                dcc.Tab(label='Test', value='tab-4'),
                dcc.Tab(label='About', value='tab-3'),
                
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
                html.H3('Power Demand Forecasting:'),
                dcc.Loading([
                    dcc.Graph(id='forecasting_graph'),
                ]),
            ], className="six columns", style={'textAlign': 'center'}),

            html.Div([
                html.H3('PV Generation Forecasting:'),
                dcc.Loading([
                    dcc.Graph(id='solar_graph'),
                ]),
            ], className="six columns", style={'textAlign': 'center'}),

            html.Div([
                html.Div([
                    html.H4('Metrics of energy consumption forecasting:'),
                ],className="six columns", style={'textAlign': 'center'}),
                html.Div([
                    html.H4('Metrics of solar generation forecasting:'),
                ],className="six columns", style={'textAlign': 'center'})
            ], className="twelve columns", style={'textAlign': 'center'}),


            html.Div([
                html.Div([
                    dcc.Loading([
                    html.Div(id='forecasting_metrics_Consumption'),
                    ]),
                ], className="six columns",style={'textAlign': 'center'}),
                html.Div([
                    dcc.Loading([
                    html.Div(id='forecasting_metrics_PV'),
                    ]),
                ], className="six columns",style={'textAlign': 'center'}),
            ], className="twelve columns",style={'textAlign': 'center'}),


        ])
    
    if tab == 'tab-2':
        return html.Div([
                html.H2('Optimal Scheduling of Energy Storage'),
                html.H4('Choose the the input parameters for the optimization:'),

                html.Div([
                    html.Div([
                        html.H5('PV Power [kWp]'),
                        daq.NumericInput(
                            min=0,
                            max=30000,
                            id='pv_power',
                            value=1200,
                            size=100,),
                    ], className="six columns", style={'textAlign': 'center'}),
                    html.Div([
                        html.H5('Battery Capacity [kWh]'),
                        daq.NumericInput(
                            min=0,
                            max=30000,
                            id='batt_capacity',
                            value=800,
                            size=100,),
                    ], className="six columns", style={'textAlign': 'center'}),
                ], className= 'row', style={'textAlign': 'center'}),

                html.Br(),

                html.Div([

                    html.Div([
                        dcc.Dropdown(['Rule-Based Model','Optimisation Model without LCOS', 'Optimisation Model with LCOS'],
                            id='dropdown_1',
                            value='Rule-Based Model', 
                            multi=False,
                            searchable=False,
                        ),
                        html.Div(id='dropdown-output-container-1',style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                    
                    html.Div([
                        dcc.Dropdown(['Rule-Based Model','Optimisation Model without LCOS', 'Optimisation Model with LCOS'],
                            id='dropdown_2',
                            value='Optimisation Model without LCOS', 
                            multi=False,
                            searchable=False,
                        ),
                        html.Div(id='dropdown-output-container-2', style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                    
                    
                ], className="row", style={'textAlign': 'center'}),

                html.Div([
                    html.Div([
                        dcc.Loading([
                            dcc.Graph(id='Optimized-Graph_1'),
                        ], className="six columns", style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                        
                    html.Div([
                        dcc.Loading([
                            dcc.Graph(id='Optimized-Graph_2'),
                        ], className="six columns", style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                ], className="row", style={'textAlign': 'center'}),

                html.Div([
                    html.H3('Metrics:'),
                ], className="twelve columns",style={'textAlign': 'center'}),

                html.Div([
                    html.Div([
                        dcc.Loading([
                        html.Div(id='optimization_metrics_1'),
                        ]),
                    ], className="six columns",style={'textAlign': 'center'}),
                    html.Div([
                        dcc.Loading([
                        html.Div(id='optimization_metrics_2'),
                        ]),
                    ], className="six columns",style={'textAlign': 'center'}),
                ], className="twelve columns",style={'textAlign': 'center'}),

    
            ])
    if tab == 'tab-3':
        return html.Div([
            html.H3('About:'),
            dcc.Dropdown(['Overall System','Energy Consumption Forecasting', 'Solar Generation Forecasting', 'Energy Optimization', 'Team'],
                         id='dropdown',
                         value='Overall System', 
                         multi=False,
                         searchable=False,
                         ),
            html.Div(id='dropdown-output-container'),
        ]),
    if tab == 'tab-4':
        return html.Div([
            html.H2('Optimal Scheduling of Energy Storage'),
                html.H4('Choose the the input parameters for the optimization:'),

                html.Div([
                    html.Div([
                        html.H5('PV Power [kWp]'),
                        daq.NumericInput(
                            min=0,
                            max=30000,
                            id='pv_power_t',
                            value=1200,
                            size=100,),
                    ], className="six columns", style={'textAlign': 'center'}),
                    html.Div([
                        html.H5('Battery Capacity [kWh]'),
                        daq.NumericInput(
                            min=0,
                            max=30000,
                            id='batt_capacity_t',
                            value=800,
                            size=100,),
                    ], className="six columns", style={'textAlign': 'center'}),
                ], className= 'row', style={'textAlign': 'center'}),

                html.Br(),

                html.Div([

                    html.Div([
                        dcc.Dropdown(['Rule-Based Model','Optimisation Model without LCOS', 'Optimisation Model with LCOS'],
                            id='dropdown_t',
                            value='Rule-Based Model', 
                            multi=False,
                            searchable=False,
                        ),
                        html.Div(id='dropdown-output-container-t',style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                    
                    html.Div([
                        dcc.Dropdown(['Rule-Based Model','Optimisation Model without LCOS', 'Optimisation Model with LCOS'],
                            id='dropdown_t2',
                            value='Optimisation Model without LCOS', 
                            multi=False,
                            searchable=False,
                        ),
                        html.Div(id='dropdown-output-container-t2', style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                    
                    
                ], className="row", style={'textAlign': 'center'}),

                html.Div([
                    html.Div([
                        dcc.Loading([
                            html.Img(id='Optimized-Graph_t', width='100%'),
                        ], className="six columns", style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                        
                    html.Div([
                        dcc.Loading([
                            html.Img(id='Optimized-Graph_t2', width='100%'),
                        ], className="six columns", style={'textAlign': 'center'}),
                    ], className="six columns", style={'textAlign': 'center'}),
                ], className="row", style={'textAlign': 'center'}),

                html.Div([
                    html.H3('Metrics:'),
                ], className="twelve columns",style={'textAlign': 'center'}),

                html.Div([
                    html.Div([
                        dcc.Loading([
                        html.Div(id='optimization_metrics_t'),
                        ]),
                    ], className="six columns",style={'textAlign': 'center'}),
                    html.Div([
                        dcc.Loading([
                        html.Div(id='optimization_metrics_t2'),
                        ]),
                    ], className="six columns",style={'textAlign': 'center'}),
                ], className="twelve columns",style={'textAlign': 'center'}),

        ])
            
@app.callback(        
        
    dash.dependencies.Output('forecasting_graph', 'figure'),
    dash.dependencies.Output('solar_graph','figure'),
    dash.dependencies.Output('forecasting_metrics_Consumption', 'children'),
    dash.dependencies.Output('forecasting_metrics_PV', 'children'),

    dash.dependencies.Input('my-date-picker-single', 'date'),
)

def forecasting(date):

    data_graph = run_forecasting(df_Forecast_Data,date)

    forecast_fig = generate_fig_forecasting(data_graph)
    
    df_forecasting_metrics = forecasting_metrics(data_graph['Real'],data_graph['Forecasted'])

    df_solar_pred = run_forecasting_PV(Previsoes,date)
    df_forecasting_metrics_PV = forecasting_metrics_PV(df_solar_pred['Real'],df_solar_pred['Forecasted'])

    solar_fig = generate_fig_solar(df_solar_pred)

    table_metrics_Consumption = generate_table_head(df_forecasting_metrics)
    table_metrics_PV = generate_table_head(df_forecasting_metrics_PV)

    return forecast_fig, solar_fig, table_metrics_Consumption, table_metrics_PV


@app.callback(
    dash.dependencies.Output('dropdown-output-container-1', 'children'),
    dash.dependencies.Output('dropdown-output-container-2', 'children'),

    dash.dependencies.Input('dropdown_1', 'value'),
    dash.dependencies.Input('dropdown_2', 'value'),
)

def dropdown_output(dropdown_1, dropdown_2):

    if dropdown_1 == 'Rule-Based Model':
        text_1 = 'Rule-Based Model:'
    elif dropdown_1 == 'Optimisation Model without LCOS':
        text_1 = 'Optimisation Model without LCOS:'
    elif dropdown_1 == 'Optimisation Model with LCOS':
        text_1 = 'Optimisation Model with LCOS:'
    if dropdown_2 == 'Rule-Based Model':
        text_2 = 'Rule-Based Model:'
    elif dropdown_2 == 'Optimisation Model without LCOS':
        text_2 = 'Optimisation Model without LCOS:'
    elif dropdown_2 == 'Optimisation Model with LCOS':
        text_2 = 'Optimisation Model with LCOS:'
    
    return html.H4(text_1), html.H4(text_2)

@app.callback(
    dash.dependencies.Output('Optimized-Graph_1', 'figure'),
    dash.dependencies.Output('optimization_metrics_1', 'children'),
    dash.dependencies.Output('Optimized-Graph_2', 'figure'),
    dash.dependencies.Output('optimization_metrics_2', 'children'),

    dash.dependencies.Input('dropdown_1', 'value'),
    dash.dependencies.Input('dropdown_2', 'value'),
    dash.dependencies.Input('my-date-picker-single', 'date'),
    dash.dependencies.Input('pv_power', 'value'),
    dash.dependencies.Input('batt_capacity', 'value'),
)

def dropdown_output_1(dropdown_1, dropdown_2, date, pv_power, batt_capacity):
    df = dataframe_date(df_Opti_Data,date)

    data_Forecast = run_forecasting(df_Forecast_Data,date)
    data_PV = run_forecasting_PV(Previsoes,date)
    df_Data = combine_data(df, data_Forecast, data_PV)
    df_OptimalElectric_NoBatt, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, 0.0, 0.0, df_Data, True, False)

    if dropdown_1 or dropdown_2 == 'Rule-Based Model':
        df_Rule = rule_based.dumb_PV_Batt(df_Data, pv_power, batt_capacity)
        Graph_Rule = px.line(df_Rule,labels={'x': ' ', 'value': 'Power [kW]'})
        Graph_Rule.update_layout(legend=dict(
            yanchor="bottom",
            y=-0.50,
            xanchor="center",
            x=0.50,
            orientation="h",
        ),legend_title_text="")
        table_metrics_Rule = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_Rule, pv_power, batt_capacity))

    if dropdown_1 or dropdown_2 == 'Optimisation Model without LCOS':
        df_OptimalElectric_No_LCOS, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, batt_capacity, 0.0, df_Data, False, False)
        Graph_NoLCOS = px.line(df_OptimalElectric_No_LCOS,labels={'x': ' ', 'value': 'Power [kW]'})
        Graph_NoLCOS.update_layout(legend=dict(
            yanchor="bottom",
            y=-0.50,
            xanchor="center",
            x=0.50,
            orientation="h",

        ),legend_title_text="")
        table_metrics_No_LCOS = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_OptimalElectric_No_LCOS, pv_power, batt_capacity))

    if dropdown_1 or dropdown_2 == 'Optimisation Model with LCOS':
        df_OptimalElectric_LCOS, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, batt_capacity, 0.0, df_Data, True, False)
        Graph_LCOS = px.line(df_OptimalElectric_LCOS,labels={'x': ' ', 'value': 'Power [kW]'})
        Graph_LCOS.update_layout(legend=dict(
            yanchor="bottom",
            y=-0.50,
            xanchor="center",
            x=0.50,
            orientation="h",

        ),legend_title_text="")
        table_metrics = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_OptimalElectric_LCOS, pv_power, batt_capacity))
    
    if dropdown_1 == 'Rule-Based Model':
        Graph_1 = Graph_Rule
        table_metrics_1 = table_metrics_Rule
    elif dropdown_1 == 'Optimisation Model without LCOS':
        Graph_1 = Graph_NoLCOS
        table_metrics_1 = table_metrics_No_LCOS
    elif dropdown_1 == 'Optimisation Model with LCOS':
        Graph_1 = Graph_LCOS
        table_metrics_1 = table_metrics
    if dropdown_2 == 'Rule-Based Model':
        Graph_2 = Graph_Rule
        table_metrics_2 = table_metrics_Rule
    elif dropdown_2 == 'Optimisation Model without LCOS':
        Graph_2 = Graph_NoLCOS
        table_metrics_2 = table_metrics_No_LCOS
    elif dropdown_2 == 'Optimisation Model with LCOS':
        Graph_2 = Graph_LCOS
        table_metrics_2 = table_metrics

    return Graph_1, table_metrics_1, Graph_2, table_metrics_2

@app.callback(
    dash.dependencies.Output('dropdown-output-container-t', 'children'),
    dash.dependencies.Output('dropdown-output-container-t2', 'children'),

    dash.dependencies.Input('dropdown_t', 'value'),
    dash.dependencies.Input('dropdown_t2', 'value'),
)

def dropdown_output(dropdown_t, dropdown_t2):

    if dropdown_t == 'Rule-Based Model':
        text_1 = 'Rule-Based Model:'
    elif dropdown_t == 'Optimisation Model without LCOS':
        text_1 = 'Optimisation Model without LCOS:'
    elif dropdown_t == 'Optimisation Model with LCOS':
        text_1 = 'Optimisation Model with LCOS:'
    if dropdown_t2 == 'Rule-Based Model':
        text_2 = 'Rule-Based Model:'
    elif dropdown_t2 == 'Optimisation Model without LCOS':
        text_2 = 'Optimisation Model without LCOS:'
    elif dropdown_t2 == 'Optimisation Model with LCOS':
        text_2 = 'Optimisation Model with LCOS:'
    
    return html.H4(text_1), html.H4(text_2)


@app.callback(
    dash.dependencies.Output('Optimized-Graph_t', component_property='src'),
    dash.dependencies.Output('optimization_metrics_t', 'children'),
    dash.dependencies.Output('Optimized-Graph_t2', component_property='src'),
    dash.dependencies.Output('optimization_metrics_t2', 'children'),

    dash.dependencies.Input('dropdown_t', 'value'),
    dash.dependencies.Input('dropdown_t2', 'value'),
    dash.dependencies.Input('my-date-picker-single', 'date'),
    dash.dependencies.Input('pv_power_t', 'value'),
    dash.dependencies.Input('batt_capacity_t', 'value'),
)

def dropdown_output_1(dropdown_1, dropdown_2, date, pv_power, batt_capacity):

    df = dataframe_date(df_Opti_Data,date)
    data_Forecast = run_forecasting(df_Forecast_Data,date)
    data_PV = run_forecasting_PV(Previsoes,date)
    df_Data = combine_data(df, data_Forecast, data_PV)
    df_OptimalElectric_NoBatt, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, 0.0, 0.0, df_Data, True, False)

    if dropdown_1 or dropdown_2 == 'Rule-Based Model':
        df_Rule = rule_based.dumb_PV_Batt(df_Data, pv_power, batt_capacity)
        Graph_Rule = graph.plot_results(df_Rule,"",batt_capacity)
        buf = BytesIO()
        Graph_Rule.savefig(buf, format="png")
        fig_data_Rule = base64.b64encode(buf.getbuffer()).decode("ascii")
        Graph_Rule = f'data:image/png;base64,{fig_data_Rule}'

        table_metrics_Rule = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_Rule, pv_power, batt_capacity))

    if dropdown_1 or dropdown_2 == 'Optimisation Model without LCOS':
        df_OptimalElectric_No_LCOS, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, batt_capacity, 0.0, df_Data, False, False)
        Graph_NoLCOS = graph.plot_results(df_OptimalElectric_No_LCOS,"",batt_capacity)
        buf = BytesIO()
        Graph_NoLCOS.savefig(buf, format="png")
        fig_data_NoLCOS = base64.b64encode(buf.getbuffer()).decode("ascii")
        Graph_NoLCOS = f'data:image/png;base64,{fig_data_NoLCOS}'

        table_metrics_No_LCOS = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_OptimalElectric_No_LCOS, pv_power, batt_capacity))

    if dropdown_1 or dropdown_2 == 'Optimisation Model with LCOS':
        df_OptimalElectric_LCOS, optimal_Value , df_OptimalThermal, model, results = OptimizationModel.Optimization(pv_power, batt_capacity, 0.0, df_Data, True, False)
        Graph_LCOS = graph.plot_results(df_OptimalElectric_LCOS,"",batt_capacity)

        buf = BytesIO()
        Graph_LCOS.savefig(buf, format="png")
        fig_data_LCOS = base64.b64encode(buf.getbuffer()).decode("ascii")
        Graph_LCOS = f'data:image/png;base64,{fig_data_LCOS}'

        table_metrics = generate_table_wHead(optimization_metrics(df_OptimalElectric_NoBatt, df_OptimalElectric_LCOS, pv_power, batt_capacity))
    
    if dropdown_1 == 'Rule-Based Model':
        Graph_1 = Graph_Rule
        table_metrics_1 = table_metrics_Rule 
    elif dropdown_1 == 'Optimisation Model without LCOS':
        Graph_1 = Graph_NoLCOS
        table_metrics_1 = table_metrics_No_LCOS
    elif dropdown_1 == 'Optimisation Model with LCOS':
        Graph_1 = Graph_LCOS
        table_metrics_1 = table_metrics
    if dropdown_2 == 'Rule-Based Model':
        Graph_2 = Graph_Rule
        table_metrics_2 = table_metrics_Rule
    elif dropdown_2 == 'Optimisation Model without LCOS':
        Graph_2 = Graph_NoLCOS
        table_metrics_2 = table_metrics_No_LCOS
    elif dropdown_2 == 'Optimisation Model with LCOS':
        Graph_2 = Graph_LCOS
        table_metrics_2 = table_metrics

    return Graph_1, table_metrics_1, Graph_2, table_metrics_2

@app.callback(
        
    dash.dependencies.Output('dropdown-output-container', 'children'),

    dash.dependencies.Input('dropdown', 'value'),
)

def dropdown_output(dropdown):
    if dropdown == 'Overall System':
        return html.Div([
            html.H4('Overall System:'),
            html.Div([
                html.Img(
                    src=b64_image('src/Optimisação/System.png'),
                    style={'width': '60%', 'height': '60%'}
                ),
            ],className="twelve columns", style={'textAlign': 'center'}),

        ])
    elif dropdown == 'Energy Consumption Forecasting':
        return html.Div([
            html.H4('This algorithm is used for forecasting the power demand.'),
            html.H4('Using a XGBoost algorithm to forecast the energy consumption.'),

            html.H4('Five different algorithms were tested:'),
            html.Ul([
                html.Li('XGBoost;'),
                html.Li('Random Forest;'),
                html.Li('Artificial Neural Network;'),
                html.Li('Linear Regression;'),
                html.Li('Support Vector Regressor;'),
            ], style={'padding-left': '50px','font-size': '20px'}),

            html.Br(style={'height': '50px'}),

            html.H4('The algorithm uses the following features:'),
            html.Div([
                html.Img(
                    src=b64_image('src/consum_fore/features.png'),
                    style={'width': '60%', 'height': '60%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(),
            html.H4('Optuna was used to optimize the hyperparameters of algorithms.'),
            html.Br(),

            html.H5('The top 20 features selected by the XGBoost algorithm were:'),
            html.Div([
                html.Img(
                    src=b64_image('src/consum_fore/feature_importance.png'),
                    style={'width': '35%', 'height': '35%'}
                    ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(),

            html.H5('The performance and errors metrics of the different algorithms are:'),
            html.Div([
                generate_table_head(df_consm)
                ]),

            html.Br(style={'height': '50px'}),
            html.H5('The following graphs show scatter plots of the real vs predicted values for the algorthims:'),
            html.Div([
                html.Div([
                    html.Img(
                        src=b64_image('src/consum_fore/lr_r2.png'),
                        style={'width': '120%', 'height': '120%'}
                    ),
                ], className="two columns"),
                html.Div([
                    html.Img(
                        src=b64_image('src/consum_fore/rf_r2.png'),
                        style={'width': '120%', 'height': '120%'}
                        ),
                ], className="two columns"),
                html.Div([
                    html.Img(
                        src=b64_image('src/consum_fore/ann_r2.png'),
                        style={'width': '120%', 'height': '120%'}
                        ),
                ], className="two columns"),
                html.Div([
                    html.Img(
                        src=b64_image('src/consum_fore/xgboost_r2.png'),
                        style={'width': '120%', 'height': '120%'}
                        ),
                ], className="two columns"),
                html.Div([
                    html.Img(
                        src=b64_image('src/consum_fore/svr_r2.png'),
                        style={'width': '120%', 'height': '120%'}
                        ),
                ], className="two columns"),
            ],className="twelve columns", style={'textAlign': 'center'}),

            html.Br(style={'height': '100px'}),
            html.Br(style={'height': '100px'}),
            html.H5('RMSE boxplot of the different algorithms:'),

            html.Div([
                html.Img(
                    src=b64_image('src/consum_fore/rmse_boxplot.png'),
                    style={'width': '35%', 'height': '35%'}
                    ),
            ], className="twelve columns", style={'textAlign': 'center'}),

        ])
    elif dropdown == 'Solar Generation Forecasting':
        return html.Div([

            html.H4('A Support Vector Machine (SVM) algorithm is used to forecast the solar generation.'),
            html.H4('Using Simulated Annealing (SA) to optimize the hyperparameters of the SVM algorithm.'),

            html.Br(style={'height': '25px'}),

            html.H4('The algorithm uses the following features:'),
            html.Ul([
                html.Li('Lagged values for the three most recent hours of the days n, n-1 and n-2;'),
                html.Li('Temperature value at the forecasted hour and corresponding value recorded 24 hours earlier;'),
            ], style={'padding-left': '50px','font-size': '20px'}),

            html.Br(style={'height': '25px'}),

            html.H5('The data used to train and test the algorithm was:'),
            html.Div([
                generate_table_head(df_Train_Test)
                ]),

            html.Br(style={'height': '50px'}),

            html.H5('Simulated Annealing functioning flowchart:'),
            html.Div([
                html.Img(
                    src=b64_image('src/PV Gen/fluxograma_SA.png'),
                    style={'width': '20%', 'height': '20%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(style={'height': '50px'}),
            html.H5('During the SA process, 30 cycles were performed per hour. ' \
            'In each cycle, 15 random combinations of SVM parameters were tested. At the end of each cycle, the temperature' \
            'was reduced to narrow the range of available options for each parameter. ' \
            'The C (cost) parameter was tested within the range of 0 to 700, Epsilon between 0.00001 and 3, and Gamma between 0.0001 and 10. ' \
            'The objective function defined for this process was based on the RMSE obtained for the test subset.'),

            html.Br(style={'height': '50px'}),

            html.H5('The SVM parameters optimization process performed by the simulated annealing algorithm was:'),
            html.Div([
                html.Img(
                    src=b64_image('src/PV Gen/SVM_params.png'),
                    style={'width': '30%', 'height': '30%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(style={'height': '50px'}),
            html.H5('Heatmap of Correlation Matrix:'),
            html.Div([
                html.Img(
                    src=b64_image('src/PV Gen/correlation_matrix.png'),
                    style={'width': '30%', 'height': '30%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Div([
                html.Img(
                    src=b64_image('src/PV Gen/r2_graph.png'),
                    style={'width': '30%', 'height': '30%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(style={'height': '50px'}),
            html.H5('The errors metrics of the algorithm are:'),

            html.Div([
                generate_table_head(df_Gen)
            ]),
            

        ])
    elif dropdown == 'Energy Optimization':
        return html.Div([
            html.H4('This algorithm is used to optimize scheduling of the energy storage system, using the energy consumption and PV generated forecasts.'),
            html.Div([
                html.Img(
                    src=b64_image('src/Optimisação/Fluxograma.png'),
                    style={'width': '20%', 'height': '20%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),
            
            html.Br(style={'height': '50px'}),

            html.H4('MILP Optimization Algorithm Formulation:'),
            html.Ul([
                html.Li('Solved using CPLEX'),
            ], style={'padding-left': '50px','font-size': '20px'}),

            html.Div([
                html.Img(
                    src=b64_image('src/Optimisação/equation-1.png'),
                    style={'width': '40%', 'height': '40%'}
                ),
            ],className="twelve columns", style={'textAlign': 'center'}),

            html.H5('Being:'),
            html.Div([
                html.Img(
                    src=b64_image('src/Optimisação/LCOS_Equation-1.png'),
                    style={'width': '15%', 'height': '15%'}
                    ),
                ], className="twelve columns", style={'textAlign': 'center'}),

            html.Br(style={'height': '75px'}),
            html.Div([  
                html.H4('The effect of changing the PV power and battery capacity on the optimization algorithm:'),
                ], className="twelve columns"),

            html.Div([
                html.H4('Sensitivity Analysis on Savings:'),
                html.Img(
                    src=b64_image('src/Optimisação/Savings_Sensitivity.png'),
                    style={'width': '50%', 'height': '50%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),
            html.Div([
                html.H4('Sensitivity Analysis on NPV:'),
                html.Img(
                    src=b64_image('src/Optimisação/NPV_Sensitivity.png'),
                    style={'width': '50%', 'height': '50%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),
            html.Div([
                html.H4('Sensitivity Analysis on DPBP:'),
                html.Img(
                    src=b64_image('src/Optimisação/DPBP_Sensitivity.png'),
                    style={'width': '50%', 'height': '50%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),
        ])
    elif dropdown == 'Team':
        return html.Div([
            html.Div([
                html.Img(
                    src=b64_image('src/INESCTECLogotipo_CORPositivo_RGB.jpg'),
                    style={'width': '25%', 'height': '25%'}
                ),
            ], className="twelve columns", style={'textAlign': 'center'}),

            html.H4('Project developed by:',style={'font-weight': 'bold'}),
            html.H5('Team Coordinator:',style={'font-weight': 'bold'}),
            html.H5('Hermano Bernardo - hermano.bernardo@inesctec.pt'),
            html.H5('Power Demand Forecasting:',style={'font-weight': 'bold'}),
            html.H5('Bruno Palley - bruno.palley@inesctec.pt'),
            html.H5('PV Generation Forecasting:',style={'font-weight': 'bold'}),
            html.H5('Xavier Godinho - xavier.godinho@inesctec.pt'),
            html.H5('Energy Optimisation Algorithm:',style={'font-weight': 'bold'}),
            html.H5('Tomás Barosa Santos - tomas.b.santos@inesctec.pt'),
            
        ])


if __name__ == '__main__':
    #app.run(dev_tools_hot_reload=False, debug=True)
    app.run()