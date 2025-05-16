import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_axes_aligner import align
import matplotlib.dates as mdates



def plot_results(df,title,battery_capacity):

    if 'PV Sold [kW]' in df:
        df['PV Excess production [kW]'] = df['PV Excess production [kW]'] + df['PV Sold [kW]']
        df.drop(columns=['PV Sold [kW]'], inplace=True)

    df_seaborn = df.copy(deep=False)
    
    df_seaborn = df_seaborn.loc[:,['Grid Electric Consumption [kW]','PV used for Energy Demand [kW]','Battery SOC [kW]','Electrical Energy Charged [kW]','Electrical Energy Discharged [kW]','PV Excess production [kW]']]
    df_seaborn['Electrical Energy Charged [kW]'] = -df_seaborn['Electrical Energy Charged [kW]']

    df_seaborn_axis2 = df_seaborn.loc[:,['Battery SOC [kW]']]
    df_seaborn_axis2['Battery SOC [kW]'] = (df_seaborn['Battery SOC [kW]']/battery_capacity)*100

    x = df_seaborn.index
    y1 = df_seaborn['Grid Electric Consumption [kW]']
    y2 = df_seaborn['PV used for Energy Demand [kW]']
    y5 = df_seaborn['PV Excess production [kW]']
    y3 = df_seaborn['Electrical Energy Charged [kW]']
    y4 = df_seaborn['Electrical Energy Discharged [kW]']

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 7)
    
    ax.set_ylabel('Power [kW]', fontsize=20)

    ax.set_ylim((min(df_seaborn.min()), max(df_seaborn.max())*1.6))
    ax.tick_params(axis='y', labelsize=15)
    ax.xaxis.set_visible(False)

    ax2 = ax.twinx()
    ax2.set_ylabel('Battery SOC [%]', fontsize=20)
    ax2.set_ylim((0,100))
    ax2.tick_params(axis='y', labelsize=15)
    align.yaxes(ax, 0, ax2, 0, 0.22)
    stack = ax.stackplot(x, y4, y2, y1, y5, colors=['#d90166','#ff9408','#3d7afd','#ee1b01'], alpha= 0.75, labels=['Energy Discharged from Battery [kW]','PV Self-Consumption [kW]','Grid-Supplied Energy [kW]','PV Excess Generation [kW]'], linewidth=1)


    line2 = ax2.plot(df_seaborn.index, df_seaborn_axis2, color='darkslategray', label='Battery SOC [%]', linestyle='--', linewidth=1)
    line1 = ax.plot(x, y3, color='green', label='Electrical Energy Charged [kW]', linewidth=1)

    lns = line1+line2 + stack

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 0.01),fontsize='x-large',fancybox=True, shadow=True, ncol=3)

    return fig