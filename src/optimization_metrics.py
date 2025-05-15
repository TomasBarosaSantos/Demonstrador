def test(df_NoBatt,df_Opti_Data):

    Price_Energy_Grid_NoBatt = 0

    price = df_Opti_Data['Cost (Eur/kW)'] + df_Opti_Data['Access Tariff (Eur/kWh)'] + df_Opti_Data['PHP Tariff (Eur/kWh)']
    Price_Energy_Grid_NoBatt += price*df_NoBatt['Grid Electric Consumption [kW]']
    
    return Price_Energy_Grid_NoBatt