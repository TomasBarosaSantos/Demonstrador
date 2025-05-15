import numpy as np
import pandas as pd

##Solar Panel Characteristics
#
Pot_Panels = 400.0 #Panel Power in Wp
Eff_Panel = 0.193 #Panel Efficiency    
Eff_Inver = 0.98 #Inverter Efficiency
PV_Area = 2.0 #In square meters

##Battery Characteristics
#
BATT_CRate = 1 #Charge and Discharge Rate
BATT_MinCharge = 0.2 #Minimun Charge Allowed in Battery
BATT_MaxCharge = 0.8 #Maximun Charge Allowed in Battery
BATT_InitialCharge = 0.2 #Initial Charge in Battery
BATT_ChargeEff = 0.96 #Charging Efficiency
BATT_DischargeEff = 0.96 #Discharging Efficiency
DOC_LossHour = ((0.30/10)/365)/24 #Loss in Capacity (percentage) per hour of life of the Battery


def dumb_PV_Batt(df,PV_Installed,Batt_Installed):
    '''
    Function that simulated the functioning of a PV system with energy storage system without any optimisation\n
    
    Inputs:\n
    \tdf: Pandas DataFrame with the data of the building\n
    \tPV_Installed: Installed Capacity of the PV system in kW\n
    \tBatt_Installed: Installed Capacity of the Battery in kWh\n

    Outputs:\n
    \tdf_Output: Pandas DataFrame with the output of the simulation
    
    '''

    #Create a Dataframe to save the output
    column_names = ['Power Demand [kW]','PV Power Generation [kW]',
                    'Grid Electric Consumption [kW]','PV used for Energy Demand [kW]',
                    'PV Excess production [kW]','Battery SOC [kW]','Electrical Energy Charged [kW]',
                    'Electrical Energy Discharged [kW]']

    df_Output = pd.DataFrame(columns=column_names)

    #Initialize the SOC of the energy storage and the PV Excess
    SOCBatt = BATT_InitialCharge * Batt_Installed
    PV_Excess = 0
    
    df['PV Power V2'] = df['Irradiance (W/m2)']/1000 * Eff_Inver * PV_Area * Eff_Panel * PV_Installed/(Pot_Panels/1000)

    df = df.reset_index()

    for i in range(len(df)):

        #Get the energy load for i-th hour
        Load = df.at[i,'Electric Power (kW)'] + df.at[i,'Thermal Power (kW)']
        Demand = Load
        #Get the generated energy by the PV system for the i-th hour
        PV_Gen = df.at[i,'PV Power V2']

        #Initialize the variables for the energy charged and discharged
        BATT_Charged = 0.0
        BATT_Discharged = 0.0

        #If the Load is greater than the PV Generation
        if Load >= PV_Gen:

            #The load not fulfilled by the PV system is taken from the grid or energy storage system
            Load = Load - PV_Gen
            PV_Excess = 0.0

            #Verify if the energy storage system has enough energy to supply the remaining load
            if SOCBatt > BATT_MinCharge*Batt_Installed and Load > 0.0:

                #Compute the available energy in the energy storage system
                Discharge_Available = (SOCBatt - BATT_MinCharge*Batt_Installed)

                #Compute the energy that will be discharged from the energy storage system
                BATT_Discharged = min(Load/BATT_DischargeEff,Discharge_Available/BATT_DischargeEff)

                #Update the state of charge of the energy storage system
                SOCBatt = SOCBatt - BATT_Discharged

                #Update the remaining load
                Load = Load - min(Load,Discharge_Available)

        else:
            
            #If the PV Generation is greater than the Load, the excess energy is used to charge the energy storage system
            PV_Excess = PV_Gen - Load
            Load = 0.0

            #Verify if the energy storage system has enough capacity to store the excess energy
            if SOCBatt < BATT_MaxCharge*Batt_Installed:
                
                #Compute the available energy that can be stored in the energy storage system
                Charge_Available = (BATT_MaxCharge*Batt_Installed - SOCBatt)

                #Compute the energy that will be stored in the energy storage system
                BATT_Charged = min(PV_Excess,Charge_Available)*BATT_ChargeEff

                #Compute the excess energy that will not be stored
                PV_Excess = max(PV_Excess - min(PV_Excess,Charge_Available),0)

                #Update the state of charge of the energy storage system
                SOCBatt = SOCBatt + BATT_Charged
        
        #Save the variables in a new row in the output dataframe
        df_Output.loc[i] = [Demand, PV_Gen, Load, (PV_Gen - PV_Excess), PV_Excess, SOCBatt, BATT_Charged, BATT_Discharged]

    df_Output['Datetime'] = df['Datetime']
    df_Output = df_Output.set_index('Datetime', drop=True)


    return df_Output