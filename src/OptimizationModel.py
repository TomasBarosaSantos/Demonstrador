import numpy as np
import pandas as pd
import pyomo
from pyomo.environ import *
import datetime
import cplex # type: ignore

##Define Optimizarion Problem Fuction
#Inputs: 1-Photovoltaic Power in kW, 2-Installed Battery Capacity in kWh, 3-Installed Thermal Capacity in kWh, 4-Dataframe including necessary data

def Optimization(PV_InstalledPower: int = 1200, BATT_InstalledCapacity: int = 400, Thermal_Installed:int = 0, df_Opti = pd.DataFrame(), Use_LCOS:bool = True, Sell_Excess:bool = False):
     
    """
    Optimization model that solves the problem of minimizing the cost of energy consumption for a building with photovoltaic panels, electrical battery and thermal storage.\n

    Inputs:\n
    \t PV_InstalledPower - Installed Power of the Photovoltaic Panels in kW\n
    \t BATT_InstalledCapacity - Installed Capacity of the Electrical Battery in kWh\n
    \t Thermal_Installed - Installed Capacity of the Thermal Storage in kWh\n
    \t df_Opti - Dataframe containing the necessary data for the optimization problem\n

    Outputs:\n
    \t df_OptimalElectric - Dataframe containing the optimal solution for the electrical energy\n
    \t cost_model - Optimal cost of the solution\n
    \t df_OptimalThermal - Dataframe containing the optimal solution for the thermal energy\n
    \t model_results - Pyomo model with the results of the optimization\n
    \t results - Results of the optimization problem\n

    """
    ##Check if the Dataframe is empty
    if df_Opti.empty:
        return print("Dataframe is empty") #Return error message if the Dataframe is empty
    
    ##Solar Panel Calculations
    #
    Pot_Panels = 400.0 #Panel Power in Wp
    Eff_Panel = 0.193 #Panel Efficiency    
    Eff_Inver = 0.98 #Inverter Efficiency
    PV_Area = 2.0 #In square meters
    df_Opti.loc[:,'PV Power (kW)'] = df_Opti.loc[:,'Irradiance (W/m2)']/1000 * Eff_Inver * PV_Area * Eff_Panel #Calculate production for 1 panel

    ##Electrical Battery Characteristics
    #
    BATT_CRate = 1 #Charge and Discharge Rate
    BATT_MinCharge = 0.2 #Minimun Charge Allowed in Battery
    BATT_MaxCharge = 0.8 #Maximun Charge Allowed in Battery
    BATT_InitialCharge = 0.2 #Initial Charge in Battery
    BATT_ChargeEff = 0.96 #Charging Efficiency
    BATT_DischargeEff = 0.96 #Discharging Efficiency
    DOC_LossHour = ((0.30/10)/365)/24 #Loss in Capacity (percentage) per hour of life of the Battery
    MAXBatt_Capacity = BATT_InstalledCapacity

    ##Thermal Storage Characteristics
    #
    Thermal_CRate = 0.5 #Charge and Discharge Rate
    Thermal_MinCharge = 0 #Minimun Charge Allowed in Thermal Storage
    Thermal_MaxCharge = 0.9 #Maximun Charge Allowed in Thermal Storage
    Thermal_InitialCharge = 0.2 #Initial Charge in Thermal Storage
    Thermal_Efficiency = 1 #Thermal Efficiency (Preliminary)

    ##LCOE of Battery
    #
    if BATT_InstalledCapacity != 0 and Use_LCOS == True:
        BATT_Cost = 180 * BATT_InstalledCapacity + 90*BATT_CRate*BATT_InstalledCapacity #Investment cost per kWh
        LCOS_BATT = BATT_Cost/(3000*BATT_InstalledCapacity) #Levelized Cost of Storage (LCOS) of the energy provided by battery
    else:
        BATT_Cost = 0
        LCOS_BATT = 0
    
    ##Self-Discharge
    #
    self_discharge_eta = 0.03/30/24 # Self-discharge % per hour

    ##Sell Excess Energy
    #
    if Sell_Excess == True:
        
        Sale_Price = (df_Opti['Cost (Eur/kW)'] - df_Opti['Cost (Eur/kW)']/10) #Price of sale for the Excess Energy from PV

    else:
        Sale_Price = np.zeros(len(df_Opti))#Price of sale for the Excess Energy from PV

    ##Optimization Problem Formulation and solving
    #
    def optimization_problem(PV_Power, BATT_Capacity, Thermal_Capacity):
        
        ##Calculate the photovoltaic production (Preliminary)
        df_Opti.loc[:,'PV Power v2 (kW)'] = df_Opti.loc[:,'PV Power (kW)'] * PV_Power/(Pot_Panels/1000)

        #Create a list of zeros to use when there isnt a thermal demand
        listofzeros =[]
        for i in range(len(df_Opti)):
            listofzeros.append(0)

        listofDOC =[]
        for i in range(len(df_Opti)):
            #listofDOC.append(MAXBatt_Capacity - MAXBatt_Capacity*i*DOC_LossHour)
            listofDOC.append(MAXBatt_Capacity*(1-DOC_LossHour*i))

        ##Optimization Problem Formulation

        ##Model Creation 
        model = ConcreteModel()

        ##Defining Parameters for the model (Input data)
        model.Period = RangeSet(0, len(df_Opti)-1) #Timesteps of optimization problem
        model.BuyPrice = Param(model.Period, initialize=list(df_Opti['Cost (Eur/kW)'] + df_Opti['Access Tariff (Eur/kWh)'] + df_Opti['PHP Tariff (Eur/kWh)']), within=Any, doc='Buy Price (Eur/kWh)')
        #model.BuyPrice = Param(model.Period, initialize=list(df_Opti['Cost (Eur/kW)']), within=Any, doc='Buy Price (Eur/kWh)')

        #Electric and Thermal Demand Parameters
        if Thermal_Installed == 0.0:
            model.ElectricDemand = Param(model.Period, initialize=list(df_Opti['Electric Power (kW)'] + df_Opti['Thermal Power (kW)']), within=Any, doc='Elecric Energy (kW)')
            model.ThermalDemand = Param(model.Period, initialize=listofzeros, within=Any, doc='Thermal Energy (kW)')
        else:
            model.ElectricDemand = Param(model.Period, initialize=list(df_Opti['Electric Power (kW)']), within=Any, doc='Elecric Energy (kW)')
            model.ThermalDemand = Param(model.Period, initialize=list(df_Opti['Thermal Power (kW)']), within=Any, doc='Thermal Energy (kW)')               

        #PV Generation Parameters                                                                                  
        model.PVGen = Param(model.Period, initialize=list(df_Opti['PV Power v2 (kW)']), within= NonNegativeReals, doc='PV Generated')

        ##Defining Decision Variables
        #Grid Consumption Variables
        model.GridElectric = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='Electric Energy from the Grid for building electric demand (kWh)')
        model.GridThermal = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='Electric Energy from the Grid for thermal demand (kWh)')

        #PV Panels Consumption Variables
        model.PVExc = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='Excess PV Energy(kWh)')
        model.PVE = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='PV Energy for electric demand (kWh)')
        model.PVT = Var(model.Period,bounds=(0.0,None), domain=NonNegativeReals, doc='PV Energy for thermal demand (kWh)')

        #Storage Systems Variables (Electric Battery and Thermal Storage)
        if BATT_Capacity == 0.0:
            model.BATTCharge = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='Energy used for Charging Electric Battery')
            model.BATTDischarge = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='Energy used for Discharging Electric Battery')
            model.SOCBATT = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='State of Charge of Electric Battery')
            model.DOC = Var(model.Period, initialize=listofDOC, domain=Reals, doc='Depht of Charge of Electric Battery')
        else:
            model.BATTCharge = Var(model.Period,bounds=(0.0,BATT_CRate*BATT_Capacity), domain=Reals, doc='Energy used for Charging Electric Battery')
            model.BATTDischarge =Var(model.Period,bounds=(0.0,BATT_CRate*BATT_Capacity), domain=Reals, doc='Energy used for Discharging Electric Battery')
            model.SOCBATT = Var(model.Period,bounds=(BATT_MinCharge*BATT_Capacity,BATT_MaxCharge*BATT_Capacity), domain=Reals, doc='State of Charge of Battery')
            model.DOC = Param(model.Period, initialize=listofDOC, domain=Reals, doc='Depht of Charge of Electric Battery')

        if Thermal_Installed == 0.0:
            model.ThermalCharge = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='Energy used for Charging Thermal Storage')
            model.ThermalDischarge = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='Energy used for Discharging Thermal Storage')
            model.SOCThermal = Var(model.Period,bounds=(0.0,0.0), domain=Reals, doc='State of Charge of Thermal Storage')
        else:
            model.ThermalCharge = Var(model.Period,bounds=(0.0,Thermal_CRate*Thermal_Installed), domain=Reals, doc='Energy used for Charging Thermal Storage')
            model.ThermalDischarge = Var(model.Period,bounds=(0.0,Thermal_CRate*Thermal_Installed), domain=Reals, doc='Energy used for Discharging Thermal Storage')
            model.SOCThermal = Var(model.Period,bounds=(Thermal_MinCharge*Thermal_Installed,Thermal_MaxCharge*Thermal_Installed), domain=Reals, doc='State of Charge of Thermal Storage')


        #Binary Variable - Charge (1)/Discharge(0)
        model.BATTBinary = Var(model.Period,bounds=(0,1), domain=Binary, doc='Binary variable that dictates charging and discharge of Electric Battery')
        model.ThermalBinary = Var(model.Period,bounds=(0,1), domain=Binary, doc='Binary variable that dictates charging and discharge Thermal Storage')

        #Defining objective fuction
        def obj_rule(model):
                
            return sum((model.GridElectric[t] + model.GridThermal[t]) * model.BuyPrice[t] + (model.BATTDischarge[t])*LCOS_BATT - model.PVExc[t]*Sale_Price[t] for t in model.Period)


        model.obj = Objective(rule=obj_rule, sense=minimize)


        ##Defining Constraints
        #State of Charge of Battery constraints
        def SOC_BATT(model,t):
                if t == 0:
                    return (model.SOCBATT[t] == BATT_InitialCharge*BATT_Capacity*(1-self_discharge_eta) + (model.BATTCharge[t]*BATT_ChargeEff - model.BATTDischarge[t]/BATT_DischargeEff))
                else:
                    return (model.SOCBATT[t] == model.SOCBATT[t-1]*(1-self_discharge_eta) + (model.BATTCharge[t]*BATT_ChargeEff - model.BATTDischarge[t]/BATT_DischargeEff))

        model.C1 = Constraint(model.Period, rule= SOC_BATT)

        #State of Charge of thermal storage constraints
        def SOC_Thermal(model,t):
                if t == 0:
                    return (model.SOCThermal[t] == Thermal_InitialCharge*Thermal_Capacity + (model.ThermalCharge[t]*Thermal_Efficiency  - model.ThermalDischarge[t]/Thermal_Efficiency))
                else:
                    return (model.SOCThermal[t] == model.SOCThermal[t-1] + (model.ThermalCharge[t]*Thermal_Efficiency  - model.ThermalDischarge[t]/Thermal_Efficiency ))

        model.C2 = Constraint(model.Period, rule= SOC_Thermal) 

        model.C3 = ConstraintList()
        model.C4 = ConstraintList()
        model.C5 = ConstraintList()
        model.C6 = ConstraintList()
        model.C7 = ConstraintList()
        model.C8 = ConstraintList()
        model.C9 = ConstraintList()
        model.C10 = ConstraintList()
        model.C11 = ConstraintList()

        #Photovoltaic Production Constraint
        for t in model.Period:
            
            model.C3.add(model.PVGen[t] == model.PVExc[t] + model.PVE[t] + model.PVT[t])

        #Electric Demand Constraint
        for t in model.Period:
                
            model.C4.add(model.ElectricDemand[t] == model.GridElectric[t] + model.PVE[t] + (model.BATTDischarge[t] - model.BATTCharge[t]))

        #Thermal Demand Constraint
        for t in model.Period:
                
            model.C5.add(model.ThermalDemand[t] == model.GridThermal[t] + model.PVT[t] + (model.ThermalDischarge[t] - model.ThermalCharge[t]))

        #Excess Photovoltaic Constraint
        for t in model.Period:
            
            model.C6.add(model.PVExc[t]<= model.PVGen[t])

        #Battery Charging Constraint
        for t in model.Period:
             
             model.C7.add(model.BATTCharge[t] <= model.BATTBinary[t] * BATT_CRate*BATT_Capacity)

        #Battery Discharging Constraint
        for t in model.Period:
             
             model.C8.add(model.BATTDischarge[t] <= (1 - model.BATTBinary[t]) * BATT_CRate*BATT_Capacity)

        #Thermal Charging Constraint
        for t in model.Period:
             
             model.C9.add(model.ThermalCharge[t] <= model.ThermalBinary[t] * Thermal_CRate*Thermal_Capacity)

        #Thermal Discharging Constraint
        for t in model.Period:
             
             model.C10.add(model.ThermalDischarge[t] <= (1 - model.ThermalBinary[t]) * Thermal_CRate*Thermal_Capacity)

        for t in model.Period:
            
            model.C11.add(model.SOCBATT[t] <= model.DOC[t])

        #Solving the problem
        opt = SolverFactory('cplex') #Choosing solver
        #results = opt.solve(model, tee=True, logfile = "cplex.log",) #Solve problem
        results = opt.solve(model) #Solve problem
        results_optimal = format(value(model.obj), ".2f") #Get Optimal Cost


        return results_optimal, model, results #Output Variables: 1-Optimal Cost, 2-Solved Model, 3-Results
    
        
    cost_model, model, results = optimization_problem(PV_InstalledPower,BATT_InstalledCapacity,Thermal_Installed) #Calling optimization problem for the given input data


    ##Graphic Representation of Decision Variables

    #Demand Data 
    if Thermal_Installed == 0.0:
        electric_demand = df_Opti['Electric Power (kW)'] + df_Opti['Thermal Power (kW)']
    else:
        electric_demand = df_Opti['Electric Power (kW)']
        thermal_demand =  df_Opti['Thermal Power (kW)']

    #PV Production Data
    generation = [value(model.PVGen[key]) for key in model.PVGen]

    #Electric Decision Variables
    optimal_GridElectric = [value(model.GridElectric[key]) for key in model.GridElectric] 
    optimal_PVExc = [value(model.PVExc[key]) for key in model.PVExc]
    optimal_PVE = [value(model.PVE[key]) for key in model.PVE]

    #Dataframe of Electrical Solution
    df_OptimalElectric = pd.DataFrame({"Power Demand [kW]":electric_demand,
                                        "PV Power Generation [kW]": generation,
                                        "Grid Electric Consumption [kW]":optimal_GridElectric,
                                        "PV used for Energy Demand [kW]":optimal_PVE,
                                        "PV Excess production [kW]":optimal_PVExc
                                        })
    
    df_OptimalElectric.reset_index(inplace=True, drop=False)

    #Electric Battery Decision Variables
    if BATT_InstalledCapacity != 0.0:
        optimal_SOC = [value(model.SOCBATT[key]) for key in model.SOCBATT] 
        optimal_BATTCharge = [value(model.BATTCharge[key]) for key in model.BATTCharge] 
        optimal_BATTDischarge = [value(model.BATTDischarge[key]) for key in model.BATTDischarge] 
        optimal_BinaryBATT = [value(model.BATTBinary[key]) for key in model.BATTBinary] 

        #Dataframe of Battery Solution
        df_OptimalBATT = pd.DataFrame({"Battery SOC [kW]":optimal_SOC,
                                        "Electrical Energy Charged [kW]":optimal_BATTCharge,
                                        "Electrical Energy Discharged [kW]":optimal_BATTDischarge,
                                        "Binary Battery Value " : optimal_BinaryBATT
                                        })

        #Join Electric Dataframe with Battery Dataframe
        df_OptimalElectric = df_OptimalElectric.join(df_OptimalBATT) 

    #Thermal Storage Decision Variables
    if Thermal_Installed != 0.0:
        optimal_GridThermal = [value(model.GridThermal[key]) for key in model.GridThermal] 
        optimal_PVT = [value(model.PVT[key]) for key in model.PVT]
        optimal_SOCThermal = [value(model.SOCThermal[key]) for key in model.SOCThermal] 
        optimal_ThermalCharge = [value(model.ThermalCharge[key]) for key in model.ThermalCharge]
        optimal_ThermalDischarge = [value(model.ThermalDischarge[key]) for key in model.ThermalDischarge]
        optimal_BinaryThermal = [value(model.ThermalBinary[key]) for key in model.ThermalBinary] 

        #Dataframe of Thermal Solution
        df_OptimalThermal = pd.DataFrame({"Electric Thermal Demand (kW)":thermal_demand,
                                            "PV Generation (kW)": generation,
                                            "Grid Electric Cooling Consumption (kW)":optimal_GridThermal,
                                            "PV used for Electric Cooling Demand (kW)":optimal_PVT,
                                            "SOC Thermal Storage (kW)":optimal_SOCThermal,
                                            "Charged Energy Thermal Storage (kW)":optimal_ThermalCharge,
                                            "Disharged Energy Thermal Storage (kW)": optimal_ThermalDischarge,
                                            "Binary Theraml Value " : optimal_BinaryThermal
                                            })
    else:
         df_OptimalThermal = pd.DataFrame()
         
    #Set index as Datetime
    df_OptimalElectric.set_index('Datetime',inplace=True,drop=True)
    
    model_results = model

    #Clear model
    model.clear

    return df_OptimalElectric, cost_model, df_OptimalThermal, model_results, results