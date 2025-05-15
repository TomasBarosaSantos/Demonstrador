#NPV Computation

def NPV_Computation(Savings, PV_Installed, Cost_PV, BATT_Installed, Cost_BATT, BATT_cycles):

    InitalInvestment = PV_Installed * Cost_PV + BATT_Installed * Cost_BATT
    lifespan = 20
    update_rate = 0.04
    Max_cycles = 3000
    Accumulated_Savings = 0

    for i in range(0,lifespan):
        
        discounted_savings = Savings/((1+update_rate)**i)
        Accumulated_Savings = Accumulated_Savings + discounted_savings

    if BATT_cycles != 0.0:

        if Max_cycles/BATT_cycles < lifespan:

            if Max_cycles/BATT_cycles < 1:
                BATT_Adquision_Year = Max_cycles/BATT_cycles
                
            else:
                BATT_Adquision_Year = round(Max_cycles/BATT_cycles,0)
                
            Adquision_Amount = lifespan/BATT_Adquision_Year

            Accumulated_BATT = 0
            for i in range(1,int(Adquision_Amount)):
                
                discounted_BATT = (BATT_Installed * Cost_BATT)/((1+update_rate)**(i*BATT_Adquision_Year))
                Accumulated_BATT = Accumulated_Savings + discounted_BATT

        else :
            Accumulated_BATT = 0
    else:
        Accumulated_BATT = 0

    NPV = round((-InitalInvestment + Accumulated_Savings - Accumulated_BATT),2)

    return NPV