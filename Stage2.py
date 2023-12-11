import pandas as pd
import numpy as np
import pulp as pl

from solarenergyproduction import *
from fetchprices import *

solar_production = solar_energy_production()["Amount(MW)"].values
intraday_prices = fetch_day_ahead_prices()["Price(EUR/MW)"].values

tolerance = 1e-5  # You may adjust the tolerance level as necessary
big_M = 1000
data_points = 24  # The number of hours in the planning horizon, typically 24 for a single day.
time_frame = range(1,data_points)
grid_capacity_limit_MW = 50


# Battery Parameters
battery_capacity_MW = 10  # The maximum amount of energy that the battery can store, measured in Megawatts (MW).
battery_c_value = 0.5  #  Represents the battery's discharge rate. A C-value of 0.5 means the battery can be dis/charged at a rate that would deplete its capacity in 2 hours.
battery_efficiency = 0.9  # The efficiency of the battery in storing and discharging energy. A value of 0.9 (or 90%) means that 10% of energy is lost during charging and discharging cycles.
battery_discharge_limit = 0.8 * battery_capacity_MW # The maximum amount of energy that can be discharged from the battery at any given time, calculated as a percentage of the battery's capacity.
battery_lifetime_cycles = 5000  # The total number of charge-discharge cycles the battery can undergo before its capacity significantly degrades.
battery_SOC = 0 # Initial state of charge 

def check_constraints(intraday_vars, battery_charge_vars, battery_discharge_vars, battery_SOC_vars, battery_action):
    
    # Check 50 MW grid buy/sell constraint
    for t in time_frame:
        if not pl.value(intraday_vars[t-1][t]) <= grid_capacity_limit_MW:
            print(f"Grid buy/sell constraint violated at t={t}")
            
    # Check the energy balance constraints
    for t in time_frame:
        if not (pl.value(intraday_vars[t-1][t] + pl.value(battery_charge_vars[t]) <= solar_production[t-1] + battery_discharge_vars[t]*battery_efficiency) + tolerance):
            print(f"Energy balance constraint violated at t={t}")
            
    # Check battery operational constraints
    for t in range(data_points):
        if not (pl.value(battery_charge_vars[t]) <= battery_capacity_MW * battery_c_value):
            print(f"Battery charge limit constraint violated at t={t}")
        if not (pl.value(battery_discharge_vars[t]) <= battery_discharge_limit * battery_c_value):
            print(f"Battery discharge limit constraint violated at t={t}")
        if not (pl.value(battery_SOC_vars[t]) - pl.value(battery_discharge_vars[t]) >= 0):
            print(f"Battery SOC discharge amount constraint violated at t={t}")
        if not (pl.value(battery_SOC_vars[t]) + pl.value(battery_charge_vars[t]) <= battery_capacity_MW + tolerance):
            print(f"Battery SOC upper limit constraint violated at t={t}")
        if not (pl.value(battery_charge_vars[t]) <= big_M * (1 - pl.value(battery_action[t]))):
            print(f"Battery action charge constraint violated at t={t}")
        if not (pl.value(battery_discharge_vars[t]) <= big_M * pl.value(battery_action[t])):
            print(f"Battery action discharge constraint violated at t={t}")

    # Check constraints for state of charge
    if not (pl.value(battery_SOC_vars[0]) == battery_SOC + pl.value(battery_charge_vars[0]) - pl.value(battery_discharge_vars[0])):
        print(f"Initial state of charge constraint violated")
    for t in time_frame:
        if not (abs(pl.value(battery_SOC_vars[t]) - pl.value(battery_SOC_vars[t-1]) - pl.value(battery_charge_vars[t]) + pl.value(battery_discharge_vars[t])) < tolerance):
            print(f"State of charge constraint violated at t={t}")

    # Check constraints to ensure state of charge is within bounds
    for t in range(data_points):
        if not (pl.value(battery_SOC_vars[t]) <= battery_capacity_MW):
            print(f"Upper bound SOC constraint violated at t={t}")
        if not (pl.value(battery_SOC_vars[t]) >= 0):
            print(f"Lower bound SOC constraint violated at t={t}")

# Call this function with the appropriate arguments after solving the optimization problem



def revenue_generated():

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    intraday_vars = pl.LpVariable.dicts("Intraday", (range(data_points), range(data_points)), lowBound=0, cat="Continuous") # The energy (MWh) sold in the intraday market. Each element (t1, t2) denotes energy sold at hour t1 for delivery at hour t2.
   
    battery_charge_vars = pl.LpVariable.dicts("Battery_Charge", range(data_points), lowBound=0, cat="Continuous") # This variable represents the amount of energy (in MW) that is charged into the battery at each hour t within the planning horizon (defined by num_hours).
    battery_discharge_vars = pl.LpVariable.dicts("Battery_Discharge", range(data_points), lowBound=0, cat="Continuous") # Represents the amount of energy (in MW) discharged from the battery at each hour t within the planning horizon.
    battery_action = pl.LpVariable.dicts("Battery_Action", range(data_points), cat="Binary") # Prevent simultaneous charging and discharging of the battery within the same hour using a Binary variable.
    battery_SOC_vars = pl.LpVariable.dicts("Battery_SOC", range(data_points), lowBound=0, upBound=battery_capacity_MW, cat="Continuous") # Decision variable for state of charge of the battery


    # Set the objective function
    problem += (
        pl.lpSum([intraday_vars[t-1][t] * intraday_prices[t] for t in time_frame])
    ) 

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            intraday_vars[t-1][t]
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            intraday_vars[t-1][t] + battery_charge_vars[t]
            <= solar_production[t-1] + battery_discharge_vars[t]*battery_efficiency
        )


    # Battery operational constraints
    for t in range(data_points):
        problem += battery_charge_vars[t] <= battery_capacity_MW * battery_c_value
        problem += battery_charge_vars[0] == 0

        problem += battery_discharge_vars[t] <= battery_discharge_limit * battery_c_value
        problem += battery_discharge_vars[0] == 0

        problem += battery_SOC_vars[t] - battery_discharge_vars[t] >= 0
        problem += battery_SOC_vars[t] + battery_charge_vars[t] <= battery_capacity_MW

        problem += battery_charge_vars[t] <= big_M * (1 - battery_action[t])
        problem += battery_discharge_vars[t] <= big_M * battery_action[t]


    # Constraints for state of charge
    problem += battery_SOC_vars[0] == battery_SOC + battery_charge_vars[0] - battery_discharge_vars[0]
    for t in time_frame:
        problem += battery_SOC_vars[t] == battery_SOC_vars[t-1] + battery_charge_vars[t] - battery_discharge_vars[t]

    # Constraints to ensure state of charge is within bounds
    for t in range(data_points):
        problem += battery_SOC_vars[t] <= battery_capacity_MW
        problem += battery_SOC_vars[t] >= 0


    # Solve the problem and print the results
    problem.solve()
    print("Status: ", pl.LpStatus[problem.status])
    print("Optimal revenue: ", pl.value(problem.objective))

    # Print the selling decisions
    print("\nSelling decisions:")
    for t in time_frame:
        intraday_val = pl.value(intraday_vars[t-1][t])
        if intraday_val > 0:
            print(f"Hour {t-1}: Sell {intraday_val:.2f} MWh at intraday price of hour {t}")

    # Print the battery charging decisions
    print("\nBattery charging decisions:")
    for t in range(data_points):
        battery_charge_val = pl.value(battery_charge_vars[t])
        if battery_charge_val > 0:
            print(f"Hour {t}: Charge the battery with {battery_charge_val:.2f} MWh")

    # Print the battery discharging decisions
    print("\nBattery discharging decisions:")
    for t in range(data_points):
        battery_discharge_val = pl.value(battery_discharge_vars[t])
        if battery_discharge_val > 0:
            print(f"Hour {t}: Discharge the battery: {battery_discharge_val:.2f} MWh")

    # Print State of Charge in battery for checking
    print("\nBattery State of Charge:")
    for t in range(data_points):
        battery_SOC_val = pl.value(battery_SOC_vars[t])
        if battery_SOC_val > 0:
            print(f"Hour {t}: Energy in Battery: {battery_SOC_val:.2f} MWh")

    check_constraints(intraday_vars, battery_charge_vars, battery_discharge_vars, battery_SOC_vars, battery_action)


revenue_generated()