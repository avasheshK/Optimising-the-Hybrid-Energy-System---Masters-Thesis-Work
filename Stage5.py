import pandas as pd
import numpy as np
import pulp as pl

from solarproductionforecast import *
from fetchprices import *

solar_production, forecasted_production = solar_energy_production_forecast() 
solar_production = solar_production["Amount(MW)"].values
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


# Hydrogen Parameters
hydrogen_storage_capacity_kg = 1250 # The total amount of hydrogen (in kilograms) that can be stored.
initial_hydrogen_storage = 0 # Initial amount of hydrogen in storage

electrolyser_size_MW = 6 # The capacity of the electrolyser, measured in Megawatts.
electrolyser_conversion_rate = 0.03333 # Conversion rate as a separate constant measured in MWh/kg.
electrolyser_efficiency = 0.6 # The efficiency of the electrolyser as a variable input.
electrolyser_conversion_efficiency = electrolyser_efficiency/electrolyser_conversion_rate # The formula for calculating electrolyser conversion efficiency in kg/MWh.

hydrogen_fuel_cell_size_MW = 10  # The capacity of the hydrogen fuel cell, measured in Megawatts.    
fuel_cell_conversion_efficiency = 0.017 # The amount of electrical energy produced from 1 kg of hydrogen, measured in MWh/kg.

hydrogen_demand_flat = 100  # The constant amount of hydrogen demanded
hydrogen_market_price_EUR_per_kg = 6 # The selling price of hydrogen per kilogram
hydrogen_cost_price_EUR_per_kg = 6 # The cost price of hydrogen per kilogram


def check_constraints(intraday_vars, grid_buy_vars, battery_charge_vars, battery_discharge_vars, battery_SOC_vars, battery_action, hydrogen_storage_vars, hydrogen_sold_vars, allow_hydrogen_sale_vars, electricity_to_hydrogen_vars):
    
    # Check 50 MW grid buy/sell constraint
    for t in time_frame:
        if not pl.value(intraday_vars[t-1][t]) + pl.value(grid_buy_vars[t]) <= grid_capacity_limit_MW:
            print(f"Grid buy/sell constraint violated at t={t}")
            
    # Check the energy balance constraints
    for t in time_frame:
        if not (pl.value(intraday_vars[t-1][t] + pl.value(battery_charge_vars[t]) <= solar_production[t-1] + battery_discharge_vars[t]*battery_efficiency) + grid_buy_vars[t] + tolerance):
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

    # New Hydrogen Constraints Checking Logic
    # Check constraints for hydrogen conversion and storage dynamics
    for t in time_frame:
        if not (pl.value(hydrogen_storage_vars[t]) <= hydrogen_storage_capacity_kg):
            print(f"Upper bound Hydrogen storage constraint violated at t={t}")
        if not (pl.value(hydrogen_storage_vars[t]) >= 0):
            print(f"Lower bound Hydrogen storage constraint violated at t={t}")
        if not (pl.value(electricity_to_hydrogen_vars[t]) <= electrolyser_size_MW):
            print(f"Electrolyser capacity constraint violated at t={t}")

    # Check constraints to ensure that the sum of hydrogen sold and hydrogen used to meet demand does not exceed the hydrogen stored
    for t in range(data_points):
        if not (pl.value(hydrogen_storage_vars[t]) >= hydrogen_demand_flat):
            print(f"Hydrogen demand constraint violated at t={t}")
        if not (pl.value(hydrogen_sold_vars[t]) <= 1500 * pl.value(allow_hydrogen_sale_vars[t])):
            print(f"Hydrogen sale limit constraint violated at t={t}")

    # Check constraints for the limited number of times hydrogen can be sold in a day
    if not (pl.value(pl.lpSum(allow_hydrogen_sale_vars[t] for t in range(data_points))) <= 1):
        print(f"Constraint on the number of times hydrogen can be sold in a day is violated")

# Call this function with the appropriate arguments after solving the optimization problem



def revenue_generated():

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    intraday_vars = pl.LpVariable.dicts("Intraday", (range(data_points), range(data_points)), lowBound=0, cat="Continuous") # The energy (MWh) sold in the intraday market. Each element (t1, t2) denotes energy sold at hour t1 for delivery at hour t2.
    grid_buy_vars = pl.LpVariable.dicts("Grid_Buy", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) bought from the grid at each hour t.

    battery_charge_vars = pl.LpVariable.dicts("Battery_Charge", range(data_points), lowBound=0, cat="Continuous") # This variable represents the amount of energy (in MW) that is charged into the battery at each hour t within the planning horizon (defined by num_hours).
    battery_discharge_vars = pl.LpVariable.dicts("Battery_Discharge", range(data_points), lowBound=0, cat="Continuous") # Represents the amount of energy (in MW) discharged from the battery at each hour t within the planning horizon.
    battery_action = pl.LpVariable.dicts("Battery_Action", range(data_points), cat="Binary") # Prevent simultaneous charging and discharging of the battery within the same hour using a Binary variable.
    battery_SOC_vars = pl.LpVariable.dicts("Battery_SOC", range(data_points), lowBound=0, upBound=battery_capacity_MW, cat="Continuous") # Decision variable for state of charge of the battery

    # Decision variables for hydrogen
    hydrogen_storage_vars = pl.LpVariable.dicts("Hydrogen_Storage", range(data_points), lowBound=0, cat="Continuous") # Represents the amount of hydrogen (in kg) stored at each hour t within the planning horizon.
    electricity_to_hydrogen_vars = pl.LpVariable.dicts("Electricity_To_Hydrogen", range(data_points), lowBound=0, cat="Continuous") #  Total energy converted to hydrogen at each hour t in (MW)
    allow_hydrogen_sale_vars = pl.LpVariable.dicts("Hydrogen_Sale_Limit", range(data_points), cat="Binary")
    hydrogen_sold_vars = pl.LpVariable.dicts("Hydrogen_Sold", range(data_points), lowBound=0, cat="Continuous") #The amount of hydrogen sold (in kg)
    hydrogen_to_electricity_vars = pl.LpVariable.dicts("Hydrogen_To_Electricity", range(data_points), lowBound=0, cat="Continuous") # The hydrogen (in kg) converted to energy at each hour t.
    hydrogen_buy_vars = pl.LpVariable.dicts("Hydrogen_Bought", range(data_points), lowBound=0, cat="Continuous") # Amount of Hydrogen bought from the Hydrogen Market in kg.

    # Set the objective function
    problem += (
        pl.lpSum([intraday_vars[t-1][t] * intraday_prices[t] for t in time_frame]) - pl.lpSum([grid_buy_vars[t] * intraday_prices[t]] for t in range(data_points)) +  pl.lpSum(hydrogen_sold_vars[t] * hydrogen_market_price_EUR_per_kg - hydrogen_buy_vars[t] * hydrogen_cost_price_EUR_per_kg for t in range(data_points))
    ) 

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            intraday_vars[t-1][t] + grid_buy_vars[t]
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            intraday_vars[t-1][t] + battery_charge_vars[t] + electricity_to_hydrogen_vars[t]
            <= solar_production[t-1] + battery_discharge_vars[t]*battery_efficiency + grid_buy_vars[t] + hydrogen_to_electricity_vars[t]*fuel_cell_conversion_efficiency
        )


    # Battery operational constraints
    for t in range(data_points):
        problem += battery_charge_vars[t] <= battery_capacity_MW * battery_c_value

        problem += battery_discharge_vars[t] <= battery_discharge_limit * battery_c_value

        problem += battery_SOC_vars[t] - battery_discharge_vars[t] >= 0
        problem += battery_SOC_vars[t] + battery_charge_vars[t] <= battery_capacity_MW

        problem += battery_charge_vars[t] <= big_M * (1 - battery_action[t])
        problem += battery_discharge_vars[t] <= big_M * battery_action[t]

    # Set the initial conditions outside the loop
    problem += battery_charge_vars[0] == 0
    problem += battery_discharge_vars[0] == 0

    # Constraints for state of charge
    problem += battery_SOC_vars[0] == battery_SOC + battery_charge_vars[0] - battery_discharge_vars[0]
    for t in time_frame:
        problem += battery_SOC_vars[t] == battery_SOC_vars[t-1] + battery_charge_vars[t] - battery_discharge_vars[t]

    # Constraints to ensure state of charge is within bounds
    for t in range(data_points):
        problem += battery_SOC_vars[t] <= battery_capacity_MW
        problem += battery_SOC_vars[t] >= 0


    # Hydrogen operational constraints
    for t in range(data_points):
        problem += electricity_to_hydrogen_vars[t] <= electrolyser_size_MW # Amount of electricity we convert to hydrogen should always be less than the size of the electrolyser
        problem += hydrogen_to_electricity_vars[t] * fuel_cell_conversion_efficiency <= hydrogen_fuel_cell_size_MW  # Amount of electricity we get from hydrogen should always be less than the size of the Fuel cell

    # Constraints for hydrogen storage dynamics
    problem += hydrogen_storage_vars[0] == initial_hydrogen_storage - hydrogen_to_electricity_vars[0] + electricity_to_hydrogen_vars[0] * electrolyser_conversion_efficiency - hydrogen_sold_vars[0] + hydrogen_buy_vars[0]
    for t in time_frame:
        problem += hydrogen_storage_vars[t] == hydrogen_storage_vars[t-1] - hydrogen_to_electricity_vars[t] + electricity_to_hydrogen_vars[t] * electrolyser_conversion_efficiency - hydrogen_sold_vars[t] + hydrogen_buy_vars[t]

    # Constraints to ensure hydrogen storage is within bounds
    for t in range(data_points):
        problem += hydrogen_storage_vars[t] <= hydrogen_storage_capacity_kg
        problem += hydrogen_storage_vars[t] >= 0

    # Ensure that the sum of hydrogen sold and hydrogen used to meet demand does not exceed the hydrogen stored
    for t in range(data_points):
        problem += hydrogen_storage_vars[t] >= hydrogen_demand_flat
        problem += hydrogen_sold_vars[t] <= 1500 * allow_hydrogen_sale_vars[t]
        problem += hydrogen_to_electricity_vars[t] <= 1500 * allow_hydrogen_sale_vars[t]
        problem += electricity_to_hydrogen_vars[t] <= big_M*(1-allow_hydrogen_sale_vars[t])
        problem += hydrogen_buy_vars[t] <= big_M*(1-allow_hydrogen_sale_vars[t])


    # Limit the number of times hydrogen can be sold in a day
    problem += pl.lpSum(allow_hydrogen_sale_vars[t] for t in range(data_points)) <= 1

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


    # Print grid buy decisions
    print("\nBuying from the grid:")
    for t in range(data_points):
        grid_buy_val = grid_buy_vars[t].value()
        if grid_buy_val > 0:
            print(f"Buy {grid_buy_val} MWh of electricity from the grid at hour {t}")
        else:
            pass 


    # Print electricity-to-hydrogen conversion decisions
    print("\nElectricity to Hydrogen Conversion decisions:")
    for t in range(data_points):
        electricity_to_hydrogen_val = electricity_to_hydrogen_vars[t].value()
        if electricity_to_hydrogen_val > 0:
            print(f"Convert {electricity_to_hydrogen_val} MWh of electricity to Hydrogen at hour {t}")
        else:
            pass

    # Print amount of hydrogen stored in total every hour
    print("\nHydrogen Storage Indicator:")
    for t in range(data_points):
        hydrogen_storage_val = hydrogen_storage_vars[t].value()
        if hydrogen_storage_val > 0:
            print(f"{hydrogen_storage_val} kg of Hydrogen stored in the container at hour {t}")
        else:
            pass

    # Print amount of hydrogen bought every hour
    print("\nHydrogen Buying decisions:")
    for t in range(data_points):
        hydrogen_buy_val = hydrogen_buy_vars[t].value()
        if hydrogen_buy_val > 0:
            print(f"{hydrogen_buy_val} kg of Hydrogen bought at hour {t}")
        else:
            pass

    # Print amount of hydrogen converted to electricity every hour
    print("\nHydrogen to Electricity Conversion decisions:")
    for t in range(data_points):
        hydrogen_to_electricity_val = hydrogen_to_electricity_vars[t].value()
        if hydrogen_to_electricity_val > 0:
            print(f"{hydrogen_to_electricity_val} kg of Hydrogen converted to {hydrogen_to_electricity_val * fuel_cell_conversion_efficiency} MWh of electricity at hour {t}")
        else:
            pass
    print("\nHydrogen Selling decisions:")
    for t in range(data_points):
        hydrogen_sold_val = hydrogen_sold_vars[t].value()
        if hydrogen_sold_val > 0:
            print(f"{hydrogen_sold_val} kg of Hydrogen at hour {t}")
        else:
            pass

    # Print the total amount of hydrogen sold
    total_hydrogen_sold = sum([hydrogen_sold_vars[t].value() for t in range(data_points)])
    print(f"\nTotal hydrogen sold: {total_hydrogen_sold} kg\n")


    check_constraints(intraday_vars, grid_buy_vars, battery_charge_vars, battery_discharge_vars, battery_SOC_vars, battery_action, hydrogen_storage_vars, hydrogen_sold_vars, allow_hydrogen_sale_vars, electricity_to_hydrogen_vars)


revenue_generated()