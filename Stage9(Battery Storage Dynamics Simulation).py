import pandas as pd
import numpy as np
import pulp as pl
from astral import LocationInfo
from astral.sun import sun
import itertools
from solarproductionforecast import *
from fetchprices import *
from DataAnalysis import *

from datetime import timedelta

forecast_df, actual_df, df = solar_energy_production_forecast()

solar_production_forecast = forecast_df["Amount(MW)"].values
solar_production_actual = actual_df["Amount(MW)"].values
solar_historical_data = df["Amount(MW)"].values
day_ahead_spot_prices = fetch_spot_prices()["Price(EUR/MW)"].values

positive_imbalance_price, negative_imbalance_price = extract_imbalance_prices()

tolerance = 1e-5  # You may adjust the tolerance level as necessary
big_M = 1000
data_points = 24  # The number of hours in the planning horizon, typically 24 for a single day.
time_frame = range(1,data_points)
grid_capacity_limit_MW = 50


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

# Define parameter ranges
capacity_range = np.linspace(5, 20, 4)  # 5, 10, 15, 20 MW
c_value_range = np.linspace(0.2, 1, 5)  # 0.2, 0.4, 0.6, 0.8, 1
efficiency_range = np.linspace(0.8, 0.95, 4)  # 0.8, 0.85, 0.9, 0.95
discharge_limit_range = np.linspace(0.5, 1, 4)  

# Generate combinations
battery_params_list = list(itertools.product(capacity_range, c_value_range, efficiency_range, discharge_limit_range))


def revenue_generated(battery_capacity_MW, battery_c_value, battery_efficiency, battery_discharge_limit):

    # Battery Parameters
    battery_capacity_MW = 10  # The maximum amount of energy that the battery can store, measured in Megawatts (MW).
    battery_c_value = 0.5  #  Represents the battery's discharge rate. A C-value of 0.5 means the battery can be dis/charged at a rate that would deplete its capacity in 2 hours.
    battery_efficiency = 0.9  # The efficiency of the battery in storing and discharging energy. A value of 0.9 (or 90%) means that 10% of energy is lost during charging and discharging cycles.
    battery_discharge_limit = 0.8 * battery_capacity_MW # The maximum amount of energy that can be discharged from the battery at any given time, calculated as a percentage of the battery's capacity.
    battery_SOC = 0 # Initial state of charge 


    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    day_ahead_vars = pl.LpVariable.dicts("Intraday", (range(data_points)), lowBound=0, cat="Continuous") # The energy (MWh) sold in the day ahead market. Each element t denotes energy sold for delivery at hour t. All the deals are made for the next day.
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
        pl.lpSum([day_ahead_vars[t] * day_ahead_spot_prices[t] for t in time_frame]) - pl.lpSum([grid_buy_vars[t] * day_ahead_spot_prices[t]] for t in range(data_points)) +  pl.lpSum(hydrogen_sold_vars[t] * hydrogen_market_price_EUR_per_kg - hydrogen_buy_vars[t] * hydrogen_cost_price_EUR_per_kg for t in range(data_points))
    )  

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + grid_buy_vars[t]
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + battery_charge_vars[t] + electricity_to_hydrogen_vars[t] 
            <=  battery_discharge_vars[t]*battery_efficiency + grid_buy_vars[t] + hydrogen_to_electricity_vars[t]*fuel_cell_conversion_efficiency + solar_production_actual[t-1] 
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
    return pl.value(problem.objective)




def simulate_solar_production_for_finland(df, num_scenarios, battery_params_list):
    city = LocationInfo("Helsinki", "Finland", "Europe/Helsinki", 60.1699, 24.9384)
    
    overall_results = []
    all_results = []
    for battery_params in battery_params_list:
        # New DataFrame to hold the simulated data in the same format as the actual data
        simulated_data_list = []

        optimization_results = []
        battery_capacity_MW, battery_c_value, battery_efficiency, battery_discharge_limit = battery_params

        solar_production_values = df['Amount(MW)'][df['Amount(MW)'] > 0]
        shape, loc, scale = stats.gamma.fit(solar_production_values, floc=0)  # we can fix location to 0 to simplify

        grouped = df.groupby(df['Time'].dt.date)

        for date, group in grouped:
            sun_times = sun(city.observer, date=date)
            sunrise = sun_times['sunrise'].hour + sun_times['sunrise'].minute / 60
            sunset = sun_times['sunset'].hour + sun_times['sunset'].minute / 60
            
            for scenario in range(num_scenarios):
                simulated_gamma_data = stats.gamma.rvs(shape, loc, scale, size=24)
                # Generate timestamps for each simulated hour
                timestamps = [pd.Timestamp(date) + timedelta(hours=i) for i in range(24)]
                
                # Apply sunrise/sunset mask
                simulated_production_forecast = np.where(
                    (np.arange(24) >= sunrise) & (np.arange(24) <= sunset),
                    simulated_gamma_data,
                    0
                )
                
                # Append each hour's simulated production to the list
                for timestamp, production in zip(timestamps, simulated_production_forecast):
                    simulated_data_list.append({
                            'Time': timestamp, 
                            'Scenario': scenario, 
                            'Simulated(MW)': production
                        })
                daily_result = revenue_generated(battery_capacity_MW, battery_c_value, battery_efficiency, battery_discharge_limit)
                optimization_results.append(daily_result)
                for daily_result in optimization_results:
                    all_results.append({
                        'battery_params': battery_params,
                        'revenue': daily_result
                    })
                

       # Convert all_results to a DataFrame
    results_df = pd.DataFrame(all_results)
    results_df[['Capacity (MW)', 'C-Value', 'Efficiency', 'Discharge Limit']] = pd.DataFrame(results_df['battery_params'].tolist(), index=results_df.index)
    return results_df

# Call the function to simulate solar production
results_df = simulate_solar_production_for_finland(df, num_scenarios=1, battery_params_list=battery_params_list)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Capacity vs Expected Revenue
axs[0, 0].scatter(results_df['Capacity (MW)'], results_df['revenue'], color='blue')
axs[0, 0].set_xlabel('Battery Capacity (MW)')
axs[0, 0].set_ylabel('Expected Revenue')
axs[0, 0].set_title('Capacity vs Expected Revenue')

# C-Value vs Expected Revenue
axs[0, 1].scatter(results_df['C-Value'], results_df['revenue'], color='red')
axs[0, 1].set_xlabel('C-Value')
axs[0, 1].set_ylabel('Expected Revenue')
axs[0, 1].set_title('C-Value vs Expected Revenue')

# Efficiency vs Expected Revenue
axs[1, 0].scatter(results_df['Efficiency'], results_df['revenue'], color='green')
axs[1, 0].set_xlabel('Efficiency')
axs[1, 0].set_ylabel('Expected Revenue')
axs[1, 0].set_title('Efficiency vs Expected Revenue')

# Discharge Limit vs Expected Revenue
axs[1, 1].scatter(results_df['Discharge Limit'], results_df['revenue'], color='purple')
axs[1, 1].set_xlabel('Discharge Limit')
axs[1, 1].set_ylabel('Expected Revenue')
axs[1, 1].set_title('Discharge Limit vs Expected Revenue')

plt.tight_layout()
plt.show()