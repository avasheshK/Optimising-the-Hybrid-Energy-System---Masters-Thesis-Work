import pandas as pd
import numpy as np
import pulp as pl
from astral import LocationInfo
from astral.sun import sun
from scipy.stats import t
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



def revenue_generated_all_assets(day_ahead_spot_prices):

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    day_ahead_vars = pl.LpVariable.dicts("Day_Ahead", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) sold in the day ahead market. Each element t denotes energy sold for delivery at hour t. All the deals are made for the next day.
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

    # After solving the problem, extract the final SOC and hydrogen storage for the day
    electricity_sold = pl.value(pl.lpSum(day_ahead_vars[t] for t in range(data_points)))
    battery_storage_usage = pl.value(pl.lpSum(battery_charge_vars[t] for t in range(data_points)))
    electricity_to_hydrogen_conversion = pl.value(pl.lpSum(electricity_to_hydrogen_vars[t] for t in range(data_points)))

    
    # Check the status of the solution
    status = pl.LpStatus[problem.status]
    print(f"Solver status: {status}")


    if status == 'Optimal':
        electricity_sold = pl.value(pl.lpSum(pl.value(day_ahead_vars[t]) for t in range(data_points)))
        battery_storage_usage = pl.value(pl.lpSum(pl.value(battery_charge_vars[t]) for t in range(data_points)))
        electricity_to_hydrogen_conversion = pl.value(pl.lpSum(pl.value(electricity_to_hydrogen_vars[t]) for t in range(data_points)))

        return pl.value(problem.objective), electricity_sold, battery_storage_usage, electricity_to_hydrogen_conversion

    # If not optimal, return None or another indicator of infeasibility
    return None, None, None, None

def revenue_generated_no_hydrogen(day_ahead_spot_prices):

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    day_ahead_vars = pl.LpVariable.dicts("Day_Ahead", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) sold in the day ahead market. Each element t denotes energy sold for delivery at hour t. All the deals are made for the next day.
    grid_buy_vars = pl.LpVariable.dicts("Grid_Buy", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) bought from the grid at each hour t.

    battery_charge_vars = pl.LpVariable.dicts("Battery_Charge", range(data_points), lowBound=0, cat="Continuous") # This variable represents the amount of energy (in MW) that is charged into the battery at each hour t within the planning horizon (defined by num_hours).
    battery_discharge_vars = pl.LpVariable.dicts("Battery_Discharge", range(data_points), lowBound=0, cat="Continuous") # Represents the amount of energy (in MW) discharged from the battery at each hour t within the planning horizon.
    battery_action = pl.LpVariable.dicts("Battery_Action", range(data_points), cat="Binary") # Prevent simultaneous charging and discharging of the battery within the same hour using a Binary variable.
    battery_SOC_vars = pl.LpVariable.dicts("Battery_SOC", range(data_points), lowBound=0, upBound=battery_capacity_MW, cat="Continuous") # Decision variable for state of charge of the battery

    # Set the objective function
    problem += (
        pl.lpSum([day_ahead_vars[t] * day_ahead_spot_prices[t] for t in time_frame]) - pl.lpSum([grid_buy_vars[t] * day_ahead_spot_prices[t]] for t in range(data_points)) )  

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + grid_buy_vars[t]
        ) <= grid_capacity_limit_MW


     # Add the energy balance constraints
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + battery_charge_vars[t] 
            <=  battery_discharge_vars[t]*battery_efficiency + grid_buy_vars[t] + solar_production_actual[t-1] 
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



    # Solve the problem and print the results
    problem.solve()

    
    # Check the status of the solution
    status = pl.LpStatus[problem.status]
    print(f"Solver status: {status}")


    if status == 'Optimal':
        electricity_sold = pl.value(pl.lpSum(pl.value(day_ahead_vars[t]) for t in range(data_points)))
        battery_storage_usage = pl.value(pl.lpSum(pl.value(battery_charge_vars[t]) for t in range(data_points)))


        return pl.value(problem.objective), electricity_sold, battery_storage_usage

    # If not optimal, return None or another indicator of infeasibility
    return None, None, None, None

def revenue_generated_no_battery_nor_hydrogen(day_ahead_spot_prices):

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    day_ahead_vars = pl.LpVariable.dicts("Day_Ahead", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) sold in the day ahead market. Each element t denotes energy sold for delivery at hour t. All the deals are made for the next day.
    grid_buy_vars = pl.LpVariable.dicts("Grid_Buy", range(data_points), lowBound=0, cat="Continuous") # The energy (MWh) bought from the grid at each hour t.



    # Set the objective function
    problem += (
        pl.lpSum([day_ahead_vars[t] * day_ahead_spot_prices[t] for t in time_frame]) - pl.lpSum([grid_buy_vars[t] * day_ahead_spot_prices[t]] for t in range(data_points))
    )  

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + grid_buy_vars[t]
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            day_ahead_vars[t] 
            <=   grid_buy_vars[t]  + solar_production_actual[t-1] 
        )


    # Solve the problem and print the results
    problem.solve()

    
    # Check the status of the solution
    status = pl.LpStatus[problem.status]
    print(f"Solver status: {status}")


    if status == 'Optimal':
        electricity_sold = pl.value(pl.lpSum(pl.value(day_ahead_vars[t]) for t in range(data_points)))
        return pl.value(problem.objective), electricity_sold

    # If not optimal, return None or another indicator of infeasibility
    return None, None, None, None


def simulate_market_prices(market_prices, scale_factor=1, num_simulations=100):
    # Calculate mean and standard deviation
    mean_price = np.mean(market_prices)
    std_dev_price = np.std(market_prices) * scale_factor

    # Initialize an array to store simulated prices
    simulated_prices = np.zeros(num_simulations)

    for i in range(num_simulations):
        # Generate a price and check if it's positive
        price = t.rvs(df=2, loc=mean_price, scale=std_dev_price)
        while price <= 15 or price >= 40:
            # If price is non-positive, re-generate it
            price = t.rvs(df=10, loc=mean_price, scale=std_dev_price)
        simulated_prices[i] = price

    return simulated_prices

def simulate_price_impact_on_revenue(num_price_scenarios=100):
    price_revenue_items = []
    price_revenue_items2 = []
    price_revenue_items3 = []

    for scenario in range(num_price_scenarios):
        # Simulate market prices for this scenario
        simulated_day_ahead_prices = simulate_market_prices(day_ahead_spot_prices)

        # Calculate revenue
        result = revenue_generated_all_assets(simulated_day_ahead_prices)
        result2 = revenue_generated_no_hydrogen(simulated_day_ahead_prices)
        result3 = revenue_generated_no_battery_nor_hydrogen(simulated_day_ahead_prices)

        # Only process results if they are valid
        if result[0] is not None:
            mean_price_scenario = np.mean(simulated_day_ahead_prices)
            price_revenue_items.append((mean_price_scenario, *result))
            price_revenue_items2.append((mean_price_scenario, *result2))
            price_revenue_items3.append((mean_price_scenario, *result3))

    return price_revenue_items, price_revenue_items2, price_revenue_items3

# Run the simulation with price variation
price_revenue_items, price_revenue_items2, price_revenue_items3 = simulate_price_impact_on_revenue()

# Convert to DataFrame for easy handling
price_revenue_df = pd.DataFrame(price_revenue_items, columns=['Mean Price', 'Revenue','Electricity Sold (in MW)', 'Battery Storage Usage', 'Electricity to Hydrogen'])
price_revenue_df2 = pd.DataFrame(price_revenue_items2, columns=['Mean Price', 'Revenue','Electricity Sold (in MW)','Battery Storage Usage'])
price_revenue_df3 = pd.DataFrame(price_revenue_items3, columns=['Mean Price', 'Revenue','Electricity Sold (in MW)'])
# Ensure price_revenue_df is sorted by 'Mean Price' if not already
price_revenue_df.sort_values('Mean Price', inplace=True)
price_revenue_df2.sort_values('Mean Price', inplace=True)
price_revenue_df3.sort_values('Mean Price', inplace=True)
# Assuming price_revenue_df is already sorted by 'Mean Price' and contains the required columns


# Optional: Print the range of simulated prices
print("Range of Simulated Prices:")
print(f"Min: {price_revenue_df['Mean Price'].min()}, Max: {price_revenue_df['Mean Price'].max()}")

def plot_market_prices1():
    # Revenue vs. Mean Market Price
    fig, ax = plt.subplots(figsize=(14, 7))
    color = 'tab:red'
    ax.set_ylabel('Revenue', color=color)
    ax.plot(price_revenue_df['Mean Price'], price_revenue_df['Revenue'], color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True)
    ax.set_title('Impact of Market Prices on Revenue with Battery and Hydrogen')
    plt.tight_layout()
    plt.show()


def plot_market_prices2():
    fig, ax = plt.subplots(figsize=(14, 7))
    color = 'tab:blue'
    ax.set_ylabel('Revenue', color=color)
    ax.plot(price_revenue_df2['Mean Price'], price_revenue_df2['Revenue'], color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True)
    ax.set_title('Impact of Market Prices on Revenue with Battery but without Hydrogen')
    plt.tight_layout()
    plt.show()


def plot_market_prices3():
    fig, ax = plt.subplots(figsize=(14, 7))
    color = 'tab:green'
    ax.set_ylabel('Revenue', color=color)
    ax.plot(price_revenue_df3['Mean Price'], price_revenue_df3['Revenue'], color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True)
    ax.set_title('Impact of Market Prices on Revenue without Battery or Hydrogen')
    plt.tight_layout()
    plt.show()


plot_market_prices1()
plot_market_prices2()
plot_market_prices3()