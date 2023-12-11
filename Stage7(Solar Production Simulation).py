import pandas as pd
import numpy as np
import pulp as pl
from astral import LocationInfo
from astral.sun import sun
import seaborn as sns
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

def calculate_likelihoods(optimization_results):
    # Calculate the likelihood of breaking even or making a profit
    break_even_likelihood = np.mean(np.array(optimization_results) == 0)
    profit_likelihood = np.mean(np.array(optimization_results) > 0)
    loss_likelihood = np.mean(np.array(optimization_results) < 0)

    return {
        'break_even_likelihood': break_even_likelihood,
        'profit_likelihood': profit_likelihood,
        'loss_likelihood': loss_likelihood
    }

def revenue_generated(simulated_production_forecast):

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
        + pl.lpSum([max(0,solar_production_actual[t] - simulated_production_forecast[t]) * positive_imbalance_price[t] + min(0,solar_production_actual[t] - simulated_production_forecast[t]) * negative_imbalance_price[t] for t in range(data_points)])
    )  

    # Add 50 MW grid buy/sell constraint
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + grid_buy_vars[t] + abs(solar_production_actual[t] - solar_production_forecast[t])
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + battery_charge_vars[t] + electricity_to_hydrogen_vars[t] + solar_production_actual[t-1] - solar_production_forecast[t-1]
            <=  battery_discharge_vars[t]*battery_efficiency + grid_buy_vars[t] + hydrogen_to_electricity_vars[t]*fuel_cell_conversion_efficiency
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


def simulate_solar_production_for_finland(df, num_scenarios=1):
    city = LocationInfo("Helsinki", "Finland", "Europe/Helsinki", 60.1699, 24.9384)
    optimization_results = []
    # New DataFrame to hold the simulated data in the same format as the actual data
    simulated_data_list = []

    # Fit a gamma distribution to the historical data
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
            daily_result = revenue_generated(simulated_production_forecast)
            optimization_results.append(daily_result)
            

     # Convert the list of simulated data to a DataFrame
    simulated_df = pd.DataFrame(simulated_data_list)

    # Group by Time and calculate the average simulated production across all scenarios
    average_simulated_df = simulated_df.groupby('Time')['Simulated(MW)'].mean().reset_index()


    # Calculate statistical measures of the distribution of revenues
    expected_revenue = np.mean(optimization_results)
    revenue_std_dev = np.std(optimization_results)
    percentile_0 = np.percentile(optimization_results,0)
    percentile_5 = np.percentile(optimization_results, 5)
    percentile_95 = np.percentile(optimization_results, 95)
    percentile_100 = np.percentile(optimization_results,100)

    return average_simulated_df, optimization_results, {
        'expected_revenue': expected_revenue,
        'revenue_std_dev': revenue_std_dev,
        'percentile_0': percentile_0,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'percentile_100': percentile_100,
    }
# Generate the PDF and probabilities
generate_gamma_pdf(solar_historical_data, k=4)

# Call the function to simulate solar production
average_simulated_df,optimization_results, expected_revenue = simulate_solar_production_for_finland(df, num_scenarios=1)

likelihoods = calculate_likelihoods(optimization_results)

# Save the simulated DataFrame to CSV

average_simulated_df.to_csv('simulated_solar_production.csv', index=False)

# Data for the bar chart
likelihood_data = {
    'Outcome': ['Break Even', 'Profit', 'Loss'],
    'Likelihood': [likelihoods['break_even_likelihood'], likelihoods['profit_likelihood'], likelihoods['loss_likelihood']]
}

# Plotting the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Outcome', y='Likelihood', data=likelihood_data)
plt.title('Likelihood of Different Revenue Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Likelihood')
plt.show()


# Output the likelihoods
print(f"Likelihood of Breaking Even: {likelihoods['break_even_likelihood']}")
print(f"Likelihood of Making a Profit: {likelihoods['profit_likelihood']}")
print(f"Likelihood of a Loss: {likelihoods['loss_likelihood']}")

print(f"Expected Revenue: {expected_revenue}")


# Setting style for nicer aesthetics
sns.set(style="whitegrid")

# Create a DataFrame for the revenues and their corresponding percentiles
revenues_df = pd.DataFrame({
    'Revenue': optimization_results
})
revenues_df['Percentile'] = pd.qcut(revenues_df['Revenue'], q=[0, 0.05, 0.95, 1], labels=False)
percentiles_labels = ['0-5th', '5th-95th', '95th-100th']
revenues_df['Percentile'] = revenues_df['Percentile'].map(dict(zip(range(3), percentiles_labels)))

# Plotting the box plot with seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Percentile', y='Revenue', data=revenues_df, palette="Set2")

# Setting plot title and labels
plt.title('Revenue Distribution Across Percentiles', fontsize=16)
plt.xlabel('Percentile', fontsize=12)
plt.ylabel('Revenue (EUR)', fontsize=12)

# Improving the layout
plt.tight_layout()

# Display the plot
plt.show()