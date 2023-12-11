import pandas as pd
import numpy as np
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from solarproductionforecast import *
from fetchprices import *
from datetime import timedelta
from matplotlib.ticker import FuncFormatter

# Create a formatter function
def format_integers(x, pos):
    return f'{int(x)}'



forecast_df, actual_df, df = solar_energy_production_forecast()
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

def revenue_generated(solar_production):

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
            day_ahead_vars[t] + grid_buy_vars[t] + solar_production[t]
        ) <= grid_capacity_limit_MW


    # Add the energy balance constraints
    for t in time_frame:
        problem += (
            day_ahead_vars[t] + battery_charge_vars[t] + electricity_to_hydrogen_vars[t] 
            <=  battery_discharge_vars[t]*battery_efficiency + grid_buy_vars[t] + hydrogen_to_electricity_vars[t]*fuel_cell_conversion_efficiency + solar_production[t]
        )


    # Battery operational constraints
    for t in range(data_points):
        problem += battery_charge_vars[t] <= battery_capacity_MW * battery_c_value
        problem += battery_charge_vars[t] >= 0
        problem += battery_discharge_vars[t] >= 0

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
        problem += electricity_to_hydrogen_vars[t] >= 0
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

    if pl.LpStatus[problem.status] == 'Optimal':
    # Proceed with extracting variable values
        # After solving the problem, extract the final SOC and hydrogen storage for the day
        battery_storage_usage = pl.value(pl.lpSum(battery_charge_vars[t] for t in range(data_points)))
        final_hydrogen_storage = pl.value(hydrogen_storage_vars[data_points - 1])
        hydrogen_to_electricity_conversion = pl.value(pl.lpSum(hydrogen_to_electricity_vars[t] for t in range(data_points)))
        electricity_to_hydrogen_conversion = pl.value(pl.lpSum(electricity_to_hydrogen_vars[t] for t in range(data_points)))
        total_solar_production = pl.value(pl.lpSum(solar_production[t] for t in range(data_points)))

        # Return the daily revenue, final battery SOC, and final hydrogen storage
        return pl.value(problem.objective), battery_storage_usage, final_hydrogen_storage, hydrogen_to_electricity_conversion, int(electricity_to_hydrogen_conversion), total_solar_production
    else:
        # Handle non-optimal solution appropriately

         return None, None, None, None, None, None
 

# Initialize results_df outside the loop with correct columns
results_df = pd.DataFrame(columns=['Date', 'Revenue', 'Amount Charged to Battery', 'HydrogenStorage', 'Hydrogen to Electricity', 'Electricity to Hydrogen', 'Total Solar Production'])

# Loop over each day in the dataset
for date in df['Time'].dt.date.unique():
    daily_data = df[df['Time'].dt.date == date]
    solar_production = daily_data["Amount(MW)"].values
    
    # Call your revenue_generated function (updated for daily data)
    daily_revenue, battery_storage_use, daily_hydrogen_storage, htoeconversion, etohconversion, daily_solar_production = revenue_generated(solar_production)
    
    if daily_revenue is not None:   
        # Create a temporary DataFrame to hold the current day's results
        temp_df = pd.DataFrame({
            'Date': [pd.to_datetime(date)],
            'Revenue': [daily_revenue],
            'Amount Charged to Battery': [battery_storage_use],
            'HydrogenStorage': [daily_hydrogen_storage],
            'Hydrogen to Electricity': [htoeconversion],
            'Electricity to Hydrogen': [etohconversion],
            'Total Solar Production': [daily_solar_production]
        })
        # Append the temporary DataFrame to results_df
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
# Now convert 'Date' to datetime and set it as the index, outside the loop
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df.set_index('Date', inplace=True)




def add_season_shading(ax, results_df):
    # Get unique years present in the data
    if results_df.index.year is not None:
        unique_years = results_df.index.year.unique()

        # Define colors for each season for consistency
        season_colors = {
            'Spring': 'lightgreen',
            'Summer': 'lightcoral',
            'Fall': 'goldenrod',
            'Winter': 'lightskyblue'
        }

        for year in unique_years:
            print(results_df)
            int(year)
            # Define your seasons and their start/end dates
            seasons = {
                'Fall': (pd.Timestamp(year=year, month=8, day=23), pd.Timestamp(year=year, month=11, day=21)),
                'Winter': (pd.Timestamp(year=year, month=11, day=22), pd.Timestamp(year=year+1, month=3, day=22)),
                'Spring': (pd.Timestamp(year=year+1, month=3, day=23), pd.Timestamp(year=year+1, month=6, day=22)),
                'Summer': (pd.Timestamp(year=year+1, month=6, day=23), pd.Timestamp(year=year+1, month=8, day=22)),
            }

            # Assume 'Revenue' is already plotted on ax1

            # Now loop through the seasons and shade the regions
            for season, (start, end) in seasons.items():
                color = season_colors[season]
                # If the start date is later in the year than the end date, it means the season wraps around to the next year
                if start > end:
                    ax.axvspan(start, pd.Timestamp(year=year, month=12, day=31), color=color, alpha=0.3, label=f'{season} {year}')
                    ax.axvspan(pd.Timestamp(year=year+1, month=1, day=1), end, color=color, alpha=0.3)
                else:
                    ax.axvspan(start, end, color=color, alpha=0.3, label=f'{season} {year}')

        # Create custom legend entries
        legend_entries = [mpatches.Patch(color=color, label=season) for season, color in season_colors.items()]

        # Add a custom legend with a bit of transparency
        ax.legend(handles=legend_entries, loc='upper left', bbox_to_anchor=(0.125, 0.885), fancybox=True, shadow=True, ncol=1)


# Function to plot Revenue with seasons
def plot_revenue_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['Revenue'], color='tab:red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue (EUR)')
    ax.set_title('Daily Revenue Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()

# Function to plot Battery SOC with seasons
def plot_battery_soc_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['Amount Charged to Battery'], color='tab:blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount Charged to Battery')
    ax.set_title('Amount Charged to Battery Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()

# Function to plot Hydrogen Storage with seasons
def plot_hydrogen_storage_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['HydrogenStorage'], color='tab:green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Hydrogen Storage (kg)')
    ax.set_title('Hydrogen Storage Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()

# Function to plot Hydrogen To Electricity Conversion with seasons
def plot_hydrogen_to_electricity_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['Hydrogen to Electricity'], color='tab:pink')
    ax.set_xlabel('Date')
    ax.set_ylabel('Hydrogen To Electricity Conversion (kg)')
    ax.set_title('Hydrogen to Electricity Conversion Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()

# Function to plot Hydrogen To Electricity Conversion with seasons
def plot_electricity_to_hydrogen_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['Electricity to Hydrogen'], color='tab:purple')
    ax.set_xlabel('Date')
    ax.set_ylabel('Electricity To Hydrogen Conversion')
    ax.set_title('Electricity to Hydrogen Conversion Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()

# Function to plot Hydrogen To Electricity Conversion with seasons
def plot_total_solar_production_with_seasons(results_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(results_df.index, results_df['Total Solar Production'], color='tab:orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Solar Production (kg)')
    ax.set_title('Daily Solar Production Over the Year with Seasonal Shading')
    add_season_shading(ax, results_df)
    plt.tight_layout()
    plt.show()


# Call the plotting functions
plot_revenue_with_seasons(results_df)
plot_battery_soc_with_seasons(results_df)
plot_hydrogen_storage_with_seasons(results_df)
plot_hydrogen_to_electricity_with_seasons(results_df)
plot_electricity_to_hydrogen_with_seasons(results_df)
plot_total_solar_production_with_seasons(results_df)
