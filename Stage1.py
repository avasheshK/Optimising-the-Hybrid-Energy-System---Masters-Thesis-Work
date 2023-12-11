import pandas as pd
import numpy as np
import pulp as pl

from solarenergyproduction import *
from fetchprices import *

solar_production = solar_energy_production()["Amount(MW)"].values
intraday_prices = fetch_day_ahead_prices()["Price(EUR/MW)"].values

data_points = 24  # The number of hours in the planning horizon, typically 24 for a single day.

def revenue_generated():

    # Create the LP problem
    problem = pl.LpProblem("Revenue_Maximization", pl.LpMaximize)

    # Create decision variables
    intraday_vars = pl.LpVariable.dicts("Intraday", (range(data_points), range(data_points)), lowBound=0, cat="Continuous") # The energy (MWh) sold in the intraday market. Each element (t1, t2) denotes energy sold at hour t1 for delivery at hour t2.

    # Set the objective function
    problem += (
        pl.lpSum([intraday_vars[t1][min(t1+1,data_points-1)] * intraday_prices[min(t1+1,data_points-1)] for t1 in range(data_points)])
    ) 

    # Add 50 MW grid buy/sell constraint
    for t in range(data_points):
        problem += (
            intraday_vars[t][min(t+1,data_points-1)]
        ) <= 50


    # Add the energy balance constraints
    for t in range(data_points):
        problem += (
            intraday_vars[t][min(t+1,data_points-1)] 
            == solar_production[t]
        )


    # Solve the problem and print the results
    problem.solve()
    print("Status: ", pl.LpStatus[problem.status])
    print("Optimal revenue: ", pl.value(problem.objective))

    # Print the selling decisions
    print("\nSelling decisions:")
    for t in range(data_points):
        intraday_val = pl.value(intraday_vars[t][min(t+1,data_points-1)])
        if intraday_val > 0:
            print(f"Hour {t}: Sell {intraday_val:.2f} MWh at intraday price of hour {t+1}")



revenue_generated()