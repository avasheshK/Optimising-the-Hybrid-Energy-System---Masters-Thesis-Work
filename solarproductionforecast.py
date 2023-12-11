
import pandas as pd
from datetime import datetime

def solar_energy_production_forecast():

    pd.set_option('display.max_rows', None)
    # Load your solar production data
    df = pd.read_csv('solarproduction.csv', parse_dates=['Start time UTC', 'End time UTC', 'Start time UTC+03:00', 'End time UTC+03:00'])

    df[['Solar power production forecast - hourly update']] = df[['Solar power production forecast - hourly update']]/10
    # Assuming the original data has columns 'start_time' and 'value'
    df.drop(['Start time UTC+03:00','End time UTC','End time UTC+03:00'], inplace=True, axis=1)
    # Assuming the last column is the solar power production
    df.rename(columns={'Solar power production forecast - hourly update': 'Amount(MW)','Start time UTC':'Time'}, inplace=True)

    # Convert 'Time' to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Forecasting for the next day using persistence method
    forecast_day = pd.Timestamp('2022-10-21')  # Adjust this date to your target forecast day

    previous_day = forecast_day - pd.DateOffset(days=1)

    # Filter out the relevant day from last year
    forecast_data = df[df['Time'].dt.date == previous_day.date()].copy()

    # If there's no data for that day in the previous year, return an empty DataFrame
    if forecast_data.empty:
        print(f"No data available for {previous_day.date()} to perform persistence forecasting.")
        return pd.DataFrame(columns=['Year', 'Month', 'Day', 'Time', 'Amount(MW)'])

    # Adjust the dates in the forecast data to the next day
    forecast_data['Time'] = forecast_data['Time'] + pd.DateOffset(days=1)

    # Extract actual production data for the target day
    actual_data = df[df['Time'].dt.date == forecast_day.date()].copy()

    # Function to reformat dataframe
    def reformat_dataframe(data):
        data['Year'] = data['Time'].dt.year
        data['Month'] = data['Time'].dt.month
        data['Day'] = data['Time'].dt.day
        data['Time'] = data['Time'].dt.hour
        return data[['Year', 'Month', 'Day', 'Time', 'Amount(MW)']]

    forecast_data_formatted = reformat_dataframe(forecast_data)
    actual_data_formatted = reformat_dataframe(actual_data)

    return forecast_data_formatted, actual_data_formatted, df


def extract_imbalance_prices():
    # Read the CSV file
    df = pd.read_csv('imbalancepricing.csv')

    # Extract only the required columns
    positive_imbalance_price = df["Generation / + Imbalance price [EUR/MWh] - MBA|FI"]
    negative_imbalance_price = df["Generation / - Imbalance price [EUR/MWh] - MBA|FI"]

    return positive_imbalance_price, negative_imbalance_price

solar_energy_production_forecast()