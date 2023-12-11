
import pandas as pd

def fetch_day_ahead_prices():

    df = pd.read_csv("DayAheadPrice15thApril.csv")

    return df

def fetch_spot_prices():

    df = pd.read_csv("DayAheadPrice14thApril.csv")

    return df

def fetch_FCR_prices():

    df = pd.read_csv("FCR14thApril2023.csv",sep=";")

    df.drop(['Total','SE1 FCRN','SE2 FCRN','SE3 FCRN','SE4 FCRN','DK2 FCRN','FCR-D upp Pris (EUR/MW)','Total FCRD upp','SE1 FCRD upp','SE2 FCRD upp','SE3 FCRD upp','SE4 FCRD upp','DK2 FCRD upp','FCR-D ned Pris (EUR/MW)','Total FCRD ned','SE1 FCRD ned','SE2 FCRD ned','SE3 FCRD ned','SE4 FCRD ned','DK2 FCRD ned'], inplace=True, axis=1)

    df.rename(columns={'Datum': 'Date', 'FCR-N Pris (EUR/MW)': 'Prices'}, inplace=True)

    year = []
    month = []
    day = []
    time = []
    price = []

    for idx, row in df.iterrows():
        time_parts =  str(row.Date).split(' ')[1]
        time.append(int(time_parts.split(':')[0]))
        date_parts = str(row.Date).split(' ')[0].split('-')
        year.append(int(date_parts[0]))
        month.append(int(date_parts[1]))
        day.append(int(date_parts[2]))
        price_decimal = row.Prices.replace(",",".")
        price.append(round(float(price_decimal),2))


    df['Year'] = year
    df['Month'] = month
    df['Day'] = day
    df['Time'] = time
    df['Prices'] = price

    df.rename(columns={'Prices': 'Prices(EUR/MW)'}, inplace=True)
    
    df = df[['Year','Month','Day','Time','Prices(EUR/MW)']]

    return df

