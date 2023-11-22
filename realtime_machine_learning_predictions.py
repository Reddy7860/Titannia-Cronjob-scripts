import numpy as np
from datetime import datetime, timedelta
from pandasql import sqldf
import warnings
import yfinance as yf
import pandas as pd
from pymongo import MongoClient
import pandas_ta as ta
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import requests
from pytz import timezone
import os

# Ignore warnings
warnings.filterwarnings("ignore")

discord_url = "https://discord.com/api/v9/channels/1123408593283194973/messages"

discord_headers = {
    "Authorization": "Bot MTEyMzQwNjAzMDc1MjIwNjg2OA.G1K5_1.T8afjI1NUBL3cGnzkvp_P1Ja9kaJA-nXy92f-8",
    "Content-Type": "application/json"
}

# MongoDB connection details
username = 'Titania'
password = 'Mahadev'
cluster_url = 'cluster0.zq3w2cn.mongodb.net'
database_name = 'United_States_Titania_Trading'
collection_name = 'final_orders_raw_data'

# MongoDB connection URI
uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority"

# Connect to the MongoDB client
client = MongoClient(uri)

# Access the database and collection
db = client[database_name]
collection = db[collection_name]

collection_realtime_machine_learning_data = db['realtime_machine_learning_data']

# Get today's date
today = datetime.now().strftime('%Y-%m-%d')

print(today)

# Query the collection with today's date
final_orders_raw_data = collection.find({"execution_date": today})

# Convert the cursor to a DataFrame
final_orders_raw_data = pd.DataFrame(list(final_orders_raw_data))

print(final_orders_raw_data)

# Define the ticker symbols for S&P 500, Dow Jones, and Nasdaq
s_and_p_ticker = "^GSPC"
dow_jones_ticker = "^DJI"
nasdaq_ticker = "^IXIC"

start_date = '2023-06-01'

data = yf.download([s_and_p_ticker, dow_jones_ticker, nasdaq_ticker], start=start_date)

# Extract the necessary columns from the data
extracted_data = data[["Open", "High", "Low", "Close"]]

# Rename the columns
extracted_data.columns = ["s_and_p_Open", "s_and_p_High", "s_and_p_Low", "s_and_p_Close",
                          "dow_jones_Open", "dow_jones_High", "dow_jones_Low", "dow_jones_Close",
                          "nasdaq_Open", "nasdaq_High", "nasdaq_Low", "nasdaq_Close"]

# Reset the index to get the date as a separate column
extracted_data.reset_index(inplace=True)

extracted_data["s_and_p_Previous_1_Open"] = extracted_data["s_and_p_Open"].shift(1)
extracted_data["s_and_p_Previous_2_Open"] = extracted_data["s_and_p_Open"].shift(2)
extracted_data["s_and_p_Previous_1_High"] = extracted_data["s_and_p_High"].shift(1)
extracted_data["s_and_p_Previous_2_High"] = extracted_data["s_and_p_High"].shift(2)
extracted_data["s_and_p_Previous_1_Low"] = extracted_data["s_and_p_Low"].shift(1)
extracted_data["s_and_p_Previous_2_Low"] = extracted_data["s_and_p_Low"].shift(2)
extracted_data["s_and_p_Previous_1_Close"] = extracted_data["s_and_p_Close"].shift(1)
extracted_data["s_and_p_Previous_2_Close"] = extracted_data["s_and_p_Close"].shift(2)

extracted_data["dow_jones_Previous_1_Open"] = extracted_data["dow_jones_Open"].shift(1)
extracted_data["dow_jones_Previous_2_Open"] = extracted_data["dow_jones_Open"].shift(2)
extracted_data["dow_jones_Previous_1_High"] = extracted_data["dow_jones_High"].shift(1)
extracted_data["dow_jones_Previous_2_High"] = extracted_data["dow_jones_High"].shift(2)
extracted_data["dow_jones_Previous_1_Low"] = extracted_data["dow_jones_Low"].shift(1)
extracted_data["dow_jones_Previous_2_Low"] = extracted_data["dow_jones_Low"].shift(2)
extracted_data["dow_jones_Previous_1_Close"] = extracted_data["dow_jones_Close"].shift(1)
extracted_data["dow_jones_Previous_2_Close"] = extracted_data["dow_jones_Close"].shift(2)

extracted_data["nasdaq_Previous_1_Open"] = extracted_data["nasdaq_Open"].shift(1)
extracted_data["nasdaq_Previous_2_Open"] = extracted_data["nasdaq_Open"].shift(2)
extracted_data["nasdaq_Previous_1_High"] = extracted_data["nasdaq_High"].shift(1)
extracted_data["nasdaq_Previous_2_High"] = extracted_data["nasdaq_High"].shift(2)
extracted_data["nasdaq_Previous_1_Low"] = extracted_data["nasdaq_Low"].shift(1)
extracted_data["nasdaq_Previous_2_Low"] = extracted_data["nasdaq_Low"].shift(2)
extracted_data["nasdaq_Previous_1_Close"] = extracted_data["nasdaq_Close"].shift(1)
extracted_data["nasdaq_Previous_2_Close"] = extracted_data["nasdaq_Close"].shift(2)

# Calculate previous day's price movement percentage
extracted_data["s_and_p_Prev_Percentage"] = extracted_data["s_and_p_Previous_1_Close"].pct_change() * 100
extracted_data["dow_jones_Prev_Percentage"] = extracted_data["dow_jones_Previous_1_Close"].pct_change() * 100
extracted_data["nasdaq_Prev_Percentage"] = extracted_data["nasdaq_Previous_1_Close"].pct_change() * 100

# Calculate log price movement
extracted_data["s_and_p_Log_Previous_1_Price_Movement"] = np.log(extracted_data["s_and_p_Previous_1_Close"] / extracted_data["s_and_p_Previous_1_Close"].shift(1))
extracted_data["dow_jones_Log_Previous_1_Price_Movement"] = np.log(extracted_data["dow_jones_Previous_1_Close"] / extracted_data["dow_jones_Previous_1_Close"].shift(1))
extracted_data["nasdaq_Log_Previous_1_Price_Movement"] = np.log(extracted_data["nasdaq_Previous_1_Close"] / extracted_data["nasdaq_Previous_1_Close"].shift(1))


def calculate_classic_pivots(data,stk_idx):
    pivot_data = data.tail(1)
    # print(pivot_data)
    pivot_data.reset_index(level=0, inplace=True)
    pivot_point = (pivot_data.loc[0,'High'] + pivot_data.loc[0,'Low'] + pivot_data.loc[0,'Close'])/3

    pivot_bc = (pivot_data.loc[0,'High'] + pivot_data.loc[0,'Low'])/2
    pivot_tc = 2* pivot_point - pivot_bc

    # print(pivot_bc)
    # print(pivot_tc)
    
    classic_support_1 = round((2*pivot_point) - pivot_data.loc[0,'High'],2)
    
    classic_resistance_1 = round((2*pivot_point) - pivot_data.loc[0,'Low'],2)
        
    classic_support_2 = round(pivot_point - (classic_resistance_1 - classic_support_1),2)

    classic_resistance_2 = round((pivot_point - classic_support_1 ) + classic_resistance_1,2)

    classic_resistance_3 = round((pivot_point - classic_support_2 ) + classic_resistance_2,2)

    classic_support_3 = round(pivot_point - (classic_resistance_2 - classic_support_2),2)
    
    
    price_difference = (pivot_data.loc[0,'High'] - pivot_data.loc[0,'Low'])

    fibonnaci_resistance_1 = round((38.2*price_difference/100) + pivot_point,2)

    fibonnaci_resistance_2 = round((61.8*price_difference/100) + pivot_point,2)

    fibonnaci_resistance_3 = round((100*price_difference/100) + pivot_point,2)

    fibonnaci_support_1 = round(pivot_point - (38.2*price_difference/100),2)

    fibonnaci_support_2 = round(pivot_point - (61.8*price_difference/100),2)

    fibonnaci_support_3 = round(pivot_point - (100*price_difference/100),2)
    
    
    daily_levels_final_data.loc[stk_idx,"Date"] = pivot_data.loc[0,'Date']
    daily_levels_final_data.loc[stk_idx,"Open"] = pivot_data.loc[0,'Open']
    daily_levels_final_data.loc[stk_idx,"High"] = pivot_data.loc[0,'High']
    daily_levels_final_data.loc[stk_idx,"Low"] = pivot_data.loc[0,'Low']
    daily_levels_final_data.loc[stk_idx,"Close"] = pivot_data.loc[0,'Close']
    
    daily_levels_final_data.loc[stk_idx,"pivot_point"] = round(pivot_point,2)
    daily_levels_final_data.loc[stk_idx,"pivot_bc"] = round(pivot_bc,2)
    daily_levels_final_data.loc[stk_idx,"pivot_tc"] = round(pivot_tc,2)

    daily_levels_final_data.loc[stk_idx,"classical_support_1"] = classic_support_1
    
    daily_levels_final_data.loc[stk_idx,"classical_resistance_1"] = classic_resistance_1
        
    daily_levels_final_data.loc[stk_idx,"classical_support_2"] = classic_support_2

    daily_levels_final_data.loc[stk_idx,"classical_resistance_2"] = classic_resistance_2

    daily_levels_final_data.loc[stk_idx,"classical_resistance_3"] = classic_resistance_3

    daily_levels_final_data.loc[stk_idx,"classical_support_3"] = classic_support_3
    
    
    price_difference = (pivot_data.loc[0,'High'] - pivot_data.loc[0,'Low'])

    daily_levels_final_data.loc[stk_idx,"fibonnaci_resistance_1"] = round((38.2*price_difference/100) + pivot_point,2)

    daily_levels_final_data.loc[stk_idx,"fibonnaci_resistance_2"] = round((61.8*price_difference/100) + pivot_point,2)

    daily_levels_final_data.loc[stk_idx,"fibonnaci_resistance_3"] = round((100*price_difference/100) + pivot_point,2)

    daily_levels_final_data.loc[stk_idx,"fibonnaci_support_1"] = round(pivot_point - (38.2*price_difference/100),2)

    daily_levels_final_data.loc[stk_idx,"fibonnaci_support_2"] = round(pivot_point - (61.8*price_difference/100),2)

    daily_levels_final_data.loc[stk_idx,"fibonnaci_support_3"] = round(pivot_point - (100*price_difference/100),2)


latest_merged_df = pd.DataFrame()


for idx in range(0,len(final_orders_raw_data)):
    print(idx)
    new_df = pd.DataFrame()
    ticker = final_orders_raw_data.loc[idx,'Stock']
    current_data = yf.download(ticker,period="5d",interval="5m")
    current_data = pd.DataFrame(current_data)
    current_data.reset_index(inplace=True)

    # Download the data using yfinance
    stock_data = yf.download(ticker,period="5d",interval="1d")

    # Convert the data to a pandas DataFrame
    stock_data = pd.DataFrame(stock_data)

    stock_data.reset_index(inplace=True)

    # print(stock_data)
    # print(current_data)
    current_value = final_orders_raw_data.loc[idx,'Value']

    # Convert DateTime column to datetime type
    current_data['Datetime'] = pd.to_datetime(current_data['Datetime'])

    # Create a separate date column
    current_data['date'] = current_data['Datetime'].dt.date

    # Sort the dataframe by DateTime
    current_data.sort_values('Datetime', inplace=True)

    # Initialize Day_High_Till_Time and Day_Low_Till_Time columns
    current_data['Day_High_Till_Time'] = current_data.groupby('date')['High'].cummax().shift()
    current_data['Day_Low_Till_Time'] = current_data.groupby('date')['Low'].cummin().shift()

    # Forward fill the missing values
    current_data['Day_High_Till_Time'].ffill(inplace=True)
    current_data['Day_Low_Till_Time'].ffill(inplace=True)

    # Convert NaN values to empty string
    current_data['Day_High_Till_Time'] = current_data['Day_High_Till_Time'].fillna('').astype(str)
    current_data['Day_Low_Till_Time'] = current_data['Day_Low_Till_Time'].fillna('').astype(str)

    # current_data.loc[current_data['Day_High_Till_Time'] == '', 'Day_High_Till_Time'] = current_data['High']
    # current_data.loc[current_data['Day_Low_Till_Time'] == '', 'Day_Low_Till_Time'] = current_data['Low']

    current_data['Datetime'] = pd.to_datetime(current_data['Datetime'])

    current_data['date'] = current_data['Datetime'].dt.date

    current_data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    current_data['rsi'] = current_data.ta.rsi(close='Close',length = 14)
    current_data['sma_20'] = ta.sma(current_data["Close"], length=20)
    current_data.ta.bbands(close = 'Close', length=20, std=2,append = True)
    current_data['SMA_Call'] = current_data.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
    current_data['RSI_Call'] = current_data.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
    current_data['MACD_Call'] = current_data.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
    current_data['Pivot_Call'] = ''
    current_data['PCR_Call'] = ''

    # Calculate Stochastic Oscillator
    current_data.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)

    # Calculate Average True Range (ATR)
    current_data.ta.atr(high='High', low='Low', close='Close', length=14, append=True)

    # Calculate On-Balance Volume (OBV)
    current_data.ta.obv(close='Close', volume='Volume', append=True)

    daily_levels_final_data = pd.DataFrame()

    for stk_idx in range(0,len(stock_data)):
        calculate_classic_pivots(pd.DataFrame(stock_data.loc[stk_idx,]).transpose(),stk_idx)

    current_data['date'] = pd.to_datetime(current_data['date'])
    daily_levels_final_data['Date'] = pd.to_datetime(daily_levels_final_data['Date'])

    # Merge the dataframes on the date columns
    merged_df = pd.merge(current_data, daily_levels_final_data, how='inner', left_on='date', right_on='Date')

    # Merge the dataframes on the date columns
    merged_df = pd.merge(merged_df, extracted_data, how='inner')

    # Optionally, you can drop the duplicate 'Date' column
    merged_df = merged_df.drop(['Date'], axis=1)

    merged_df = merged_df.rename(columns={'Open_x': 'Open', 'High_x':'High','Low_x':'Low','Close_x':'Close', 'Open_y': 'day_open', 'High_y': 'day_high', 'Low_y': 'day_low', 'Close_y': 'day_close'})

    merged_df['Pivot_Call'] = merged_df.apply(lambda x: 'Buy' if x['Close'] >= x['pivot_bc'] else 'Sell', axis=1)
    hist_df = merged_df[['Datetime','Open', 'High','Low', 'Close','Volume']]
    hist_df.set_index(pd.DatetimeIndex(hist_df["Datetime"]), inplace=True)
    hist_df.ta.vwap(high='High', low='Low',close='Close',volume='Volume', append=True)
    hist_df.ta.supertrend(high='High',low='Low',close='Close',append=True)
    hist_df.reset_index(inplace=True,drop=True)
    # print(hist_df.tail(5))
    # print(merged_df.tail(5))
    result = pd.merge(merged_df, hist_df, on="Datetime")
    result.reset_index(inplace=True,drop=True)  

    result = result[['Datetime', 'Open_x', 'High_x', 'Low_x', 'Close_x','Volume_x',
                    'Day_High_Till_Time', 'Day_Low_Till_Time', 
                    'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                    'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV','VWAP_D', 'SUPERT_7_3.0',
                    'pivot_point', 'pivot_bc', 'pivot_tc',
                    'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3',
                    'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                    's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                    's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                    's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                    's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                    'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                    'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                    'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                    'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                    'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                    'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                    'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                    'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                    's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                    'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                    'dow_jones_Log_Previous_1_Price_Movement',
                    'nasdaq_Log_Previous_1_Price_Movement'
                    ]]
    

    result.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
                      'Day_High_Till_Time', 'Day_Low_Till_Time',
                      'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20','BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                      'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV','VWAP_D', 'supertrend',
                      'pivot_point', 'pivot_bc', 'pivot_tc',
                        'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3',
                        'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                        's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                        's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                        's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                        's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                        'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                        'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                        'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                        'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                        'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                        'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                        'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                        'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                        's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                        'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                        'dow_jones_Log_Previous_1_Price_Movement',
                        'nasdaq_Log_Previous_1_Price_Movement'
                      ]
    # print(result)
    result['VWAP_D'] = result['VWAP_D'].replace(np.nan, 0)
    result['supertrend'] = result['supertrend'].replace(np.nan, 0)

    result['BB_Call'] = result.apply(lambda x: 'Buy' if x['Close'] <= x['BBL_20_2.0'] else 'Sell' if x['Close'] >= x['BBU_20_2.0'] else 'Neutral', axis=1)
    result['VWAP_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['VWAP_D'] else 'Sell',axis = 1)
    result['SuperTrend_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['supertrend'] else 'Sell',axis = 1)
    result['date'] = pd.to_datetime(result['Datetime'], format='%Y-%m-%d')

    # result = result[[ 'Datetime', 'Open', 'High', 'Low', 'Close','Volume','Day_High_Till_Time', 'Day_Low_Till_Time', 'SMA_Call', 'RSI_Call', 'MACD_Call','Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV']]

    result = result[[ 'Datetime', 'Open', 'High', 'Low', 'Close','Volume',
                 'Day_High_Till_Time', 'Day_Low_Till_Time',
                'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 
                'SMA_Call', 'RSI_Call', 'MACD_Call','Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV', 'VWAP_D','supertrend',
                'pivot_point', 'pivot_bc', 'pivot_tc',
                'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3', 
                'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                'dow_jones_Log_Previous_1_Price_Movement',
                'nasdaq_Log_Previous_1_Price_Movement'
                ]]


    for row in range(0,len(result)):
        buy_probability = 0
        sell_probability = 0
        if result.loc[row,'SMA_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'SMA_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'RSI_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'RSI_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'MACD_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'MACD_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'Pivot_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'Pivot_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'BB_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'BB_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'PCR_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'PCR_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'VWAP_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'VWAP_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5

        if result.loc[row,'SuperTrend_Call'] == 'Buy':
            buy_probability = buy_probability + 12.5
        elif result.loc[row,'SuperTrend_Call'] == 'Sell':
            sell_probability = sell_probability + 12.5


        result.loc[row,'buy_probability'] = buy_probability
        result.loc[row,'sell_probability'] = sell_probability

    result['date'] = pd.to_datetime(result['Datetime'])
    result['date'] = result['date'].dt.date

    daily_levels_final_data['Date'] = pd.to_datetime(daily_levels_final_data['Date'])

    # Convert the 'Date' column in stock_data and 'DateTime' column in merged_df to datetime type
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Previous_1_Open'] = stock_data['Open'].shift(1)
    stock_data['Previous_1_High'] = stock_data['High'].shift(1)
    stock_data['Previous_1_Low'] = stock_data['Low'].shift(1)
    stock_data['Previous_1_Close'] = stock_data['Close'].shift(1)
    stock_data['Previous_1_Volume'] = stock_data['Volume'].shift(1)
    stock_data['Previous_2_Open'] = stock_data['Open'].shift(2)
    stock_data['Previous_2_High'] = stock_data['High'].shift(2)
    stock_data['Previous_2_Low'] = stock_data['Low'].shift(2)
    stock_data['Previous_2_Close'] = stock_data['Close'].shift(2)
    stock_data['Previous_2_Volume'] = stock_data['Volume'].shift(2)

    # Convert the 'date' column in 'result' dataframe to datetime type
    result['date'] = pd.to_datetime(result['date'])

    # Convert the 'Date' column in 'stock_data' dataframe to datetime type
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Merge the dataframes based on the date columns
    result = result.merge(stock_data, left_on='date', right_on='Date', how='left')

    result = result.rename(columns={'Open_x': 'Open', 'High_x':'High','Low_x':'Low','Close_x':'Close','Volume_x':'Volume'})

    result = result.drop(['Open_y', 'High_y','Low_y','Close_y','Volume_y'], axis=1)

    # print(final_orders_raw_data.loc[idx,"Datetime"])

    new_df.loc[0,"Call_time"] = final_orders_raw_data.loc[idx,"Datetime"]
    new_df.loc[0,"Strategy"] = final_orders_raw_data.loc[idx,"Strategy"]
    new_df.loc[0,"Call"] = final_orders_raw_data.loc[idx,"Signal"]
    new_df.loc[0,"stock"] = final_orders_raw_data.loc[idx,"Stock"]
    new_df.loc[0,"Value"] = final_orders_raw_data.loc[idx,"Value"]
    new_df.loc[0,"points"] = final_orders_raw_data.loc[idx,"Target"]
    new_df.loc[0,'Target_SL'] = ''
    

    # Convert 'Call_time' column in 'new_df' to datetime type
    new_df['Call_time'] = pd.to_datetime(new_df['Call_time'])

    # Convert 'Datetime' column in 'result' to datetime type
    result['Datetime'] = pd.to_datetime(result['Datetime'])

    # print("new_df head")

    # print(new_df.head())
    # print(result)

    temp_merged_df = new_df.merge(result, left_on='Call_time', right_on='Datetime', how='inner')

    # print(temp_merged_df.columns)

    # temp_merged_df = temp_merged_df[['stock','Strategy','Call_time','Open','High','Low','Close','Volume','Day_High_Till_Time', 'Day_Low_Till_Time','Previous_1_Open','Previous_1_High', 'Previous_1_Low', 'Previous_1_Close','Previous_1_Volume','Previous_2_Open', 'Previous_2_High', 'Previous_2_Low','Previous_2_Close','Previous_2_Volume','Call','Value','points','Target_SL','SMA_Call','RSI_Call','MACD_Call','Pivot_Call','PCR_Call','BB_Call','VWAP_Call','SuperTrend_Call','buy_probability','sell_probability','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV']]

    temp_merged_df.replace({"Buy": 1, "Sell": -1, "Neutral": 0}, inplace=True)
    temp_merged_df['PCR_Call'].replace({"": 0})

    temp_merged_df['Day'] = temp_merged_df['Call_time'].dt.day
    temp_merged_df['Month'] = temp_merged_df['Call_time'].dt.month
    temp_merged_df['Year'] = temp_merged_df['Call_time'].dt.year
    temp_merged_df['Hour'] = temp_merged_df['Call_time'].dt.hour
    temp_merged_df['Minute'] = temp_merged_df['Call_time'].dt.minute

    # Calculate day of the week in numbers (0: Monday, 1: Tuesday, ..., 6: Sunday)
    temp_merged_df['DayOfWeek'] = temp_merged_df['Call_time'].dt.dayofweek
    # Calculate quarter
    temp_merged_df['Quarter'] = temp_merged_df['Call_time'].dt.quarter

    latest_merged_df = latest_merged_df.append(temp_merged_df)
    
    
# Define the stock symbols
stocks = ['^IXIC', '^GSPC', '^DJI']

# Get today's date
today = datetime.now().date()

# Find the previous weekday (Friday if today is Monday)
start_date = today - timedelta(days=1)
while start_date.weekday() > 4:
    start_date -= timedelta(days=1)

# Find the next weekday (Monday if today is Friday)
end_date = today + timedelta(days=1)
while end_date.weekday() > 4:
    end_date += timedelta(days=1)

print(start_date)
print(end_date)

# Fetch the historical data from Yahoo Finance
data = yf.download(stocks, start=start_date, end=end_date, interval='5m')

# Filter the data to include only the trading hours (9:30 AM to 4:00 PM Eastern Time)
data = data.between_time('09:30', '16:00')

# Reset the index and include the 'Date' column
data.reset_index(inplace=True)
data['Date'] = data['Datetime'].dt.date

# Select the required columns using cross-section (xs) method
df_pivot = pd.DataFrame()
df_pivot['DateTime'] = data['Datetime']
df_pivot['Date'] = data['Date']
df_pivot['Open_Dow_Jones'] = data['Open']['^DJI']
df_pivot['Open_nasdaq'] = data['Open']['^IXIC']
df_pivot['Open_s_and_p'] = data['Open']['^GSPC']
df_pivot['High_Dow_Jones'] = data['High']['^DJI']
df_pivot['High_nasdaq'] = data['High']['^IXIC']
df_pivot['High_s_and_p'] = data['High']['^GSPC']
df_pivot['Low_Dow_Jones'] = data['Low']['^DJI']
df_pivot['Low_nasdaq'] = data['Low']['^IXIC']
df_pivot['Low_s_and_p'] = data['Low']['^GSPC']
df_pivot['Close_Dow_Jones'] = data['Close']['^DJI']
df_pivot['Close_nasdaq'] = data['Close']['^IXIC']
df_pivot['Close_s_and_p'] = data['Close']['^GSPC']

# Rearrange the columns
df_pivot = df_pivot[['DateTime', 'Date', 'Open_Dow_Jones', 'Open_nasdaq', 'Open_s_and_p',
         'High_Dow_Jones', 'High_nasdaq', 'High_s_and_p', 'Low_Dow_Jones',
         'Low_nasdaq', 'Low_s_and_p', 'Close_Dow_Jones', 'Close_nasdaq',
         'Close_s_and_p']]

filtered_df = pd.DataFrame()

if len(latest_merged_df) > 0:
    # Merge the dataframes on the date columns
    latest_merged_df = pd.merge(latest_merged_df, df_pivot, how='inner', left_on="Call_time", right_on="DateTime")

    print(latest_merged_df)

    latest_merged_df = latest_merged_df.drop(['DateTime'], axis=1)

    latest_merged_df.reset_index(inplace=True,drop=True)

    ml_latest_merged_df = latest_merged_df.copy()
    ml_latest_merged_df.drop('Call_time', axis=1, inplace=True)
    ml_latest_merged_df.drop('PCR_Call', axis=1, inplace=True)

    filter_stocks = ['AAPL','BA','CSCO','CVX','DD','GE','GS','IBM','INTC','JNJ','JPM','KO','MCD','MMM','MRK','MSFT','NKE','PFE','PG','TRV','TSLA','UNH','V','XOM']

    filtered_df = ml_latest_merged_df[ml_latest_merged_df['stock'].isin(filter_stocks)]

    filtered_df.reset_index(inplace=True,drop=True)

    print(filtered_df)

# Define the features
features = ['Open', 'High', 'Low', 'Close', 'Volume','Day_High_Till_Time', 'Day_Low_Till_Time',
            'Previous_1_Open', 'Previous_1_High', 'Previous_1_Low','Previous_1_Close', 'Previous_1_Volume',
            'Previous_2_Open', 'Previous_2_High', 'Previous_2_Low', 'Previous_2_Close', 'Previous_2_Volume', 
            'Call', 'Value', 'points',
            'buy_probability', 'sell_probability',
            'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0',
            'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATRr_14', 'OBV', 'VWAP_D', 'supertrend',
            'pivot_point', 'pivot_bc', 'pivot_tc',
            'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3', 
            'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
            'Day', 'Month', 'Year', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
            's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
            's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
            's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
            's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
            'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
            'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
            'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
            'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
            'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
            'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
            'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
            'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
            's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
            'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
            'dow_jones_Log_Previous_1_Price_Movement',
            'nasdaq_Log_Previous_1_Price_Movement',
            'Open_Dow_Jones','Open_nasdaq','Open_s_and_p',
            'High_Dow_Jones','High_nasdaq','High_s_and_p',
            'Low_Dow_Jones','Low_nasdaq','Low_s_and_p',
            'Close_Dow_Jones','Close_nasdaq','Close_s_and_p']

pd.set_option('display.max_columns', None)

if len(filtered_df) >0:
    for idx, row in filtered_df.iterrows():
        try:
            filter_ml_pred_row = filtered_df.loc[(filtered_df['stock'] == row['stock']) & (filtered_df['Strategy'] == row['Strategy'])]
            print(filter_ml_pred_row[['stock','Strategy','Call','Value','points']])

            filter_ml_pred_row.reset_index(inplace=True,drop=True)
            Strategy = filter_ml_pred_row.loc[0,"Strategy"]
            stock = filter_ml_pred_row.loc[0,"stock"]

            if Strategy != 'Volume_Breakout':
                print(Strategy)
                if Strategy == '5_Cand_ABC':
                    Strategy = '5_cand_ABC'
                elif Strategy == '3_Cand_ABC':
                    Strategy = '3_cand_ABC'
                elif Strategy == 'Reds Brahmos':
                    Strategy = 'brahmos'
                elif Strategy == 'Reds Rocket':
                    Strategy = 'rocket'
                elif Strategy == 'Sweths Violation':
                    Strategy = 'violation'
                else:
                    Strategy = Strategy.lower()

                # Check if the communication has already been sent for the given datetime, strategy, and stock
                filter_query = {
                    'Datetime': filter_ml_pred_row['Datetime'].iloc[0].strftime("%d-%m-%Y %H:%M:%S"),  # Convert datetime to string format
                    'Strategy': filter_ml_pred_row.loc[0,"Strategy"],
                    'stock': stock,
                    'communication_sent': 1  # Assuming 'communication_sent' is the column that indicates if the communication was sent
                }

                print(filter_query)

                existing_communication = collection_realtime_machine_learning_data.find_one(filter_query)

                if existing_communication:
                    # Communication already sent, skip the entry
                    print("Communication already sent for:", row['Datetime'], row['Strategy'], row['stock'])
                    continue
                else:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    logistic_min_max = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_logistic_regression_min_max_scalar.pkl")
                    logistic_max_abs = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_logistic_regression_max_abs_scalar.pkl")
                    random_forest_min_max = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_rf_min_max_scalar.pkl")
                    random_forest_max_abs = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_rf_max_abs_scalar.pkl")
                    xg_boost_min_max = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_xg_boost_min_max_scalar.pkl")
                    xg_boost_max_abs = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_xg_boost_max_abs_scalar.pkl")
                    light_gbm_min_max = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_light_gbm_min_max_scalar.pkl")
                    cnn_model_file = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_best_model_cnn.h5")

                    scaler_min_max = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_scaler_min_max_scalar.pkl")
                    scaler_max_abs = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_scaler_max_abs_scalar.pkl")
                    scaler_cnn = os.path.join(current_dir, "US_Stock_Models", stock, f"{Strategy}_scaler_cnn.pkl")


                    # Load the scalers
                    min_max_scalar = joblib.load(scaler_min_max)
                    max_abs_scalar = joblib.load(scaler_max_abs)
                    cnn_scalar = joblib.load(scaler_cnn)

                    # Load the input values
                    input_values = filter_ml_pred_row[features]
                    input_array = np.array(input_values)

                    # Perform scaling on the input array
                    input_scaled_min_max = min_max_scalar.transform(input_array)
                    input_scaled_max_abs = max_abs_scalar.transform(input_array)

                    # Load the models
                    logistic_min_max = joblib.load(logistic_min_max)
                    logistic_max_abs = joblib.load(logistic_max_abs)
                    random_forest_min_max = joblib.load(random_forest_min_max)
                    random_forest_max_abs = joblib.load(random_forest_max_abs)
                    xg_boost_min_max = joblib.load(xg_boost_min_max)
                    xg_boost_max_abs = joblib.load(xg_boost_max_abs)
                    light_gbm_min_max = joblib.load(light_gbm_min_max)
                    cnn_model_file = load_model(cnn_model_file)

                    # Make predictions using the loaded model
                    predictions_logistic_min_max = logistic_min_max.predict(input_scaled_min_max)
                    predictions_logistic_max_abs = logistic_max_abs.predict(input_scaled_max_abs)
                    predictions_random_forest_min_max = random_forest_min_max.predict(input_scaled_min_max)
                    predictions_random_forest_max_abs = random_forest_max_abs.predict(input_scaled_max_abs)
                    predictions_xg_boost_min_max = xg_boost_min_max.predict(input_scaled_min_max)
                    predictions_xg_boost_max_abs = xg_boost_max_abs.predict(input_scaled_max_abs)
                    predictions_light_gbm_min_max = light_gbm_min_max.predict(input_scaled_min_max)
                    # predictions_light_gbm_max_abs = light_gbm_max_abs.predict(input_scaled_max_abs)
                    predictions_cnn_model_file = cnn_model_file.predict(input_scaled_min_max)


                    # Create a new DataFrame
                    output_df = pd.DataFrame()

                    # Add columns for the model predictions
                    output_df['Logistic_MinMax_Predictions'] = predictions_logistic_min_max
                    output_df['Logistic_MaxAbs_Predictions'] = predictions_logistic_max_abs
                    output_df['RandomForest_MinMax_Predictions'] = predictions_random_forest_min_max
                    output_df['RandomForest_MaxAbs_Predictions'] = predictions_random_forest_max_abs
                    output_df['XGBoost_MinMax_Predictions'] = predictions_xg_boost_min_max
                    output_df['XGBoost_MaxAbs_Predictions'] = predictions_xg_boost_max_abs
                    output_df['LightGBM_MinMax_Predictions'] = predictions_light_gbm_min_max
                    # output_df['LightGBM_MaxAbs_Predictions'] = predictions_light_gbm_max_abs
                    output_df['CNN_Predictions'] = (predictions_cnn_model_file > 0.5).astype(int)


                    print(output_df)

                    print(filter_ml_pred_row.columns)

                    # Create a DataFrame with the desired columns
                    communication_df = filter_ml_pred_row[['Datetime', 'stock', 'Strategy', 'Call', 'Value', 'points','buy_probability','sell_probability']]

                    communication_df['Value'] = round(communication_df['Value'],2)
                    communication_df['points'] = round(communication_df['points'],2)

                    # Convert 'Datetime' to datetime object
                    communication_df['Datetime'] = pd.to_datetime(communication_df['Datetime'], unit='ms')

                    # Format 'Datetime' column in dd-mm-yyyy hh:mm:ss format
                    communication_df['Datetime'] = communication_df['Datetime'].dt.strftime("%d-%m-%Y %H:%M:%S")

                    # Convert Call value to Buy or Sell
                    communication_df['Call'] = communication_df['Call'].apply(lambda x: 'Buy' if x == 1 else 'Sell')

                    # Convert communication_df to JSON
                    communication_json = communication_df.to_json(orient='records')

                    # Construct the payload
                    payload = {
                        "content": communication_json,
                        "nonce": "1123400874408804352",
                        "tts": False,
                        "flags": 0
                    }

                    # Send the POST request to update the Discord channel
                    response = requests.post(discord_url, headers=discord_headers, json=payload)

                    # Calculate the count of 1s and 0s
                    count_1 = output_df.sum().sum()
                    count_0 = output_df.shape[1] - count_1

                    # Format the message
                    message = f"Logistic_MinMax_Predictions  ->  {'Target will hit' if output_df['Logistic_MinMax_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"Logistic_MaxAbs_Predictions  ->  {'Target will hit' if output_df['Logistic_MaxAbs_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"RandomForest_MinMax_Predictions  ->  {'Target will hit' if output_df['RandomForest_MinMax_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"RandomForest_MaxAbs_Predictions  ->  {'Target will hit' if output_df['RandomForest_MaxAbs_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"XGBoost_MinMax_Predictions  ->  {'Target will hit' if output_df['XGBoost_MinMax_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"XGBoost_MaxAbs_Predictions  ->  {'Target will hit' if output_df['XGBoost_MaxAbs_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"LightGBM_MinMax_Predictions  ->  {'Target will hit' if output_df['LightGBM_MinMax_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"CNN_Predictions  ->  {'Target will hit' if output_df['CNN_Predictions'][0] == 1 else 'Stoploss will hit'}\n" \
                            f"--------------------------------------------------------"

                    # Construct the payload
                    payload = {
                        "content": message,
                        "nonce": "1123400874408804352",
                        "tts": False,
                        "flags": 0
                    }

                    # Send the POST request to update the Discord channel
                    response = requests.post(discord_url, headers=discord_headers, json=payload)

                    if response.status_code == 200:
                        print("Message sent successfully!")
                        output_df['communication_sent'] = 1
                        output_df = pd.concat([communication_df[['Datetime','Strategy','stock']],filter_ml_pred_row[features], output_df], axis=1)
                        # Upload the row to the database
                        row_data = output_df.iloc[0].to_dict()
                        # row_data = {str(key): value for key, value in row_data.items()}  # Convert keys to strings
                        row_data['alert'] = True
                        row_data['execution_time'] = datetime.now(timezone("America/New_York")).strftime("%d-%m-%Y %H:%M:%S")
                        row_data['execution_date'] = datetime.now(timezone("America/New_York")).strftime("%d-%m-%Y")
                        # print(row_data)
                        collection_realtime_machine_learning_data.insert_one(row_data)
                        print("Data uploaded to the database successfully!")
                    else:
                        print("Failed to send the message. Status code:", response.status_code)
        except Exception as e:
            print("An error occurred:", str(e))

        