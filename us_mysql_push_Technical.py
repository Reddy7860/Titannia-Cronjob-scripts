from datetime import datetime,timedelta
import pandas as pd
import time
from pytz import timezone 
import yfinance as yf
import pandas_ta as ta
import numpy as np
import pandas_ta as ta

from pandas.io import sql
from pandasql import sqldf

from pymongo import MongoClient
import pymongo

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

start_time = datetime.now()
print("Script execution started")
print(start_time)

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE")
db = client["United_States_Titania_Trading"]

list_of_collections = db.list_collection_names()
print(list_of_collections)

intervals = ['FIVE_MINUTE']
interval = ['5m']
tables = ['technical_indicator_5_minutes']
stock_tables = ['Stocks_data_5_minutes']
sql_df = pd.DataFrame(list(zip(intervals,interval, tables,stock_tables)),
               columns =['intervals','interval', 'tables',"stock_tables"])


# us_stocks_data = pd.read_csv("/Users/apple/Downloads/Reddy_Stocks_Application/data/US - 30 Stocks.csv")
us_stocks_data = pd.read_csv("/home/sjonnal3/Hate_Speech_Detection/Trading_Application/US - 30 Stocks.csv")

collection = db['technical_indicator_5_minutes']

for stk in range(0,len(us_stocks_data)):
    print(us_stocks_data.loc[stk,])
    stock = us_stocks_data.loc[stk,"Symbol"]
    data = yf.download(tickers=stock, period="2d", interval="5m")
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    data = data.rename(columns = {'index':'Datetime'})
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    data['rsi'] = data.ta.rsi(close='Close',length = 14)
    data['sma_20'] = ta.sma(data["Close"], length=20)
    data.ta.bbands(close = 'Close', length=20, std=2,append = True)
    data['SMA_Call'] = data.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
    data['RSI_Call'] = data.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
    data['MACD_Call'] = data.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
    data['Pivot_Call'] = ''
    data['PCR_Call'] = ''
    
    nifty_support_data = (
        db["support_and_resistance"]
        .find({"Stock": stock})
        .sort([("Execution_date", -1)])
        .limit(1)
    )
    nifty_support_data = pd.DataFrame(list(nifty_support_data))

    pivot_bc = nifty_support_data.loc[0,"pivot_bc"]
    pivot_tc = nifty_support_data.loc[0,"pivot_tc"]
    
    data['Pivot_Call'] = data.apply(lambda x: 'Buy' if x['Close'] >= pivot_bc else 'Sell', axis=1)
    hist_df = data[['Datetime','Open', 'High','Low', 'Close','Volume']]
    hist_df.set_index(pd.DatetimeIndex(hist_df["Datetime"]), inplace=True)
    hist_df.ta.vwap(high='High', low='Low',close='Close',volume='Volume', append=True)
    hist_df.ta.supertrend(high='High',low='Low',close='Close',append=True)
    hist_df.reset_index(inplace=True,drop=True)
    result = pd.merge(data, hist_df, on="Datetime")
    result.reset_index(inplace=True,drop=True)

    result = result[[ 'Datetime', 'Open_x', 'High_x', 'Low_x', 'Close_x','Volume_x', 'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','VWAP_D', 'SUPERT_7_3.0']]
    result.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume','MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20','BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0','SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'VWAP_D','supertrend']
    # print(result)
    result['VWAP_D'] = result['VWAP_D'].replace(np.nan, 0)
    result['supertrend'] = result['supertrend'].replace(np.nan, 0)

    result['BB_Call'] = result.apply(lambda x: 'Buy' if x['Close'] <= x['BBL_20_2.0'] else 'Sell' if x['Close'] >= x['BBU_20_2.0'] else 'Neutral', axis=1)
    result['VWAP_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['VWAP_D'] else 'Sell',axis = 1)
    result['SuperTrend_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['supertrend'] else 'Sell',axis = 1)

    result = result[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume','SMA_Call', 'RSI_Call',
    'MACD_Call', 'Pivot_Call', 'PCR_Call',
    'BB_Call', 'VWAP_Call', 'SuperTrend_Call']]
    
    result['Date'] = pd.to_datetime(result['Datetime']).dt.strftime('%Y-%m-%d')
    result[result['Date'] == max(result['Datetime']).strftime('%Y-%m-%d')]
    
    result.reset_index(level=0, inplace=True,drop = True)
    
    for row in range(0,len(result)):
        buy_probability = 0
        sell_probability = 0
        
        if result.loc[row,'SMA_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'SMA_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

        if result.loc[row,'RSI_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'RSI_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

        if result.loc[row,'MACD_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'MACD_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

        if result.loc[row,'Pivot_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'Pivot_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

        if result.loc[row,'BB_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'BB_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

#         if result.loc[row,'PCR_Call'] == 'Buy':
#             buy_probability = buy_probability + 14.28
#         elif result.loc[row,'PCR_Call'] == 'Sell':
#             sell_probability = sell_probability + 14.28

        if result.loc[row,'VWAP_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'VWAP_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28

        if result.loc[row,'SuperTrend_Call'] == 'Buy':
            buy_probability = buy_probability + 14.28
        elif result.loc[row,'SuperTrend_Call'] == 'Sell':
            sell_probability = sell_probability + 14.28


        result.loc[row,'buy_probability'] = buy_probability
        result.loc[row,'sell_probability'] = sell_probability
        
    print(result.tail())
    
    result['Stock'] = stock
    
    x = collection.delete_many({"Stock":stock})
    print(x.deleted_count, " documents deleted.")
    
    collection.insert_many(result.to_dict('records'))
    
end_time = datetime.now()

print(end_time)

print("Duration: {}".format(end_time - start_time))
