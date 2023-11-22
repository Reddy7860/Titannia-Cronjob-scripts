import json
import pandas as pd
from datetime import timedelta, date
import datetime
from datetime import datetime
import os
import requests
from pytz import timezone 
import re
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import warnings
warnings.filterwarnings("ignore")
from nsepy import get_history
import datetime
from nsepy import get_history
import pandas as pd

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://ganeshreddyus786:Mahadev_143@cluster0.opyv2si.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

# Define the symbol of the Nifty Futures
symbol_list = ['BANKNIFTY',
'NIFTY',
'AARTIIND',
'ABB',
'ABBOTINDIA',
'ABCAPITAL',
'ABFRL',
'ACC',
'ADANIENT',
'ADANIPORTS',
'ALKEM',
'AMBUJACEM',
'APOLLOHOSP',
'APOLLOTYRE',
'ASHOKLEY',
'ASIANPAINT',
'ASTRAL',
'ATUL',
'AUBANK',
'AUROPHARMA',
'AXISBANK',
'BAJAJ-AUTO',
'BAJAJFINSV',
'BAJFINANCE',
'BALKRISIND',
'BALRAMCHIN',
'BANDHANBNK',
'BANKBARODA',
'BATAINDIA',
'BEL',
'BERGEPAINT',
'BHARATFORG',
'BHARTIARTL',
'BHEL',
'BIOCON',
'BOSCHLTD',
'BPCL',
'BRITANNIA',
'BSOFT',
'CANBK',
'CANFINHOME',
'CHAMBLFERT',
'CHOLAFIN',
'CIPLA',
'COALINDIA',
'COFORGE',
'COLPAL',
'CONCOR',
'COROMANDEL',
'CROMPTON',
'CUB',
'CUMMINSIND',
'DABUR',
'DALBHARAT',
'DEEPAKNTR',
'DELTACORP',
'DIVISLAB',
'DIXON',
'DLF',
'DRREDDY',
'EICHERMOT',
'ESCORTS',
'EXIDEIND',
'FEDERALBNK',
'FSL',
'GAIL',
'GLENMARK',
'GMRINFRA',
'GNFC',
'GODREJCP',
'GODREJPROP',
'GRANULES',
'GRASIM',
'GUJGASLTD',
'HAL',
'HAVELLS',
'HCLTECH',
'HDFC',
'HDFCAMC',
'HDFCBANK',
'HDFCLIFE',
'HEROMOTOCO',
'HINDALCO',
'HINDCOPPER',
'HINDPETRO',
'HINDUNILVR',
'HONAUT',
'IBULHSGFIN',
'ICICIBANK',
'ICICIGI',
'ICICIPRULI',
'IDEA',
'IDFC',
'IDFCFIRSTB',
'IEX',
'IGL',
'INDHOTEL',
'INDIACEM',
'INDIAMART',
'INDIGO',
'INDUSINDBK',
'INDUSTOWER',
'INFY',
'INTELLECT',
'IOC',
'IPCALAB',
'IRCTC',
'ITC',
'JINDALSTEL',
'JKCEMENT',
'JSWSTEEL',
'JUBLFOOD',
'KOTAKBANK',
'L&TFH',
'LALPATHLAB',
'LAURUSLABS',
'LICHSGFIN',
'LT',
'LTIM',
'LTTS',
'LUPIN',
'M&M',
'M&MFIN',
'MANAPPURAM',
'MARICO',
'MARUTI',
'MCDOWELL-N',
'MCX',
'METROPOLIS',
'MFSL',
'MGL',
'MOTHERSON',
'MPHASIS',
'MRF',
'MUTHOOTFIN',
'NATIONALUM',
'NAUKRI',
'NAVINFLUOR',
'NESTLEIND',
'NMDC',
'NTPC',
'OBEROIRLTY',
'OFSS',
'ONGC',
'PAGEIND',
'PEL',
'PERSISTENT',
'PETRONET',
'PFC',
'PIDILITIND',
'PIIND',
'PNB',
'POLYCAB',
'POWERGRID',
'PVR',
'RAIN',
'RAMCOCEM',
'RBLBANK',
'RECLTD',
'RELIANCE',
'SAIL',
'SBICARD',
'SBILIFE',
'SBIN',
'SHREECEM',
'SHRIRAMFIN',
'SIEMENS',
'SRF',
'SUNPHARMA',
'SUNTV',
'SYNGENE',
'TATACHEM',
'TATACOMM',
'TATACONSUM',
'TATAMOTORS',
'TATAPOWER',
'TATASTEEL',
'TCS',
'TECHM',
'TITAN',
'TORNTPHARM',
'TORNTPOWER',
'TRENT',
'TVSMOTOR',
'UBL',
'ULTRACEMCO',
'UPL',
'VEDL',
'VOLTAS',
'WHIRLPOOL',
'WIPRO',
'ZEEL',
'ZYDUSLIFE']

start_date = datetime.date(2023, 2, 27)
# end_date = datetime.date(2023, 3, 7)
end_date = datetime.date.today()
last_thursday = datetime.date(2023, 3, 29)

final_futures_data = pd.DataFrame()

for idx in range(0,len(symbol_list)):
    print(idx)
    if symbol_list[idx] in ['BANKNIFTY','NIFTY']:
        futures_data = get_history(symbol=symbol_list[idx], start=start_date, end=end_date, index=True, futures=True, expiry_date=last_thursday, series='EQ')
    else:
        futures_data = get_history(symbol=symbol_list[idx], start=start_date, end=end_date, index=False, futures=True, expiry_date=last_thursday, series='EQ')

    # Calculate the daily % change in price and % change in OI
    futures_data['% Price Change'] = futures_data['Close'].pct_change() * 100
    futures_data['% OI Change'] = futures_data['Open Interest'].pct_change() * 100
    
    print(futures_data.tail())

    # Set the threshold values for the signals and filters
    long_buildup_thresh = 0
    short_buildup_thresh = 0
    long_unwinding_thresh = 0
    short_covering_thresh = 0
    oi_change_filter = 10
    price_change_filter = 2

    # Initialize the signal and action columns to None
    futures_data['Signal'] = None
    futures_data['Action'] = None

    # Loop over the rows of the DataFrame and generate signals
    for i in range(1, len(futures_data)):
        # Calculate the % price change and % change in OI from the previous day
        price_change_pct = futures_data.iloc[i]['% Price Change']
        oi_change_pct = futures_data.iloc[i]['% OI Change']

        # Check for Long Buildup signal
        if price_change_pct > long_buildup_thresh and oi_change_pct > long_buildup_thresh:
            # Check for filter 1: % change in OI > oi_change_filter and % price change > price_change_filter (upward movement)
            if oi_change_pct > oi_change_filter and price_change_pct > price_change_filter:
                futures_data.at[futures_data.index[i], 'Signal'] = 'Long Buildup'
                futures_data.at[futures_data.index[i], 'Action'] = 'BUY'

        # Check for Short Buildup signal
        elif price_change_pct < short_buildup_thresh and oi_change_pct > short_buildup_thresh:
            # Check for filter 2: % change in OI > oi_change_filter and % price change < -price_change_filter (downward movement)
            if oi_change_pct > oi_change_filter and price_change_pct < -price_change_filter:
                futures_data.at[futures_data.index[i], 'Signal'] = 'Short Buildup'
                futures_data.at[futures_data.index[i], 'Action'] = 'SELL'

        # Check for Long Unwinding signal
        elif price_change_pct < long_unwinding_thresh and oi_change_pct < long_unwinding_thresh:
            # Check for filter 3: % change in OI < -oi_change_filter and % price change < -price_change_filter (downward movement)
            if oi_change_pct < -oi_change_filter and price_change_pct < -price_change_filter:
                futures_data.at[futures_data.index[i], 'Signal'] = 'Long Unwinding'
                futures_data.at[futures_data.index[i], 'Action'] = 'EXIT BUY'

        # Check for Short Covering signal
        elif price_change_pct > short_covering_thresh and oi_change_pct < short_covering_thresh:
            # Check for filter 4: % change in OI < -oi_change_filter and % price change > price_change_filter (upward movement)
            if oi_change_pct < -oi_change_filter and price_change_pct > price_change_filter:
                futures_data.at[futures_data.index[i], 'Signal'] = 'Short Covering'
                futures_data.at[futures_data.index[i], 'Action'] = 'EXIT SELL'

        # Check for Exit Buy signal
        elif (futures_data.iloc[i-1]['Signal'] == 'Short Buildup' and price_change_pct > 0) or \
             (futures_data.iloc[i-1]['Signal'] == 'Long Buildup' and price_change_pct < 0):
            futures_data.at[futures_data.index[i], 'Signal'] = 'Exit Buy'
            futures_data.at[futures_data.index[i], 'Action'] = 'BUY'

        # Check for Exit Sell signal
        elif (futures_data.iloc[i-1]['Signal'] == 'Long Buildup' and price_change_pct > 0) or \
             (futures_data.iloc[i-1]['Signal'] == 'Short Buildup' and price_change_pct < 0):
            futures_data.at[futures_data.index[i], 'Signal'] = 'Exit Sell'
            futures_data.at[futures_data.index[i], 'Action'] = 'SELL'
            
    futures_data.reset_index(inplace=True)
        
    if 'index' in futures_data.columns:
        futures_data = futures_data.drop(columns=['index'])
            
    final_futures_data = final_futures_data.append(futures_data)
    
df_not_null = final_futures_data[final_futures_data['Signal'].notnull()]

# Print the first few rows of the data
# print(nifty_futures_data.tail())

collection = db["Derivatives_Actual_Daily_Signals"]

x = collection.delete_many({})

print(x.deleted_count, " documents deleted.")

df_not_null = df_not_null.astype(str)

print(df_not_null)

collection.insert_many(df_not_null.to_dict('records'))

print("Data uploaded successfully")
