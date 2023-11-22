from datetime import datetime,timedelta
import pandas as pd
from pandasql import sqldf
import pandasql as pdsql
import os
from smartapi import SmartConnect
import time
import pyotp
from pytz import timezone

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo

# import mysql.connector as mysql
# import pymysql
from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import create_engine, MetaData,\
Table, Column, Numeric, Integer, VARCHAR, update, delete

from sqlalchemy import create_engine

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

start_time = datetime.now(timezone("Asia/Kolkata"))

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

db = client["titania_trading"]

list_of_collections = db.list_collection_names()

print(list_of_collections)

pysqldf = lambda q: sqldf(q, globals())

today_now = datetime.now(timezone("Asia/Kolkata"))

print(today_now)

expiry_date = today_now
print(today_now.strftime("%w"))

if today_now.strftime("%w") == '1':
    expiry_date = today_now + timedelta(days=3)
elif today_now.strftime("%w") == '2':
    expiry_date = today_now + timedelta(days=2)
elif today_now.strftime("%w") == '3':
    expiry_date = today_now + timedelta(days=1)
elif today_now.strftime("%w") == '4':
    expiry_date = today_now + timedelta(days=0)
elif today_now.strftime("%w") == '5':
    expiry_date = today_now + timedelta(days=6)
elif today_now.strftime("%w") == '6':
    expiry_date = today_now + timedelta(days=5)
elif today_now.strftime("%w") == '7':
    expiry_date = today_now + timedelta(days=4)

print("Expiry date")
print(expiry_date.strftime("%d-%b-%Y"))

# expiry_date = '27-Oct-2022'
expiry_date = expiry_date.strftime("%d-%b-%Y")

nse_data = pd.DataFrame([["BANKNIFTY","%5ENSEBANK","BANKNIFTY-EQ"],["Nifty50","%5ENSEI","Nifty50-EQ"]],columns=["Symbol","Yahoo_Symbol","TradingSymbol"])

### Tells to run the command or not
talk_Command = 'No'


starting_strike_price = 0
ending_strike_price = 0
seq = 0
symbol = ""

collection = db.Stocks_data_1_minutes

final_data = pd.DataFrame()

current_time = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")

if current_time >= "09:15:00":
    for nse_cnt in range(0,len(nse_data)):
        todays_date = (datetime.now(timezone("Asia/Kolkata"))).strftime("%Y-%m-%d")
        print(todays_date)

        print(nse_data.loc[nse_cnt,"Symbol"])
        
        collection = db.Stocks_data_1_minutes
        live_data = pd.DataFrame()

        if nse_data.loc[nse_cnt,"Symbol"] == "BANKNIFTY":
            seq = 100
            symbol = "BankNifty"
            live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "BankNifty"})
            live_data =  pd.DataFrame(list(live_data))


        else:
            seq = 50
            symbol = "Nifty"
            live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "Nifty"})
            live_data =  pd.DataFrame(list(live_data))
        live_data = live_data.loc[live_data['Execution_Date'] == max(live_data.Execution_Date),]
        
        live_data["Datetime"] = live_data["Datetime"] + timedelta(hours=5, minutes=30)

        hist_df = live_data[['Datetime','Open', 'High','Low', 'Close','Volume']]
        
        hist_df = hist_df.drop_duplicates(keep='first')

        hist_df.reset_index(inplace=True,drop=True) 

        if symbol == "BankNifty":
            future_data = hist_df[['Datetime','Close','Volume']]
            futures_latest_close = int(future_data.tail(1)['Close'])

            print("futures_latest_close")
            print(futures_latest_close)
            future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], format='%Y-%m-%d %H:%M:%00')

            strike_price = (futures_latest_close + (100 - futures_latest_close % 100)) if futures_latest_close % 100 > 50 else (futures_latest_close - futures_latest_close % 100)
            print("latest bnf strike : ",strike_price)

            starting_strike_price = strike_price - 1000
            ending_strike_price = strike_price + 1000

            def label_bnf_strike(row):
                # print(row)
                if row['Close'] % 100 > 50:
                    strike = row['Close'] + (100 - row['Close'] % 100)
                    strike = str(strike)
                    return (strike)
                else:
                    strike = row['Close'] - row['Close'] % 100
                    strike = str(strike)
                    return (strike)
            future_data['Strike_Price'] = future_data.apply(lambda row: label_bnf_strike(row), axis=1)
        else:
            future_data = hist_df[['Datetime','Close','Volume']]
            futures_latest_close = int(future_data.tail(1)['Close'])
            print("futures_latest_close")
            print(futures_latest_close)
            future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], format='%Y-%m-%d %H:%M:%00')

            strike_price = (futures_latest_close + (50 - futures_latest_close % 50)) if futures_latest_close % 50 > 25 else (futures_latest_close - futures_latest_close % 50)
            print("latest nifty strike : ",strike_price)

            starting_strike_price = strike_price - 500
            ending_strike_price = strike_price + 500

            def label_nf_strike(row):
                if row['Close'] % 50 > 25:
                    strike = row['Close'] + (50 - row['Close'] % 50)
                    strike = str(strike)
                    return (strike)
                else:
                    strike = row['Close'] - row['Close'] % 50
                    strike = str(strike)
                    return (strike)
            future_data['Strike_Price'] = future_data.apply(lambda row: label_nf_strike(row), axis=1)

        print(future_data.head(5))

        main_data = pd.DataFrame()

        for i in range(starting_strike_price,ending_strike_price,seq):
            # print(i)
            current_data = pd.read_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+str(expiry_date)+"/"+str(i)+'.csv',index_col=False)
            current_data.drop(['level_0', 'index'], axis=1, inplace=True)
            main_data = pd.concat([main_data,current_data])
            main_data.reset_index(inplace = True, drop = True)
            
        from datetime import datetime

        main_data['time'] = main_data['time'].apply(lambda x: x if len(x) == 26 else x[:26])

        main_data['time'] = pd.to_datetime(main_data['time']).dt.strftime('%Y-%m-%d %H:%M:00')

        main_data['time'] = pd.to_datetime(main_data['time'])
        
        print(sorted(main_data['time'].dt.strftime('%Y-%m-%d').unique()))
        print(todays_date)
        # main_data = main_data.loc[main_data['time'] <= '2021-12-13 15:00:00']

        main_data = main_data.loc[main_data['time'].dt.strftime('%Y-%m-%d') == todays_date]

        print("Actual Main")
        print(main_data.tail())

        main_data.reset_index(inplace = True, drop = True)

        main_data = main_data.drop_duplicates(keep='first')
        print(main_data)
        final_data = final_data.append(main_data)

    if len(final_data) > 0 :
        final_data.reset_index(inplace=True,drop=True)
        try:
            if 'today_options_data' in list_of_collections:
                collection = db["today_options_data"]
                print("Collection exists")
                x = collection.delete_many({})
                print(x.deleted_count, " documents deleted.")
        except pymongo.errors.OperationFailure:  # If the collection doesn't exist
            print("This collection doesn't exist")

        collection = db["today_options_data"]

        collection.insert_many(final_data.to_dict('records'))

        print("Data Replaced successfully")
else:
    print("Market is Closed")


end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print("Duration: {}".format(end_time - start_time))
    