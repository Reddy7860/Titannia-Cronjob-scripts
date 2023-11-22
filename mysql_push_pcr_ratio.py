from smartapi import SmartConnect
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
# import mysql.connector as mysql
# import pymysql

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo

# import MySQLdb
# import pymysql
import warnings
warnings.filterwarnings("ignore")




from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import create_engine, MetaData,\
Table, Column, Numeric, Integer, VARCHAR, update, delete


from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# engine = create_engine("mysql+pymysql://root:Mahadev_143@localhost/titania_trading")
# print(engine)

start_time = datetime.now(timezone("Asia/Kolkata"))


# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()


ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d')

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

db = client["titania_trading"]

list_of_collections = db.list_collection_names()

print(list_of_collections)





def del_and_append_data(todays_data,ind_time,stock,table_name):
    sql = "select * from "+str(table_name)+" where Stock = '"+str(stock)+"' and Execution_Date = '" + str(ind_time) + "'"
    print(sql)
    df = pd.read_sql(sql,con=engine)

    ## There is already todays data
    if len(df) > 0:
        sql_Delete_query = "delete from "+str(table_name)+" where Stock = '"+str(stock)+"' and Execution_Date <= '" + str(ind_time) + "'" 
        cursor.execute(sql_Delete_query)
        con.commit()

    todays_data.to_sql(name=table_name,con=engine, if_exists='append', index=False)


stock = ['%5ENSEI','%5ENSEBANK']

final_futures_data = pd.DataFrame()

for i in stock:
    sql = ""
    print(i)
    collection = db.Stocks_data_15_minutes
    live_data = pd.DataFrame()
    if i == '%5ENSEI':
        live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "Nifty"}).sort("Datetime", -1)

        live_data =  pd.DataFrame(list(live_data))

#         sql = "select distinct * from Stocks_data_30_minutes where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from Stocks_data_30_minutes  order by cast(Datetime as date) desc limit 10) a) order by Datetime asc"
    else:
#         sql = "select distinct * from Stocks_data_30_minutes where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from Stocks_data_30_minutes  order by cast(Datetime as date) desc limit 10) a) order by Datetime asc"
        live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "BankNifty"}).sort("Datetime", -1)

        live_data =  pd.DataFrame(list(live_data))

#     print(sql)

#     live_data = pd.read_sql(sql,con=engine)
#     live_data.reset_index(level=0, inplace=True,drop = True)

    
    execution_dates = sorted(live_data.Execution_Date.unique())

    execution_dates = execution_dates[-10:]

    print(execution_dates)

    print(min(execution_dates))

    print(max(execution_dates))
    
    live_data = live_data.loc[live_data['Datetime'] >= min(execution_dates),]

    live_data["Datetime"] = live_data["Datetime"] + timedelta(hours=5, minutes=30)

    live_data = live_data[['Datetime','Open', 'High','Low', 'Close','Volume','Execution_Date']]

    live_data.reset_index(inplace=True,drop=True)
    
    print(live_data.head(5))
    
    live_data['Datetime'] = pd.to_datetime(live_data['Datetime'])
    live_data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    live_data['rsi'] = live_data.ta.rsi(close='Close',length = 14)
    live_data['sma_20'] = ta.sma(live_data["Close"], length=20)
    live_data.ta.bbands(close = 'Close', length=20, std=2,append = True)
    live_data['SMA_Call'] = live_data.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
    live_data['RSI_Call'] = live_data.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
    live_data['MACD_Call'] = live_data.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
    live_data['Pivot_Call'] = ''

#     filtered_live_data = live_data.dropna()
    filtered_live_data = live_data
    filtered_live_data.reset_index(level=0, inplace=True,drop = True)
    filtered_live_data['Datetime'] = pd.to_datetime(filtered_live_data['Datetime'], format='%Y-%m-%d %H:%M:%00')
    date_list = filtered_live_data['Execution_Date'].unique()

    
    for dt in range(0,len(date_list)):
        print(date_list[dt])
        ts = pd.to_datetime(str(date_list[dt]))
        try:
            futures_data = pd.DataFrame()
            stock = ''
            path = ''
            if i == '%5ENSEI':
#                 print("Fetchinng Nifty")
                path = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/Nifty/' + ts.strftime('%Y-%m-%d') +'_Futures_Options_Signals.csv'
                stock = 'Nifty'
            else:
#                 print("Fetching Bank Nifty")
                path = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/BankNifty/' + ts.strftime('%Y-%m-%d') +'_Futures_Options_Signals.csv'
                stock = 'BankNifty'
            futures_data = pd.read_csv(path)
            futures_data['pcr_call'] = ""
            futures_data['Stock'] = stock
#             print(futures_data.tail(5))
            final_futures_data = final_futures_data.append(futures_data)
        except Exception as e:
            print(e)
            
#         print(futures_data)
#         print(len(futures_data))
        if len(futures_data) > 0:
    #             print(final_futures_data)
            final_futures_data = final_futures_data.loc[:, ~final_futures_data.columns.str.contains('^Unnamed')]
    #             final_futures_data['Datetime'] = pd.to_datetime(final_futures_data['Datetime'])
            final_futures_data = final_futures_data.dropna(subset=['pcr_ratio'])
            final_futures_data.reset_index(inplace = True, drop = True)
    #             print(final_futures_data.describe())
            final_futures_data['pcr_ratio'] = final_futures_data['pcr_ratio'].astype(float)
    #             print(final_futures_data['pcr_ratio'])
            final_futures_data['pcr_call'] = final_futures_data['pcr_ratio'].apply(lambda x: 'Buy' if (x >=0 and x<=0.7) else 'Sell')

            final_futures_data = final_futures_data.sort_values(by='Datetime')
    print(final_futures_data.tail(5))
final_futures_data['Execution_Date'] = pd.to_datetime(final_futures_data['Datetime'], errors='coerce').dt.strftime('%Y-%m-%d')



# sql = "select * from technical_indicator_pcr where Execution_Date <= '" + str(ind_time) + "'"
# print(sql)
# df = pd.read_sql(sql,con=engine)

# print(len(df))


try:
    if "technical_indicator_pcr" in list_of_collections:
        collection = db["technical_indicator_pcr"]
        # print(help(collection))
        # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
        print("Collection exists")
        x = collection.delete_many({})
        # x = collection.delete_many({})
        print(x.deleted_count, " documents deleted.")
except pymongo.errors.OperationFailure:  # If the collection doesn't exist
    print("This collection doesn't exist")

collection = db["technical_indicator_pcr"]

collection.insert_many(final_futures_data.to_dict('records'))

## There is already todays data
# if len(df) > 0:
#     sql_Delete_query = "delete from technical_indicator_pcr where Execution_Date <= '" + str(ind_time) + "'" 
#     cursor.execute(sql_Delete_query)
#     con.commit()

# final_futures_data.to_sql(name="technical_indicator_pcr",con=engine, if_exists='append', index=False)

print(final_futures_data.columns)
print(final_futures_data.tail(5))


end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))