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
# import MySQLdb
# import pymysql
import pyotp


from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import create_engine, MetaData,\
Table, Column, Numeric, Integer, VARCHAR, update, delete

from sqlalchemy import create_engine
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import warnings
warnings.filterwarnings("ignore")

# engine = create_engine("mysql+pymysql://root:Mahadev_143@localhost/titania_trading")
# print(engine)

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

db = client["titania_trading"]

list_of_collections = db.list_collection_names()

print(list_of_collections)


# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()

# start_time = datetime.now(timezone("Asia/Kolkata")) + timedelta(hours=-10,minutes=30)
start_time = datetime.now(timezone("Asia/Kolkata"))
ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d')
 

# obj=SmartConnect(api_key="0mqmPUe7")
obj=SmartConnect(api_key="LPUVlRxd")
# totp = pyotp.TOTP("E76GLRMZGRVULRSD2VEEXIZTE4")
totp = pyotp.TOTP("ILBHGZB6KNXHZALKHZJN2A7PPI")
print("pyotp",totp.now())
attempts = 5
while attempts > 0:
    attempts = attempts-1
    
#     data = obj.generateSession("G304915","Acuvate@121", totp.now())
    # data = obj.generateSession("J95213","start@123", totp.now())
    data = obj.generateSession("J95213","2580", totp.now())
    if data['status']:
        break
    time.sleep(2)

print(data)

# # obj = SmartConnect(api_key="0Je28KUq")

# obj=SmartConnect(api_key="LPUVlRxd")
# # obj=SmartConnect(api_key="HShmNxno")

# # data = obj.generateSession("S1604557","bronick22@")
# data = obj.generateSession("J95213","start@123")

time.sleep(0.5)

now = datetime.now(timezone("Asia/Kolkata"))


intervals = ['ONE_MINUTE','FIVE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR','ONE_DAY']

interval = ['1m','5m','15m','30m','60m','1d']
  
tables = ['Stocks_data_1_minutes', 'Stocks_data_5_minutes', 'Stocks_data_15_minutes', 'Stocks_data_30_minutes', 'Stocks_data_60_minutes','Stocks_data_1_day']


# intervals = ['FIVE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR','ONE_DAY']

# interval = ['5m','15m','30m','60m','1d']
  
# tables = [ 'Stocks_data_5_minutes', 'Stocks_data_15_minutes', 'Stocks_data_30_minutes', 'Stocks_data_60_minutes','Stocks_data_1_day']
  
sql_df = pd.DataFrame(list(zip(intervals,interval, tables)),
               columns =['intervals','interval', 'tables'])

# print(sql_df)

def del_and_append_data(todays_data,instrumenttype,Stock,table_name):
    # sql = "select * from "+str(table_name)+" where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'  and Execution_Date = '" + str(ind_time) + "'"
    # print(sql)
    # df = pd.read_sql(sql,con=engine)

    # ## There is already todays data
    # if len(df) > 0:
    #     sql_Delete_query = "delete from "+str(table_name)+" where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'" 
    #     cursor.execute(sql_Delete_query)
    #     con.commit()
    #     # cursor.close()
    #     # con.close()
    sql_Delete_query = "delete from "+str(table_name)+" where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'" 
    cursor.execute(sql_Delete_query)
    con.commit()
    todays_data.to_sql(name=str(table_name),con=engine, if_exists='append', index=False)

def del_and_append_data_fut(todays_data,instrumenttype,Stock,table_name):
    sql = "select * from "+str(table_name)+" where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'  and Execution_Date = '" + str(ind_time) + "'"
    print(sql)
    df = pd.read_sql(sql,con=engine)

    ## There is already todays data
    if len(df) > 0:
        sql_Delete_query = "delete from "+str(table_name)+" where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'" 
        cursor.execute(sql_Delete_query)
        con.commit()
        # cursor.close()
        # con.close()

    todays_data.to_sql(name=str(table_name),con=engine, if_exists='append', index=False)


now = datetime.now(timezone("Asia/Kolkata"))

for idx in range(0,len(sql_df)):

    print(sql_df.loc[idx,])

    ## Fetching Nifty Data
    if str(sql_df.loc[idx,"intervals"]) == 'ONE_DAY':
        hist = {"exchange":"NFO",
            "symboltoken":"35003",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-60)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'ONE_HOUR':
        hist = {"exchange":"NFO",
            "symboltoken":"35003",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-30)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'THIRTY_MINUTE':
        hist = {"exchange":"NFO",
            "symboltoken":"35003",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-15)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'FIFTEEN_MINUTE':
        hist = {"exchange":"NFO",
            "symboltoken":"35003",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-10)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    else:
        hist = {"exchange":"NFO",
            "symboltoken":"35003",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    # hist = {"exchange":"NFO",
    #         "symboltoken":"53395",
    #         "interval":str(sql_df.loc[idx,"intervals"]),
    #         "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
    #         "todate":ind_time+ " 15:30"
    #        }

#     print(hist)

    resp = obj.getCandleData(hist)
    
#     print(resp['data'])

    if resp['data'] is not None:

    # obj.terminateSession("0mqmPUe7")

        hist_df = pd.DataFrame.from_dict(resp['data'])

        hist_df.columns = ['Datetime','Open', 'High','Low', 'Close','Volume']

        hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'], infer_datetime_format=True, utc=True )
        hist_df['Datetime'] = hist_df['Datetime'].dt.tz_convert('Asia/Kolkata')
        hist_df['Execution_Date'] = hist_df['Datetime'].dt.strftime('%Y-%m-%d')
        hist_df['Stock'] = "Nifty"
        hist_df['instrumenttype'] = "FUTIDX"
        
        print(hist_df.tail(5))

        # print(hist_df)
        # if str(sql_df.loc[idx,"intervals"]) == 'ONE_MINUTE' or str(sql_df.loc[idx,"intervals"]) == 'FIVE_MINUTE' or str(sql_df.loc[idx,"intervals"]) == 'FIFTEEN_MINUTE' or str(sql_df.loc[idx,"intervals"]) == 'THIRTY_MINUTE':
        #     todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]
        # else :
        #     todays_data = hist_df
        # if str(sql_df.loc[idx,"intervals"]) == 'ONE_DAY': 
        #     todays_data = hist_df
        # else:
        #     todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

        # todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]
        todays_data = hist_df

        # print(todays_data)

        if len(todays_data) > 0:
            todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]


            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    collection = db[str(sql_df.loc[idx,"tables"])]
                    # print(help(collection))
                    # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
                    print("Collection exists")
                    x = collection.delete_many({"Stock":"Nifty","instrumenttype":"FUTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")

            collection = db[str(sql_df.loc[idx,"tables"])]

            collection.insert_many(todays_data.to_dict('records'))

    #         del_and_append_data(todays_data,'FUTIDX','Nifty',str(sql_df.loc[idx,"tables"]))

        else:
            print("Market is closed")

    time.sleep(0.5)

    ## Fetching Bank Nifty DAta

    if str(sql_df.loc[idx,"intervals"]) == 'ONE_DAY':
        hist = {"exchange":"NFO",
            "symboltoken":"35002",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-60)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'ONE_HOUR':
        hist = {"exchange":"NFO",
            "symboltoken":"35002",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-30)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'THIRTY_MINUTE':
        hist = {"exchange":"NFO",
            "symboltoken":"35002",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-15)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    elif str(sql_df.loc[idx,"intervals"]) == 'FIFTEEN_MINUTE':
        hist = {"exchange":"NFO",
            "symboltoken":"35002",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-10)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }
    else:
        hist = {"exchange":"NFO",
            "symboltoken":"35002",
            "interval":str(sql_df.loc[idx,"intervals"]),
            "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
            "todate":ind_time+ " 15:30"
           }

    # hist = {"exchange":"NFO",
    #     "symboltoken":"53394",
    #     "interval":str(sql_df.loc[idx,"intervals"]),
    #     "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
    #     "todate":ind_time+ " 15:30"
    #    }

    # print(hist)
    
    if resp['data'] is not None:

        resp = obj.getCandleData(hist)

        # obj.terminateSession("0mqmPUe7")

        hist_df = pd.DataFrame.from_dict(resp['data'])

        hist_df.columns = ['Datetime','Open', 'High','Low', 'Close','Volume']

        hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'], infer_datetime_format=True, utc=True )
        hist_df['Datetime'] = hist_df['Datetime'].dt.tz_convert('Asia/Kolkata')
        hist_df['Execution_Date'] = hist_df['Datetime'].dt.strftime('%Y-%m-%d')
        hist_df['Stock'] = "BankNifty"
        hist_df['instrumenttype'] = "FUTIDX"

        # print(hist_df)

    #     con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
    #     cursor = con.cursor()

        # # todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]
        # if str(sql_df.loc[idx,"intervals"]) == 'ONE_DAY': 
        #     todays_data = hist_df
        # else:
        #     todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]
        # todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

        todays_data = hist_df

        if len(todays_data) > 0:
            print(todays_data.tail(5))
            todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]

            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    collection = db[str(sql_df.loc[idx,"tables"])]
                    print("Collection exists")
                    x = collection.delete_many({"Stock":"BankNifty","instrumenttype":"FUTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")


            collection = db[str(sql_df.loc[idx,"tables"])]
            collection.insert_many(todays_data.to_dict('records'))

    #         del_and_append_data(todays_data,'FUTIDX','BankNifty',str(sql_df.loc[idx,"tables"]))
        else:
            print("Market is closed")


    ###### Capturing the Index data
#     con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
#     cursor = con.cursor()

    data = yf.download(tickers='%5ENSEI', period="5d", interval=str(sql_df.loc[idx,"interval"]))
    data = pd.DataFrame(data)
    data.reset_index(level=0, inplace=True)
    print(data)
    if len(data) > 0:

        if str(sql_df.loc[idx,"interval"]) == "1d":
            data['Datetime'] = pd.to_datetime(data['Date'])
        else:
            data['Datetime'] = pd.to_datetime(data['Datetime'])

        data['Datetime'] = pd.to_datetime(data['Datetime'], infer_datetime_format=True, utc=True )
        data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
        data['Execution_Date'] = data['Datetime'].dt.strftime('%Y-%m-%d')

        data['Stock'] = 'Nifty'
        data['instrumenttype'] = "OPTIDX"


        todays_data = data.loc[data['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

        if len(todays_data) > 0:
            todays_data['Datetime'] = todays_data['Datetime'] - timedelta(hours=5,minutes=30)
            print(todays_data.tail(5))
            todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]
            
            

            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    collection = db[str(sql_df.loc[idx,"tables"])]
                    # print(help(collection))
                    # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
                    print("Collection exists")
                    x = collection.delete_many({"Stock":"Nifty","instrumenttype":"OPTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")

            collection = db[str(sql_df.loc[idx,"tables"])]

            collection.insert_many(todays_data.to_dict('records'))

#             del_and_append_data_fut(todays_data,'OPTIDX','Nifty',str(sql_df.loc[idx,"tables"]))
        else:
            print("Market is closed")

#     con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
#     cursor = con.cursor()

    data = yf.download(tickers='%5ENSEBANK', period="5d", interval=str(sql_df.loc[idx,"interval"]))
    data = pd.DataFrame(data)
    data.reset_index(level=0, inplace=True)

    print(data)
    if len(data) > 0 :
        if str(sql_df.loc[idx,"interval"]) == "1d":
            data['Datetime'] = pd.to_datetime(data['Date'])
        else:
            data['Datetime'] = pd.to_datetime(data['Datetime'])

        data['Datetime'] = pd.to_datetime(data['Datetime'], infer_datetime_format=True, utc=True )
        data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
        data['Execution_Date'] = data['Datetime'].dt.strftime('%Y-%m-%d')

        data['Stock'] = 'BankNifty'
        data['instrumenttype'] = "OPTIDX"


        todays_data = data.loc[data['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

        if len(todays_data) > 0:
            todays_data['Datetime'] = todays_data['Datetime'] - timedelta(hours=5,minutes=30)
            print(todays_data.tail(5))
            todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]
            
            
            
            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    collection = db[str(sql_df.loc[idx,"tables"])]
                    # print(help(collection))
                    # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
                    print("Collection exists")
                    x = collection.delete_many({"Stock":"BankNifty","instrumenttype":"OPTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")

            collection = db[str(sql_df.loc[idx,"tables"])]

            collection.insert_many(todays_data.to_dict('records'))


#             del_and_append_data_fut(todays_data,'OPTIDX','BankNifty',str(sql_df.loc[idx,"tables"]))
        else:
            print("Market is closed")



# hist = {"exchange":"NFO",
#         "symboltoken":"53395",
#         "interval":"FIVE_MINUTE",
#         "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
#         "todate":ind_time+ " 15:30"
#        }

# print(hist)
# resp = obj.getCandleData(hist)

# obj.terminateSession("J95213")

# hist_df = pd.DataFrame.from_dict(resp['data'])

# hist_df.columns = ['Datetime','Open', 'High','Low', 'Close','Volume']

# hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'], infer_datetime_format=True, utc=True )
# hist_df['Datetime'] = hist_df['Datetime'].dt.tz_convert('Asia/Kolkata')
# hist_df['Execution_Date'] = hist_df['Datetime'].dt.strftime('%Y-%m-%d')
# hist_df['Stock'] = "Nifty"
# hist_df['instrumenttype'] = "FUTIDX"

# # print(hist_df)

# todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]


# def del_and_append_data(todays_data,instrumenttype,Stock):
#     sql = "select * from titania_trading.Stocks_data_5_minutes where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"'  and Execution_Date = '" + str(ind_time) + "'"
#     print(sql)
#     df = pd.read_sql(sql,con=engine)

#     ## There is already todays data
#     if len(df) > 0:
#         sql_Delete_query = "delete from titania_trading.Stocks_data_5_minutes where instrumenttype = '"+str(instrumenttype)+"' and Stock = '"+str(Stock)+"' and Execution_Date = '" + str(ind_time) + "'" 
#         cursor.execute(sql_Delete_query)
#         con.commit()
#         cursor.close()
#         con.close()

#     todays_data.to_sql(name='Stocks_data_5_minutes',con=engine, if_exists='append', index=False)

# if len(todays_data) > 0:

#     todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]

#     del_and_append_data(todays_data,'FUTIDX','Nifty')
# else:
#     print("Market is closed")


# time.sleep(0.5)

# now = datetime.now(timezone("Asia/Kolkata"))

# hist = {"exchange":"NFO",
#         "symboltoken":"53394",
#         "interval":"FIVE_MINUTE",
#         "fromdate":(datetime.now(timezone("Asia/Kolkata"))+ timedelta(days=-5)).strftime("%Y-%m-%d")+ " 09:15",
#         "todate":ind_time+ " 15:30"
#        }

# print(hist)
# resp = obj.getCandleData(hist)

# obj.terminateSession("J95213")

# hist_df = pd.DataFrame.from_dict(resp['data'])

# hist_df.columns = ['Datetime','Open', 'High','Low', 'Close','Volume']

# hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'], infer_datetime_format=True, utc=True )
# hist_df['Datetime'] = hist_df['Datetime'].dt.tz_convert('Asia/Kolkata')
# hist_df['Execution_Date'] = hist_df['Datetime'].dt.strftime('%Y-%m-%d')
# hist_df['Stock'] = "BankNifty"
# hist_df['instrumenttype'] = "FUTIDX"

# # print(hist_df)

# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()

# todays_data = hist_df.loc[hist_df['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

# if len(todays_data) > 0:
#     # print(todays_data)
#     todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]
#     del_and_append_data(todays_data,'FUTIDX','BankNifty')
# else:
#     print("Market is closed")



# ###### Capturing the Index data
# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()

# data = yf.download(tickers='%5ENSEI', period="5d", interval="5m")
# data = pd.DataFrame(data)
# data.reset_index(level=0, inplace=True)
# data['Datetime'] = pd.to_datetime(data['Datetime'])

# data['Datetime'] = pd.to_datetime(data['Datetime'], infer_datetime_format=True, utc=True )
# data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
# data['Execution_Date'] = data['Datetime'].dt.strftime('%Y-%m-%d')

# data['Stock'] = 'Nifty'
# data['instrumenttype'] = "OPTIDX"


# todays_data = data.loc[data['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

# if len(todays_data) > 0:
#     # print(todays_data)
#     todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]

#     del_and_append_data(todays_data,'OPTIDX','Nifty')
# else:
#     print("Market is closed")



# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()

# data = yf.download(tickers='%5ENSEBANK', period="5d", interval="5m")
# data = pd.DataFrame(data)
# data.reset_index(level=0, inplace=True)
# data['Datetime'] = pd.to_datetime(data['Datetime'])

# data['Datetime'] = pd.to_datetime(data['Datetime'], infer_datetime_format=True, utc=True )
# data['Datetime'] = data['Datetime'].dt.tz_convert('Asia/Kolkata')
# data['Execution_Date'] = data['Datetime'].dt.strftime('%Y-%m-%d')

# data['Stock'] = 'BankNifty'
# data['instrumenttype'] = "OPTIDX"


# todays_data = data.loc[data['Datetime'].dt.strftime('%Y-%m-%d') == ind_time]

# if len(todays_data) > 0:
#     # print(todays_data)
#     todays_data = todays_data[["Stock","Datetime","Open","High","Low","Close","Volume","instrumenttype","Execution_Date"]]

#     del_and_append_data(todays_data,'OPTIDX','BankNifty')
# else:
#     print("Market is closed")

end_time = datetime.now(timezone("Asia/Kolkata")) 

print(end_time)

print('Duration: {}'.format(end_time - start_time))

