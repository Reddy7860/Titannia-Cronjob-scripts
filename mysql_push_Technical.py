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

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import create_engine, MetaData,\
Table, Column, Numeric, Integer, VARCHAR, update, delete


from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


start_time = datetime.now(timezone("Asia/Kolkata"))

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

db = client["titania_trading"]

list_of_collections = db.list_collection_names()

print(list_of_collections)


ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d')


intervals = ['ONE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR','ONE_DAY']

interval = ['1m','15m','30m','60m','1d']
  
tables = ['technical_indicator_1_minutes', 'technical_indicator_15_minutes', 'technical_indicator_30_minutes', 'technical_indicator_60_minutes','technical_indicator_1_day']

stock_tables = ['Stocks_data_1_minutes', 'Stocks_data_15_minutes', 'Stocks_data_30_minutes', 'Stocks_data_60_minutes','Stocks_data_1_day']


sql_df = pd.DataFrame(list(zip(intervals,interval, tables,stock_tables)),
               columns =['intervals','interval', 'tables',"stock_tables"])


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

for idx in range(0,len(sql_df)):
    
    try:
        print(sql_df.loc[idx,])

        stock = '%5ENSEI'
        collection = db[str(sql_df.loc[idx,"stock_tables"])]

        live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "Nifty"})
        live_data =  pd.DataFrame(list(live_data))

        live_data["Datetime"] = live_data["Datetime"] + timedelta(hours=5, minutes=30)
    #     live_data = live_data[['Datetime','Open', 'High','Low', 'Close','Volume','Execution_Date']]

        if sql_df.loc[idx,"intervals"] == "ONE_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-3:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #         sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+"  order by cast(Datetime as date) desc limit 3) a) order by Datetime asc"
        elif sql_df.loc[idx,"intervals"] == "FIVE_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-5:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 5) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 5) a)"
        elif sql_df.loc[idx,"intervals"] == "FIFTEEN_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-10:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 10) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 10) a)"
        elif sql_df.loc[idx,"intervals"] == "THIRTY_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-15:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 15) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 15) a)"
        elif sql_df.loc[idx,"intervals"] == "ONE_HOUR":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-30:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 30) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 30) a)"
        elif sql_df.loc[idx,"intervals"] == "ONE_DAY":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-40:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 40) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 40) a)"

        # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'Nifty' "

    #     print(sql)

        live_data.reset_index(inplace=True,drop=True)
        live_data['index'] = stock

    #     live_data = pd.read_sql(sql,con=engine)
    #     live_data.reset_index(level=0, inplace=True)
        live_data['Datetime'] = pd.to_datetime(live_data['Datetime'])
        live_data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        live_data['rsi'] = live_data.ta.rsi(close='Close',length = 14)
        live_data['sma_20'] = ta.sma(live_data["Close"], length=20)
        live_data.ta.bbands(close = 'Close', length=20, std=2,append = True)
        live_data['SMA_Call'] = live_data.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
        live_data['RSI_Call'] = live_data.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
        live_data['MACD_Call'] = live_data.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
        live_data['Pivot_Call'] = ''
        live_data['PCR_Call'] = ''
        # print(live_data)

    #     sql = "select distinct * from support_and_resistance where Stock = 'Nifty' and Execution_Date = (select max(Execution_Date) from support_and_resistance)"
    #     print(sql)
    #     technical_df = pd.read_sql(sql,con=engine)

        support_collection = db.support_and_resistance
        technical_df = support_collection.find({"Stock": "Nifty"})
        technical_df =  pd.DataFrame(list(technical_df))
        technical_df = technical_df.loc[technical_df['Execution_date'] == max(technical_df['Execution_date']),]
        technical_df.reset_index(inplace=True,drop=True)

        pivot_bc = technical_df.loc[0,"pivot_bc"]
        pivot_tc = technical_df.loc[0,"pivot_tc"]
        if stock == '%5ENSEI':
    #         technical_indicator_pcr = "select distinct * from technical_indicator_pcr where Stock = 'Nifty' order by Datetime desc"
    #         technical_data_pcr = pd.read_sql(technical_indicator_pcr,con=engine)
            live_data['Pivot_Call'] = live_data.apply(lambda x: 'Buy' if x['Close'] >= pivot_bc else 'Sell', axis=1)
            hist_df = live_data[['Datetime','Open', 'High','Low', 'Close','Volume']]
            hist_df.set_index(pd.DatetimeIndex(hist_df["Datetime"]), inplace=True)
            hist_df.ta.vwap(high='High', low='Low',close='Close',volume='Volume', append=True)
            hist_df.ta.supertrend(high='High',low='Low',close='Close',append=True)
            hist_df.reset_index(inplace=True,drop=True)
            print(hist_df.tail(5))
            print(live_data.tail(5))
            result = pd.merge(live_data, hist_df, on="Datetime")
            result.reset_index(inplace=True,drop=True)  

            result = result[['index', 'Stock', 'Datetime', 'Open_x', 'High_x', 'Low_x', 'Close_x','Volume_x', 'instrumenttype', 'Execution_Date', 'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','VWAP_D', 'SUPERT_7_3.0']]
            result.columns = ['index', 'Stock','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume','instrumenttype', 'Execution_Date','MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20','BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0','SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'VWAP_D','supertrend']
            # print(result)
            result['VWAP_D'] = result['VWAP_D'].replace(np.nan, 0)
            result['supertrend'] = result['supertrend'].replace(np.nan, 0)

            result['BB_Call'] = result.apply(lambda x: 'Buy' if x['Close'] <= x['BBL_20_2.0'] else 'Sell' if x['Close'] >= x['BBU_20_2.0'] else 'Neutral', axis=1)
            result['VWAP_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['VWAP_D'] else 'Sell',axis = 1)
            result['SuperTrend_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['supertrend'] else 'Sell',axis = 1)
            result['date'] = pd.to_datetime(result['Datetime'], format='%Y-%m-%d')

            result = result[['Stock','Execution_Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'instrumenttype','SMA_Call', 'RSI_Call',
            'MACD_Call', 'Pivot_Call', 'PCR_Call',
            'BB_Call', 'VWAP_Call', 'SuperTrend_Call']]
            # print(result)
            print("Maximum Execution Date")
            print(max(result['Execution_Date']))
            result = result[(result['Execution_Date'] == max(result['Execution_Date']))]
            # print("Todays result")
            # print(result)
            # print(technical_data_pcr.tail(5))
            result.reset_index(level=0, inplace=True,drop = True)
    #         result['PCR_Call'] = ""

            technical_collection = db.technical_indicator_pcr
            technical_data_pcr = technical_collection.find({"Stock": "Nifty"}).sort("Datetime", -1)
            technical_data_pcr =  pd.DataFrame(list(technical_data_pcr))

            technical_data_pcr['Datetime'] = technical_data_pcr['Datetime'].str[:-7]

            technical_data_pcr['Datetime'] = pd.to_datetime(technical_data_pcr['Datetime'], format='%Y-%m-%d %H:%M:%S')

            result=result.merge(technical_data_pcr[['Datetime','pcr_call']], on='Datetime', how='left')
            result['PCR_Call'] = result['pcr_call']

            result = result[['Stock', 'Execution_Date', 'Datetime', 'Open', 'High', 'Low', 'Close',
           'Volume', 'instrumenttype', 'SMA_Call', 'RSI_Call', 'MACD_Call',
           'Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call']]


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
            print(result.tail(5))

            technical_coll = db[str(sql_df.loc[idx,"tables"])]

            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    # print(help(collection))
                    # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
                    print("Collection exists")
                    x = technical_coll.delete_many({"Stock":"Nifty","instrumenttype":"FUTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")

            print(result.columns)

            temp_result = result
            temp_result['Execution_Date'] = pd.to_datetime(temp_result['Execution_Date'],format='%Y-%m-%d')



            technical_coll.insert_many(temp_result.to_dict('records'))

    #         del_and_append_data(result,ind_time,'Nifty',str(sql_df.loc[idx,"tables"]))


        stock = '%5ENSEBANK'

        collection = db[str(sql_df.loc[idx,"stock_tables"])]

        live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "BankNifty"})
        live_data =  pd.DataFrame(list(live_data))

        live_data["Datetime"] = live_data["Datetime"] + timedelta(hours=5, minutes=30)
    #     live_data = live_data[['Datetime','Open', 'High','Low', 'Close','Volume','Execution_Date']]

        if sql_df.loc[idx,"intervals"] == "ONE_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-3:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 3) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 3) a)"
        elif sql_df.loc[idx,"intervals"] == "FIVE_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-5:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 5) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 5) a)"
        elif sql_df.loc[idx,"intervals"] == "FIFTEEN_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-10:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 10) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 10) a)"
        elif sql_df.loc[idx,"intervals"] == "THIRTY_MINUTE":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-15:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 15) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 15) a)"
        elif sql_df.loc[idx,"intervals"] == "ONE_HOUR":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-30:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 30) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 30) a)"
        elif sql_df.loc[idx,"intervals"] == "ONE_DAY":
            execution_dates = sorted(live_data.Execution_Date.unique())
            execution_dates = execution_dates[-40:]
            live_data = live_data.loc[(live_data['Datetime'] >= min(execution_dates)),]
    #     	sql = "select distinct * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct cast(Datetime as date) from "+str(sql_df.loc[idx,"stock_tables"])+" order by cast(Datetime as date) desc limit 40) a) order by Datetime asc"
            # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' and Execution_Date in (select * from (select distinct Execution_Date from "+str(sql_df.loc[idx,"stock_tables"])+" limit 40) a)"

        # sql = "select * from "+str(sql_df.loc[idx,"stock_tables"])+" where instrumenttype = 'FUTIDX' and Stock = 'BankNifty' "
    #     print(sql)
    #     live_data = pd.read_sql(sql,con=engine)
        live_data.reset_index(level=0, inplace=True)
        live_data['index'] = stock

        live_data['Datetime'] = pd.to_datetime(live_data['Datetime'])
        live_data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        live_data['rsi'] = live_data.ta.rsi(close='Close',length = 14)
        live_data['sma_20'] = ta.sma(live_data["Close"], length=20)
        live_data.ta.bbands(close = 'Close', length=20, std=2,append = True)
        live_data['SMA_Call'] = live_data.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
        live_data['RSI_Call'] = live_data.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
        live_data['MACD_Call'] = live_data.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
        live_data['Pivot_Call'] = ''
        live_data['PCR_Call'] = ''
        # print(live_data)
    #     sql = "select distinct * from support_and_resistance where Stock = 'BankNifty' and Execution_Date = (select max(Execution_Date) from support_and_resistance)"
    #     print(sql)
    #     technical_df = pd.read_sql(sql,con=engine)
        technical_df = support_collection.find({"Stock": "BankNifty"})
        technical_df =  pd.DataFrame(list(technical_df))
        technical_df = technical_df.loc[technical_df['Execution_date'] == max(technical_df['Execution_date']),]
        technical_df.reset_index(inplace=True,drop=True)


        pivot_bc = technical_df.loc[0,"pivot_bc"]
        pivot_tc = technical_df.loc[0,"pivot_tc"]
        if stock == '%5ENSEBANK':
    #         technical_indicator_pcr = "select distinct * from technical_indicator_pcr where Stock = 'BankNifty' order by Datetime desc"
    #         technical_data_pcr = pd.read_sql(technical_indicator_pcr,con=engine)    	
            live_data['Pivot_Call'] = live_data.apply(lambda x: 'Buy' if x['Close'] >= pivot_bc else 'Sell', axis=1)
            hist_df = live_data[['Datetime','Open', 'High','Low', 'Close','Volume']]
            hist_df.set_index(pd.DatetimeIndex(hist_df["Datetime"]), inplace=True)
            hist_df.ta.vwap(high='High', low='Low',close='Close',volume='Volume', append=True)
            hist_df.ta.supertrend(high='High',low='Low',close='Close',append=True)
            hist_df.reset_index(inplace=True,drop=True)
            result = pd.merge(live_data, hist_df, on="Datetime")
            result.reset_index(inplace=True,drop=True)
            print(result.columns)
            result = result[['index', 'Stock', 'Datetime', 'Open_x', 'High_x', 'Low_x', 'Close_x','Volume_x', 'instrumenttype', 'Execution_Date', 'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','VWAP_D', 'SUPERT_7_3.0']]

            result.columns = ['index', 'Stock','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume','instrumenttype', 'Execution_Date','MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20','BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0','SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'VWAP_D','supertrend']
            result['VWAP_D'] = result['VWAP_D'].replace(np.nan, 0)
            result['supertrend'] = result['supertrend'].replace(np.nan, 0)
            result['BB_Call'] = result.apply(lambda x: 'Buy' if x['Close'] <= x['BBL_20_2.0'] else 'Sell' if x['Close'] >= x['BBU_20_2.0'] else 'Neutral', axis=1)
            result['VWAP_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['VWAP_D'] else 'Sell',axis = 1)
            result['SuperTrend_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['supertrend'] else 'Sell',axis = 1)
            result['date'] = pd.to_datetime(result['Datetime'], format='%Y-%m-%d')
            result = result[['Stock','Execution_Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
            'instrumenttype','SMA_Call', 'RSI_Call',
            'MACD_Call', 'Pivot_Call', 'PCR_Call',
            'BB_Call', 'VWAP_Call', 'SuperTrend_Call']]

            result = result[(result['Execution_Date'] == max(result['Execution_Date']))]
            print(max(result['Execution_Date']))
            # print(result)
            # print(result)
            result.reset_index(level=0, inplace=True,drop = True)
            technical_collection = db.technical_indicator_pcr
            technical_data_pcr = technical_collection.find({"Stock": "BankNifty"}).sort("Datetime", -1)
            technical_data_pcr =  pd.DataFrame(list(technical_data_pcr))

            technical_data_pcr['Datetime'] = technical_data_pcr['Datetime'].str[:-7]

            technical_data_pcr['Datetime'] = pd.to_datetime(technical_data_pcr['Datetime'], format='%Y-%m-%d %H:%M:%S')

            result=result.merge(technical_data_pcr[['Datetime','pcr_call']], on='Datetime', how='left')
            result['PCR_Call'] = result['pcr_call']


            result = result[['Stock', 'Execution_Date', 'Datetime', 'Open', 'High', 'Low', 'Close',
           'Volume', 'instrumenttype', 'SMA_Call', 'RSI_Call', 'MACD_Call',
           'Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call']]

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

            print(result.tail(5))

            technical_coll = db[str(sql_df.loc[idx,"tables"])]

            try:
                if str(sql_df.loc[idx,"tables"]) in list_of_collections:
                    collection = db[str(sql_df.loc[idx,"tables"])]
                    # print(help(collection))
                    # db.validate_collection(str(sql_df.loc[idx,"tables"]))  # Try to validate a collection
                    print("Collection exists")
                    x = technical_coll.delete_many({"Stock":"BankNifty","instrumenttype":"FUTIDX"})
                    # x = collection.delete_many({})
                    print(x.deleted_count, " documents deleted.")
            except pymongo.errors.OperationFailure:  # If the collection doesn't exist
                print("This collection doesn't exist")

            temp_result = result
            temp_result['Execution_Date'] = pd.to_datetime(temp_result['Execution_Date'],format='%Y-%m-%d')

            technical_coll.insert_many(temp_result.to_dict('records'))

    #         del_and_append_data(result,ind_time,'BankNifty',str(sql_df.loc[idx,"tables"]))
    except Exception as e:
        print(e)
        print("Exception in running")



end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))

