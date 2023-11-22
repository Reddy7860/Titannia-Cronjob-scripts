import pandas as pd
import numpy as np
from datetime import datetime
import smartapi
import yfinance as yf
from smartapi import SmartConnect
import json
import requests
import datetime
from datetime import datetime, timedelta
import time
import os.path
from pytz import timezone
import os
import math
from pandasql import sqldf
import pandasql as pdsql
import warnings
from datetime import timedelta
import pandas_ta as ta

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pyotp
from smartapi import SmartConnect
from google.oauth2 import service_account


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


final_data_one_minute = pd.DataFrame()
final_data_five_minute = pd.DataFrame()
final_data_fifteen_minute = pd.DataFrame()
final_data_thirty_minute = pd.DataFrame()
final_data_one_hour = pd.DataFrame()


now = datetime.now()
start_time = datetime.now()
# today_now = datetime.now(timezone("Asia/Kolkata")) 
today_now = datetime.now() 
expiry_date = today_now

if today_now.strftime("%w") == "1":
    expiry_date = today_now + timedelta(days=3)
elif today_now.strftime("%w") == "2":
    expiry_date = today_now + timedelta(days=9)
elif today_now.strftime("%w") == "3":
    expiry_date = today_now + timedelta(days=8)
elif today_now.strftime("%w") == "4":
    expiry_date = today_now + timedelta(days=7)
elif today_now.strftime("%w") == "5":
    expiry_date = today_now + timedelta(days=6)
elif today_now.strftime("%w") == "6":
    expiry_date = today_now + timedelta(days=5)
elif today_now.strftime("%w") == "7":
    expiry_date = today_now + timedelta(days=4)

print("Expiry date")
print(expiry_date)
# expiry_date = '25-01-2023 15:00:00'
# expiry_date = datetime.strptime(expiry_date, '%d-%m-%Y %H:%M:%S')

expiry_date_char = expiry_date.strftime("%Y-%m-%d")
expiry_date_month = expiry_date.strftime("%d%b%y").upper()

angel_script = pd.read_csv("/home/sjonnal3/Hate_Speech_Detection/Trading_Application/angel_script.csv", index_col=False)
credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/ferrous-module-376519-7e08f583402d.json',
)


intervals = ['ONE_MINUTE','FIVE_MINUTE','FIFTEEN_MINUTE','THIRTY_MINUTE','ONE_HOUR']

server_api = ServerApi("1")

client = MongoClient(
    "mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE",
    server_api=server_api,
)

db = client["titania_trading"]

obj = SmartConnect(api_key="LPUVlRxd")
totp = pyotp.TOTP("ILBHGZB6KNXHZALKHZJN2A7PPI")
print("pyotp", totp.now())
attempts = 5
try:
    while attempts > 0:
        attempts = attempts - 1
        # data = obj.generateSession("J95213", "start@123", totp.now())
        data = obj.generateSession("J95213", "2580", totp.now())
        if data["status"]:
            # print(data)
            break
        time.sleep(2)
# except smartapi.smartExceptions.DataException:

except Exception as e:
    print("Failure Exception : " + str(e))


def upload_data_gbq(data,destination_table,replace_type):
    try:
        data.to_gbq(destination_table=destination_table,
            project_id='ferrous-module-376519',
           if_exists=str(replace_type),credentials=credentials)
        # logging.info("data ",str(replace_type)," to table : %s",destination_table)
        print("data ",str(replace_type)," successfully : "+str(destination_table))
    except Exception as e:
        print(e)
        # logging.error("Exception occured in ",str(replace_type)," %s", destination_table)
        print("Error while ",str(replace_type)," : "+str(destination_table))

indices = ['Nifty','BankNifty']

for index in range(0,len(indices)):
    print(indices[index])
    print(f"Running for {indices[index]}")

    collection = db["Stocks_data_5_minutes"]

    data = collection.find({"Stock":str(indices[index]),"instrumenttype":"OPTIDX"})

    data =  pd.DataFrame(list(data))

    data = data[['Datetime','Open','High','Low','Close','Stock']]

    data["Datetime"] = data["Datetime"] + timedelta(hours=5, minutes=30)

    data["Spot_Price"] = data.apply(lambda x: (x["Close"] + (50 - x["Close"] % 50) if x["Close"] % 50 > 25 else (x["Close"] - x["Close"] % 50)) if x["Stock"] == "Nifty" else (x["Close"] + (100 - x["Close"] % 100) if x["Close"] % 100 > 50 else (x["Close"] - x["Close"] % 100)), axis=1).round(decimals=0)

    print(data.head())

    lowest_price = min(data['Spot_Price'])
    highest_price = max(data['Spot_Price'])

    if indices[index] == "Nifty":
        strikes_prices = list(range(int(lowest_price), int(highest_price), 50))
    else:
        strikes_prices = list(range(int(lowest_price), int(highest_price), 100))

    call_tokens = []
    put_tokens = []

    for idx in range(0,len(strikes_prices)):
        print(strikes_prices[idx])

        if indices[index] == "Nifty":
            lookup_symbol_ce = (
                        "NIFTY"
                        + str(expiry_date_month)
                        + str(strikes_prices[idx])
                        + "CE"
                        )
            lookup_symbol_pe = (
                        "NIFTY"
                        + str(expiry_date_month)
                        + str(strikes_prices[idx])
                        + "PE"
                        )
        else:
            lookup_symbol_ce = (
                        "BANKNIFTY"
                        + str(expiry_date_month)
                        + str(strikes_prices[idx])
                        + "CE"
                        )
            lookup_symbol_pe = (
                        "BANKNIFTY"
                        + str(expiry_date_month)
                        + str(strikes_prices[idx])
                        + "PE"
                        )
        print(lookup_symbol_ce)
        print(lookup_symbol_pe)

        current_script_ce = angel_script[angel_script["symbol"] == lookup_symbol_ce]
        current_script_pe = angel_script[angel_script["symbol"] == lookup_symbol_pe]

        current_script_ce.reset_index(inplace=True, drop=True)
        current_script_pe.reset_index(inplace=True, drop=True)
        
        if len(current_script_ce) > 0:
            
            token_ce = current_script_ce.loc[0, "token"]
            token_pe = current_script_pe.loc[0, "token"]

            call_tokens.append(token_ce)
            put_tokens.append(token_pe)

            for inter in range(0,len(intervals)):
                hist_ce = {
                    "exchange": "NFO",
                    "symboltoken": token_ce,
                    "interval": str(intervals[inter]),
                    "fromdate": now.strftime("%Y-%m-%d") + " 09:15",
                    "todate": now.strftime("%Y-%m-%d") + " 15:30",
                }
                print(hist_ce)
                resp_ce = obj.getCandleData(hist_ce)
                time.sleep(0.5)
                # print(resp)
                hist_df_ce = pd.DataFrame.from_dict(resp_ce["data"])

                hist_df_ce.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

                hist_df_ce["Datetime"] = pd.to_datetime(
                    hist_df_ce["Datetime"], format="%Y-%m-%d %H:%M:%S"
                )

                hist_df_ce['token'] = token_ce
                hist_df_ce['lookup'] = lookup_symbol_ce

                hist_pe = {
                        "exchange": "NFO",
                        "symboltoken": token_pe,
                        "interval": str(intervals[inter]),
                        "fromdate": now.strftime("%Y-%m-%d") + " 09:15",
                        "todate": now.strftime("%Y-%m-%d") + " 15:30",
                    }

                resp_pe = obj.getCandleData(hist_pe)

                time.sleep(0.5)
                # print(resp)
                hist_df_pe = pd.DataFrame.from_dict(resp_pe["data"])

                hist_df_pe.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

                hist_df_pe["Datetime"] = pd.to_datetime(
                    hist_df_pe["Datetime"], format="%Y-%m-%d %H:%M:%S"
                )
                hist_df_pe['token'] = token_pe
                hist_df_pe['lookup'] = lookup_symbol_pe

                if str(intervals[inter]) == "ONE_MINUTE":
                    final_data_one_minute = final_data_one_minute.append(hist_df_ce)
                    final_data_one_minute = final_data_one_minute.append(hist_df_pe)
                elif str(intervals[inter]) == "FIVE_MINUTE":
                    final_data_five_minute = final_data_five_minute.append(hist_df_ce)
                    final_data_five_minute = final_data_five_minute.append(hist_df_pe)
                elif str(intervals[inter]) == "FIFTEEN_MINUTE":
                    final_data_fifteen_minute = final_data_fifteen_minute.append(hist_df_ce)
                    final_data_fifteen_minute = final_data_fifteen_minute.append(hist_df_pe)
                elif str(intervals[inter]) == "THIRTY_MINUTE":
                    final_data_thirty_minute = final_data_thirty_minute.append(hist_df_ce)
                    final_data_thirty_minute = final_data_thirty_minute.append(hist_df_pe)
                elif str(intervals[inter]) == "ONE_HOUR":
                    final_data_one_hour = final_data_one_hour.append(hist_df_ce)
                    final_data_one_hour = final_data_one_hour.append(hist_df_pe)
                    

final_data_one_minute.reset_index(inplace=True,drop=True)
final_data_five_minute.reset_index(inplace=True,drop=True)
final_data_fifteen_minute.reset_index(inplace=True,drop=True)
final_data_thirty_minute.reset_index(inplace=True,drop=True)
final_data_one_hour.reset_index(inplace=True,drop=True)

print(final_data_one_minute.head())
print(final_data_one_minute.tail())
print(final_data_five_minute.head())
print(final_data_five_minute.tail())
print(final_data_fifteen_minute.head())
print(final_data_fifteen_minute.tail())
print(final_data_thirty_minute.head())
print(final_data_thirty_minute.tail())
print(final_data_one_hour.head())
print(final_data_one_hour.tail())

final_data_one_minute = final_data_one_minute.astype(str)
final_data_five_minute = final_data_five_minute.astype(str)
final_data_fifteen_minute = final_data_fifteen_minute.astype(str)
final_data_thirty_minute = final_data_thirty_minute.astype(str)
final_data_one_hour = final_data_one_hour.astype(str)

upload_data_gbq(final_data_one_minute,"Titania.options_one_minute_data","append")
upload_data_gbq(final_data_five_minute,"Titania.options_five_minute_data","append")
upload_data_gbq(final_data_fifteen_minute,"Titania.options_fifteen_minute_data","append")
upload_data_gbq(final_data_thirty_minute,"Titania.options_thirty_minute_data","append")
upload_data_gbq(final_data_one_hour,"Titania.options_one_hour_data","append")

print("Data Uploaded Successfully")

# end_time = datetime.now(timezone("Asia/Kolkata"))
end_time = datetime.now()

print(end_time)

print('Duration: {}'.format(end_time - start_time))


