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


from pandas.io import sql
from pandasql import sqldf

# import mysql.connector as mysql
# import pymysql

from pymongo import MongoClient
from pymongo.server_api import ServerApi

import pyotp

# import MySQLdb
# import pymysql


from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Numeric,
    Integer,
    VARCHAR,
    update,
    delete,
)


from sqlalchemy import create_engine

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


start_time = datetime.now() 
# start_time = datetime.now(timezone("Asia/Kolkata"))
print("Script execution started")
print(start_time)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

today_now = datetime.now() 

expiry_date = today_now

if today_now.strftime("%w") == "1":
    expiry_date = today_now + timedelta(days=4)
elif today_now.strftime("%w") == "2":
    expiry_date = today_now + timedelta(days=10)
elif today_now.strftime("%w") == "3":
    expiry_date = today_now + timedelta(days=9)
elif today_now.strftime("%w") == "4":
    expiry_date = today_now + timedelta(days=8)
elif today_now.strftime("%w") == "5":
    expiry_date = today_now + timedelta(days=7)
elif today_now.strftime("%w") == "6":
    expiry_date = today_now + timedelta(days=6)
elif today_now.strftime("%w") == "7":
    expiry_date = today_now + timedelta(days=5)

print("Expiry date")
print(expiry_date)
# expiry_date = '25-01-2023 15:00:00'
# expiry_date = datetime.strptime(expiry_date, '%d-%m-%Y %H:%M:%S')

expiry_date_char = expiry_date.strftime("%Y-%m-%d")
expiry_date_month = expiry_date.strftime("%d%b%y").upper()


def sweths_violation(stock, data):
    print("running : sweths_violation") 
    now = datetime.now() 
    #     print(now)
    current_time = now.strftime("%H:%M:%S")
    global increment

    final_data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

    final_data.reset_index(level=0, inplace=True, drop=True)

    if current_time >= "09:55:00":

        trigger_price = 0
        stage = ""
        # print(final_data)
        if (final_data.loc[0, "Close"] > final_data.loc[0, "Open"]) and abs(
            final_data.loc[0, "Close"] - final_data.loc[0, "Open"]
        ) >= 0.7 * abs(final_data.loc[0, "High"] - final_data.loc[0, "Low"]):
            trigger_price = final_data.loc[0, "Low"]
            stage = "Green"
        elif (final_data.loc[0, "Close"] < final_data.loc[0, "Open"]) and abs(
            final_data.loc[0, "Close"] - final_data.loc[0, "Open"]
        ) >= 0.7 * abs(final_data.loc[0, "High"] - final_data.loc[0, "Low"]):
            trigger_price = final_data.loc[0, "High"]
            stage = "Red"
        else:
            next
        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        # print(satisfied_df)
        for j in range(4, len(final_data)):
            if stage == "Green":
                if final_data.loc[j, "Close"] < trigger_price:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Sell", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )
                    call = "Sell"
            elif stage == "Red":
                if final_data.loc[j, "Close"] > trigger_price:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Buy", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )
                    call = "Buy"
            else:
                next

        # print(satisfied_df)
        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)
            #             satisfied_df['Datetime'] = pd.to_datetime(satisfied_df['Datetime'], infer_datetime_format=True, utc=True )
            #             satisfied_df['Datetime'] = satisfied_df['Datetime'].dt.tz_convert('Asia/Kolkata')

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 
            # time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
            # time_delta_mins = time_delta.total_seconds()/60

            Signal_df.loc[increment, "Strategy"] = "Sweths Violation"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1

    else:
        print("Strategy not live due to time")


def cowboy(stock, data):
    print("running : cowboy")
    now = datetime.now() 

#     final_levels_df = pd.read_csv(
#         "/home/ubuntu/Python_Automation/data/cowboy_data.csv", index_col=False
#     )
    final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_cowboy_data.csv",index_col=False)

    global increment

    #     print(final_levels_df)
    # for idx in range(0,len(nse_data)):

    satisfied_df = pd.DataFrame(
        columns=[
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "date",
            "Call",
        ]
    )

    data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

    data.reset_index(level=0, inplace=True, drop=True)

    final_data = data

    #     print(final_data)

    sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
    sub_df.reset_index(inplace=True, drop=True)

    if len(sub_df) > 0:
        #         print(final_data)
        # print(sub_df)
        # print(sub_df.loc[0,"Rider_Bullish"])
        if sub_df.loc[0, "Rider_Bullish"] == "Yes":
            satisfied_df = pd.DataFrame(
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                    "date",
                    "Call",
                ]
            )

            for j in range(0, len(final_data)):
                if final_data.loc[j, "Close"] > sub_df.loc[0, "Bullish_Level"]:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Buy", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )
                else:
                    next
        elif sub_df.loc[0, "Rider_Bearish"] == "Yes":

            for j in range(0, len(final_data)):
                if final_data.loc[j, "Close"] < sub_df.loc[0, "Bearish_Level"]:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Sell", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )
                else:
                    next

    if not satisfied_df.empty:
        satisfied_df = satisfied_df.head(1)

        #         satisfied_df['Datetime'] = pd.to_datetime(satisfied_df['Datetime'], infer_datetime_format=True, utc=True )
        #         satisfied_df['Datetime'] = satisfied_df['Datetime'].dt.tz_convert('Asia/Kolkata')

        satisfied_df.reset_index(inplace=True, drop=True)

        ind_time = datetime.now() 
        #         time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
        #         time_delta_mins = time_delta.total_seconds()/60

        #         pcr_call = technical_data.loc[0,"PCR_Call"]

        Signal_df.loc[increment, "Strategy"] = "Cowboy"
        Signal_df.loc[increment, "Stock"] = stock
        Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
        Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
        Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

        increment = increment + 1


def reds_rocket(stock, data):
    print("running : reds_rocket")
    now = datetime.now() 

#     final_levels_df = pd.read_csv(
#         "/home/ubuntu/Python_Automation/data/reds_rocket.csv", index_col=False
#     )
    final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_reds_rocket.csv",index_col=False)
    

    #     print(final_levels_df)

    global increment

    # for idx in range(0,len(nse_data)):
    #     stock = nse_data.loc[idx,"Yahoo_Symbol"]

    #     print(final_levels_df)
    sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
    sub_df.reset_index(inplace=True, drop=True)

    satisfied_df = pd.DataFrame(
        columns=[
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "date",
            "Call",
        ]
    )

    if len(sub_df) > 0:
        # Get the data

        data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

        data.reset_index(level=0, inplace=True, drop=True)

        final_data = data

        print(final_data)

        for j in range(0, len(final_data)):
            if final_data.loc[j, "Close"] > sub_df.loc[0, "Reds_High"]:
                temp_call_data = final_data.loc[
                    j,
                ]
                temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
                satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)
            elif final_data.loc[j, "Close"] < sub_df.loc[0, "Reds_Low"]:
                temp_call_data = final_data.loc[
                    j,
                ]
                temp_call_data = temp_call_data.append(
                    pd.Series("Sell", index=["Call"])
                )
                satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

        # print("reds_rocekt data")
        # print(satisfied_df)
        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)
            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            # print(time_delta_mins)

            Signal_df.loc[increment, "Strategy"] = "Reds Rocket"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1

    else:
        print("Reds Rocket Criteria not met previous day")


def reds_brahmos(stock, data):
    print("running : reds_brahmos")

    now = datetime.now() 

#     final_levels_df = pd.read_csv(
#         "/home/ubuntu/Python_Automation/data/reds_brahmos.csv", index_col=False
#     )
    final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_reds_brahmos.csv",index_col=False)

    global increment

    # for idx in range(0,len(nse_data)):
    #     stock = nse_data.loc[idx,"Yahoo_Symbol"]

    sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
    sub_df.reset_index(inplace=True, drop=True)

    #     print(sub_df)

    if len(sub_df) > 0:
        # Get the data

        data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

        data.reset_index(level=0, inplace=True, drop=True)

        final_data = data

        #         print(final_data)

        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        for j in range(0, len(final_data)):
            if final_data.loc[j, "Close"] > sub_df.loc[0, "Reds_High"]:
                temp_call_data = final_data.loc[
                    j,
                ]
                temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
                satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)
            elif final_data.loc[j, "Close"] < sub_df.loc[0, "Reds_Low"]:
                temp_call_data = final_data.loc[
                    j,
                ]
                temp_call_data = temp_call_data.append(
                    pd.Series("Sell", index=["Call"])
                )
                satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            Signal_df.loc[increment, "Strategy"] = "Reds Brahmos"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1
    else:
        print("Reds Brahmos Criteria not met previous day")


def blackout(stock, data):
    print("running : blackout")
    now = datetime.now() 

#     final_levels_df = pd.read_csv(
#         "/home/ubuntu/Python_Automation/data/blackout.csv", index_col=False
#     )
    final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_blackout.csv",index_col=False)

    global increment

    # for idx in range(0,len(nse_data)):
    #     stock = nse_data.loc[idx,"Yahoo_Symbol"]

    sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
    sub_df.reset_index(inplace=True, drop=True)

    if len(sub_df) > 0:
        # Get the data

        data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

        data.reset_index(level=0, inplace=True, drop=True)

        #         print(data)

        # Convert the date to datetime64
        data["date"] = pd.to_datetime(data["Datetime"], format="%Y-%m-%d")

        final_data = data.loc[(data["date"] >= now.strftime("%Y-%m-%d"))]

        final_data.reset_index(inplace=True, drop=True)

        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        if sub_df.loc[0, "stage"] == "Short":
            for j in range(0, len(final_data)):
                if final_data.loc[j, "Close"] < sub_df.loc[0, "target"]:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Sell", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )
        else:
            for j in range(0, len(final_data)):
                if final_data.loc[j, "Close"] > sub_df.loc[0, "target"]:
                    temp_call_data = final_data.loc[
                        j,
                    ]
                    temp_call_data = temp_call_data.append(
                        pd.Series("Buy", index=["Call"])
                    )
                    satisfied_df = satisfied_df.append(
                        temp_call_data, ignore_index=True
                    )

        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            Signal_df.loc[increment, "Strategy"] = "Blackout"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1

    else:
        print("Blackout Criteria not met previous day")


def gap_up(stock, data):
    print("running : gap_up")
    now = datetime.now() 

    current_time = now.strftime("%H:%M:%S")
    global increment

    if current_time >= "09:35:00":
#         final_levels_df = pd.read_csv(
#             "/home/ubuntu/Python_Automation/data/gaps_strategy.csv", index_col=False
#         )
        final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_gaps_strategy.csv",index_col=False)

        sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
        sub_df.reset_index(inplace=True, drop=True)
        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        #         print(sub_df)

        if len(sub_df) > 0:

            high_price = sub_df.loc[0, "Previous_High"]
            close_price = sub_df.loc[0, "Previous_Close"]

            #             cut_offtime = now
            #             # print(now)
            #             if(int(now.strftime('%M')) % 5 > 0):
            #                 cut_offtime = now + timedelta(minutes=-(int(now.strftime('%M'))%5))
            #                 cut_offtime = cut_offtime.strftime('%Y-%m-%d %H:%M:00')
            #                 # print(cut_offtime)
            #             else:
            #                 cut_offtime = now
            #                 cut_offtime = cut_offtime.strftime('%Y-%m-%d %H:%M:00')
            #                 # print(cut_offtime)

            # Convert the date to datetime64
            data["date"] = pd.to_datetime(data["Datetime"], format="%Y-%m-%d")

            #             final_data = data.loc[(data['date'] >= now.strftime("%Y-%m-%d")) & (data['Datetime'] <= cut_offtime)]
            final_data = data.loc[(data["date"] >= now.strftime("%Y-%m-%d"))]

            final_data.reset_index(inplace=True, drop=True)

            #             print(final_data)

            satisfied_df = pd.DataFrame(
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                    "date",
                    "Call",
                ]
            )

            open_price = final_data.loc[0, "Open"]

            if open_price > close_price:
                for j in range(4, len(final_data)):
                    current_date = final_data.loc[j, "Datetime"]

                    day_high = max(
                        np.nanmax(final_data[0:j]["Close"]),
                        np.nanmax(final_data[0:j]["Open"]),
                    )

                    day_low = min(
                        np.nanmin(final_data[0:j]["Close"]),
                        np.nanmin(final_data[0:j]["Open"]),
                    )

                    low_range = min(
                        final_data.loc[j - 1, "Low"],
                        final_data.loc[j - 2, "Low"],
                        final_data.loc[j - 3, "Low"],
                        final_data.loc[j - 4, "Low"],
                    )
                    high_range = max(
                        final_data.loc[j - 1, "High"],
                        final_data.loc[j - 2, "High"],
                        final_data.loc[j - 3, "High"],
                        final_data.loc[j - 4, "High"],
                    )

                    current_close = final_data.loc[j, "Close"]

                    if (
                        (abs(high_range - low_range) / low_range * 100 < 0.4)
                        and (current_close >= high_price)
                        and (current_close >= day_high)
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Buy", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

                    elif (
                        (abs(high_range - low_range) / low_range * 100 < 0.4)
                        and (current_close <= close_price)
                        and (current_close <= day_low)
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Sell", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 
            #             time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
            #             time_delta_mins = time_delta.total_seconds()/60

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            #             if(time_delta_mins >= 4):

            Signal_df.loc[increment, "Strategy"] = "Gap_up"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1


def gap_down(stock, data):
    print("running : gap_down")
    now = datetime.now() 
    #     now = datetime.now() + timedelta(hours=5,minutes=30)
    current_time = now.strftime("%H:%M:%S")
    global increment

    if current_time >= "09:35:00":
#         final_levels_df = pd.read_csv(
#             "/home/ubuntu/Python_Automation/data/gaps_strategy.csv", index_col=False
#         )
        final_levels_df = pd.read_csv("~/Downloads/Reddy_Stocks_Application/data/us_gaps_strategy.csv",index_col=False)
        sub_df = final_levels_df.loc[(final_levels_df["Stock"] == stock)]
        sub_df.reset_index(inplace=True, drop=True)

        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        if len(sub_df) > 0:
            #                 stock = sub_df.loc[0,"Stock"]
            high_price = sub_df.loc[0, "Previous_High"]
            close_price = sub_df.loc[0, "Previous_Close"]
            prev_low_price = sub_df.loc[0, "Previous_Low"]

            data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

            #             cut_offtime = now
            #             # print(now)
            #             if(int(now.strftime('%M')) % 5 > 0):
            #                 cut_offtime = now + timedelta(minutes=-(int(now.strftime('%M'))%5))
            #                 cut_offtime = cut_offtime.strftime('%Y-%m-%d %H:%M:00')
            #                 # print(cut_offtime)
            #             else:
            #                 cut_offtime = now
            #                 cut_offtime = cut_offtime.strftime('%Y-%m-%d %H:%M:00')

            # Convert the date to datetime64
            data["date"] = pd.to_datetime(data["Datetime"], format="%Y-%m-%d")

            final_data = data.loc[(data["date"] >= now.strftime("%Y-%m-%d"))]

            final_data.reset_index(inplace=True, drop=True)

            satisfied_df = pd.DataFrame(
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                    "date",
                    "Call",
                ]
            )

            open_price = final_data.loc[0, "Open"]

            if open_price < close_price:
                for j in range(4, len(final_data)):
                    current_date = final_data.loc[j, "Datetime"]

                    day_high = max(
                        np.nanmax(final_data[0:j]["Close"]),
                        np.nanmax(final_data[0:j]["Open"]),
                    )

                    day_low = min(
                        np.nanmin(final_data[0:j]["Close"]),
                        np.nanmin(final_data[0:j]["Open"]),
                    )

                    low_range = min(
                        final_data.loc[j - 1, "Low"],
                        final_data.loc[j - 2, "Low"],
                        final_data.loc[j - 3, "Low"],
                        final_data.loc[j - 4, "Low"],
                    )
                    high_range = max(
                        final_data.loc[j - 1, "High"],
                        final_data.loc[j - 2, "High"],
                        final_data.loc[j - 3, "High"],
                        final_data.loc[j - 4, "High"],
                    )

                    current_close = final_data.loc[j, "Close"]
                    current_high = final_data.loc[j, "High"]
                    current_low = final_data.loc[j, "Low"]

                    if (
                        (abs(high_range - low_range) / low_range * 100 < 0.4)
                        and (current_close >= high_price)
                        and (current_close >= day_high)
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Buy", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

                    elif (
                        (abs(high_range - low_range) / low_range * 100 < 0.4)
                        and (current_close <= prev_low_price)
                        and (current_low <= day_low)
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Sell", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

        if not satisfied_df.empty:
            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = datetime.now() 
            #             time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
            #             time_delta_mins = time_delta.total_seconds()/60

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            #             if(time_delta_mins >= 4):

            Signal_df.loc[increment, "Strategy"] = "Gap_down"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1


def abc_5_cand(stock, data):
    print("running : abc_5_cand")
    now = datetime.now() 
    #     now = datetime.now() + timedelta(hours=5,minutes=30)
    current_time = now.strftime("%H:%M:%S")
    global increment

    #     if(current_time >= "09:40:00"):

    data["Datetime"] = pd.to_datetime(data["Datetime"])

    data = data[data["Datetime"] >= now.strftime("%Y-%m-%d")]

    data.reset_index(level=0, inplace=True, drop=True)

    # Convert the date to datetime64
    data["date"] = pd.to_datetime(data["Datetime"], format="%Y-%m-%d")

    final_data = data.loc[(data["date"] >= now.strftime("%Y-%m-%d"))]

    final_data.reset_index(inplace=True, drop=True)

    satisfied_df = pd.DataFrame(
        columns=[
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "date",
            "Call",
        ]
    )

    # Starting with the third candle
    for j in range(5, len(final_data)):
        # Check if the candle is a green candle
        if final_data.loc[j, "Close"] > final_data.loc[j, "Open"]:
            # Check if the prior candles are in the reversal trend
            if (final_data.loc[j - 1, "Low"] < final_data.loc[j - 2, "Low"]) and (
                final_data.loc[j - 2, "Low"] < final_data.loc[j - 3, "Low"]
            ):

                # Get the breakout max in the reversal i.e., B Point
                reversal_high = max(
                    final_data.loc[j - 1, "High"],
                    final_data.loc[j - 2, "High"],
                    final_data.loc[j - 3, "High"],
                )

                # Get the breakout min in the reversal i.e., C point
                reversal_low = min(
                    final_data.loc[j - 1, "Low"],
                    final_data.loc[j - 2, "Low"],
                    final_data.loc[j - 3, "Low"],
                )

                # Check if the before reversal is a uptrend
                if (
                    final_data.loc[j - 3, "High"] > final_data.loc[j - 4, "High"]
                    and final_data.loc[j - 4, "High"] > final_data.loc[j - 5, "High"]
                ):
                    # Get the starting point of the trend i.e., A point
                    trend_low = min(
                        final_data.loc[j - 4, "Low"], final_data.loc[j - 5, "Low"]
                    )

                    # Check if the ABC pattern is completely followed
                    if (
                        final_data.loc[j, "Close"] > reversal_high
                        and reversal_low > trend_low
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Buy", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

        else:
            # Check if the prior candles are in the reversal trend
            if (
                final_data.loc[j - 1, "High"] > final_data.loc[j - 2, "High"]
                and final_data.loc[j - 2, "High"] > final_data.loc[j - 3, "High"]
            ):

                # Get the breakout max in the reversal i.e., B Point
                reversal_high = min(
                    final_data.loc[j - 1, "Low"],
                    final_data.loc[j - 2, "Low"],
                    final_data.loc[j - 3, "Low"],
                )

                # Get the breakout min in the reversal i.e., C point
                reversal_low = max(
                    final_data.loc[j - 1, "High"],
                    final_data.loc[j - 2, "High"],
                    final_data.loc[j - 3, "High"],
                )

                # Check if the before reversal is a uptrend
                if (
                    final_data.loc[j - 3, "Low"] < final_data.loc[j - 4, "Low"]
                    and final_data.loc[j - 4, "Low"] < final_data.loc[j - 5, "Low"]
                ):
                    # Get the starting point of the trend i.e., A point
                    trend_low = max(
                        final_data.loc[j - 4, "High"], final_data.loc[j - 5, "High"]
                    )

                    # Check if the ABC pattern is completely followed
                    if (
                        final_data.loc[j, "Close"] < reversal_high
                        and reversal_low < trend_low
                    ):
                        temp_call_data = final_data.loc[
                            j,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Sell", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

    if not satisfied_df.empty:
        satisfied_df = satisfied_df.head(1)

        satisfied_df.reset_index(inplace=True, drop=True)

        ind_time = datetime.now() 
        #         time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
        #         time_delta_mins = time_delta.total_seconds()/60

        #         pcr_call = technical_data.loc[0,"PCR_Call"]

        Signal_df.loc[increment, "Strategy"] = "5_Cand_ABC"
        Signal_df.loc[increment, "Stock"] = stock
        Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
        Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
        Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

        increment = increment + 1


def abc_3_cand(stock, data):
    print("running : abc_3_cand")
    now = datetime.now() 
    current_time = now.strftime("%H:%M:%S")
    global increment

    if current_time >= "09:40:00":
        # for idx in range(0,len(nse_data)):
        #     stock = nse_data.loc[idx,"Yahoo_Symbol"]

        # Convert the date to datetime64
        data["date"] = pd.to_datetime(data["Datetime"], format="%Y-%m-%d")

        final_data = data.loc[(data["date"] >= now.strftime("%Y-%m-%d"))]

        final_data.reset_index(inplace=True, drop=True)

        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        # Starting with the first candle
        for i in range(2, len(final_data)):

            if (
                (final_data.loc[i, "Close"] > final_data.loc[i, "Open"])
                and (final_data.loc[i - 1, "Close"] < final_data.loc[i - 1, "Open"])
                and (final_data.loc[i - 2, "Close"] > final_data.loc[i - 2, "Open"])
            ):

                if (
                    (final_data.loc[i - 1, "Low"] > final_data.loc[i - 2, "Low"])
                    and (final_data.loc[i, "Close"] > final_data.loc[i - 2, "High"])
                    and (final_data.loc[i - 1, "High"] < final_data.loc[i - 2, "High"])
                ):

                    first_range = (
                        final_data.loc[i - 2, "High"] - final_data.loc[i - 2, "Low"]
                    )
                    second_range = (
                        final_data.loc[i - 1, "High"] - final_data.loc[i - 1, "Low"]
                    )
                    if first_range / second_range >= 2:
                        temp_call_data = final_data.loc[
                            i,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Buy", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

            elif (
                (final_data.loc[i, "Close"] < final_data.loc[i, "Open"])
                and (final_data.loc[i - 1, "Close"] > final_data.loc[i - 1, "Open"])
                and (final_data.loc[i - 2, "Close"] < final_data.loc[i - 2, "Open"])
            ):

                if (
                    (final_data.loc[i - 1, "Low"] > final_data.loc[i - 2, "Low"])
                    and (final_data.loc[i, "Close"] < final_data.loc[i - 2, "Low"])
                    and (final_data.loc[i - 1, "Low"] > final_data.loc[i - 2, "Low"])
                ):

                    first_range = (
                        final_data.loc[i - 2, "High"] - final_data.loc[i - 2, "Low"]
                    )
                    second_range = (
                        final_data.loc[i - 1, "High"] - final_data.loc[i - 1, "Low"]
                    )
                    if first_range / second_range >= 2:
                        temp_call_data = final_data.loc[
                            i,
                        ]
                        temp_call_data = temp_call_data.append(
                            pd.Series("Sell", index=["Call"])
                        )
                        satisfied_df = satisfied_df.append(
                            temp_call_data, ignore_index=True
                        )

        if not satisfied_df.empty:

            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            ind_time = (datetime.now()).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ind_time = datetime.strptime(ind_time, '%Y-%m-%d %H:%M:%S')

            # time_delta = ind_time - satisfied_df.loc[0,"Datetime"]
            # time_delta_mins = time_delta.total_seconds()/60

            #             pcr_call = technical_data.loc[0,"PCR_Call"]

            Signal_df.loc[increment, "Strategy"] = "3_Cand_ABC"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1


def volume_breakout(stock, data):
    print("running : volume_breakout")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    global increment

    #     obj = SmartConnect(api_key="NsXKahCV")

    #     time.sleep(1)

    #     data = obj.generateSession("S970011","Welcome@123")

    if current_time >= "09:35:00":
        # for idx in range(0,len(nse_data)):
        #     stock = nse_data.loc[idx,"Yahoo_Symbol"]

        # print(stock)

        hist_df = data

        hist_df = hist_df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

        hist_df = hist_df[hist_df["Datetime"] >= now.strftime("%Y-%m-%d")]

        hist_df.reset_index(inplace=True, drop=True)

        #         print(hist_df)

        hist_df["Volume_Rank"] = hist_df["Volume"].rank(ascending=False)

        hist_df["price_change"] = abs(hist_df["Low"] - hist_df["High"])

        for idx in range(0, len(hist_df)):
            if hist_df.loc[idx, "Close"] > hist_df.loc[idx, "Open"]:
                hist_df.loc[idx, "perc_change"] = (
                    hist_df.loc[idx, "price_change"] * 1.00 / hist_df.loc[idx, "Low"]
                )
            else:
                hist_df.loc[idx, "perc_change"] = (
                    hist_df.loc[idx, "price_change"] * 1.00 / hist_df.loc[idx, "High"]
                )

        satisfied_df = pd.DataFrame(
            columns=[
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
                "date",
                "Call",
            ]
        )

        hist_df = hist_df.sort_values(by=["Volume_Rank"], ascending=True)
        hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"])

        first_volume_data = hist_df.loc[hist_df["Volume_Rank"] == 1]
        first_volume_data.reset_index(inplace=True, drop=True)

        breakout_high_value = first_volume_data.loc[0, "High"]
        breakout_low_value = first_volume_data.loc[0, "Low"]
        breakout_time = first_volume_data.loc[0, "Datetime"]

        temp_final_data = hist_df.loc[hist_df["Datetime"] > breakout_time]
        temp_final_data.reset_index(inplace=True, drop=True)

        for idx in range(0, len(temp_final_data)):
            if (
                temp_final_data.loc[idx, "Close"] > breakout_high_value
                and abs(temp_final_data.loc[idx, "Close"] - breakout_high_value)
                / breakout_high_value
                * 100
                <= 0.4
            ):
                temp_final_data.loc[idx, "Signal"] = "Buy"
            elif (
                temp_final_data.loc[idx, "Close"] < breakout_low_value
                and abs(temp_final_data.loc[idx, "Close"] - breakout_low_value)
                / breakout_low_value
                * 100
                <= 0.4
            ):
                temp_final_data.loc[idx, "Signal"] = "Sell"
            else:
                temp_final_data.loc[idx, "Signal"] = ""

        temp_final_data = temp_final_data.loc[
            temp_final_data["Volume_Rank"] <= 15,
        ]

        temp_final_data = temp_final_data.sort_values(by=["Datetime"], ascending=True)

        temp_final_data["match"] = temp_final_data.Signal.eq(
            temp_final_data.Signal.shift()
        )

        final_temp_data = temp_final_data.loc[
            (temp_final_data["match"] == False)
            & (
                (temp_final_data["Signal"] == "Buy")
                | (temp_final_data["Signal"] == "Sell")
            )
        ]

        final_temp_data.reset_index(inplace=True, drop=True)

        # print(final_temp_data)

        if not final_temp_data.empty:
            for j in range(0, len(final_temp_data)):
                temp_call_data = final_temp_data.loc[
                    j,
                ]
                temp_call_data = temp_call_data.append(
                    pd.Series(final_temp_data.loc[j, "Signal"], index=["Call"])
                )
                satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

        if not satisfied_df.empty:

            satisfied_df = satisfied_df.head(1)

            satisfied_df.reset_index(inplace=True, drop=True)

            # print(pcr_call)

            Signal_df.loc[increment, "Strategy"] = "Volume_Breakout"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Call"]
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "Datetime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]

            increment = increment + 1


def options_chain_volume_breakout(stock):
    print("running : options_chain_volume_breakout")
    now = datetime.now() 
    current_time = now.strftime("%H:%M:%S")

    #     print(current_time)

    global increment

    current_sym = ""

    try:

        if current_time >= "09:35:00":
            if stock == "%5ENSEI":
                current_sym = "Nifty"
            else:
                current_sym = "BankNifty"
            try:
                fut_path = (
                    "/home/ubuntu/Options_Chain/"
                    + current_sym
                    + "/"
                    + now.strftime("%Y-%m-%d")
                    + "_Futures_Options_Signals.csv"
                )
                opt_path = (
                    "/home/ubuntu/Options_Chain/"
                    + current_sym
                    + "/"
                    + now.strftime("%Y-%m-%d")
                    + "_Options_Signals.csv"
                )

                #             print(fut_path)

                futures_data = pd.read_csv(fut_path)

                opt_data = pd.read_csv(opt_path)

            except Exception as e:
                #                 /home/ubuntu/Options_Chain/
                # /Users/apple/Desktop/Python_Stocks_Automation/Options_data/
                fut_path = (
                    "/home/ubuntu/Options_Chain/"
                    + current_sym
                    + "/"
                    + (now - timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                    + "_Futures_Options_Signals.csv"
                )
                opt_path = (
                    "/home/ubuntu/Options_Chain/"
                    + current_sym
                    + "/"
                    + (now - timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
                    + "_Options_Signals.csv"
                )

                futures_data = pd.read_csv(fut_path)

                opt_data = pd.read_csv(opt_path)

            final_data = pd.merge(
                opt_data,
                futures_data,
                how="inner",
                left_on="Datetime",
                right_on="Datetime",
            )

            # final_data = final_data[[u'Datetime',u'Call_Interpretation_x']]

            final_data = final_data[
                [
                    "Datetime",
                    "Call_Interpretation",
                    "Put_Interpretation",
                    "pcr_ratio_x",
                    "current_call_volume",
                    "current_put_volume",
                    "Call_Majority_x",
                    "Put_Majority_x",
                    "call_volume_rank_x",
                    "put_volume_rank_x",
                    "signal",
                    "Strike_Price",
                    "future_volume",
                    "call_traded_volume",
                    "call_pchange",
                    "call_changeinopeninterest",
                    "put_traded_volume",
                    "put_pchange",
                    "put_changeinopeninterest",
                    "pcr_ratio_y",
                    "fut_volume_rank",
                    "call_volume_rank_y",
                    "put_volume_rank_y",
                    "call_value",
                    "put_value",
                ]
            ]

            # print(final_data.columns)
            final_data.columns = [
                "Datetime",
                "Call_Interpretation",
                "Put_Interpretation",
                "overall_pcr",
                "current_call_volume",
                "current_put_volume",
                "overall_Call_Majority",
                "overall_Put_Majority",
                "overall_call_volume_rank",
                "overall_put_volume_rank",
                "overall_signal",
                "ATM_Strike_Price",
                "future_volume",
                "call_fut_traded_volume",
                "ATM_call_pchange",
                "ATM_call_changeinopeninterest",
                "ATM_put_traded_volume",
                "ATM_put_pchange",
                "ATM_put_changeinopeninterest",
                "ATM_pcr_ratio",
                "fut_volume_rank",
                "ATM_call_volume_rank",
                "ATM_put_volume_rank",
                "ATM_call_value",
                "put_value",
            ]

            final_data["sum"] = (
                final_data["fut_volume_rank"]
                + final_data["overall_call_volume_rank"]
                + final_data["overall_put_volume_rank"]
            )

            selected_data = final_data.sort_values("sum").head(1)

            selected_data.reset_index(inplace=True, drop=True)

            # stock = nse_data.loc[stck,"Yahoo_Symbol"]

            hist_df = yf.download(tickers=stock, period="2d", interval="1m")
            hist_df = pd.DataFrame(hist_df)

            hist_df.reset_index(level=0, inplace=True)

            hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"])

            hist_df = hist_df[hist_df["Datetime"] >= now.strftime("%Y-%m-%d")]

            hist_df.reset_index(inplace=True, drop=True)

            hist_df = hist_df[
                ["Datetime", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            ]

            hist_df.columns = [
                "Datetime",
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            ]

            # print(hist_df.head(5))

            temp_data = hist_df[
                pd.to_datetime(hist_df["Datetime"]) >= selected_data.iloc[0, 0]
            ]

            temp_data.reset_index(drop=True, inplace=True)

            range_low = temp_data.loc[0, "Low"]
            range_high = temp_data.loc[0, "High"]

            print("Critical range for the day")

            print(range_low)
            print(range_high)
            print(temp_data.loc[0, "Datetime"])

            temp_data["position"] = ""

            for idx in range(1, len(temp_data)):
                if (
                    temp_data.loc[idx, "Low"] <= range_low
                    and temp_data.loc[idx, "Low"] <= range_high
                    and temp_data.loc[idx, "High"] <= range_high
                    and temp_data.loc[idx, "High"] <= range_low
                ):

                    temp_data.loc[idx, "position"] = "Sell"
                elif (
                    temp_data.loc[idx, "Low"] >= range_low
                    and temp_data.loc[idx, "Low"] >= range_high
                    and temp_data.loc[idx, "High"] >= range_high
                    and temp_data.loc[idx, "High"] >= range_low
                ):
                    temp_data.loc[idx, "position"] = "Buy"

            signals_df = temp_data[temp_data["position"] != ""]

            signals_df.reset_index(drop=True, inplace=True)

            signals_df["next_1_position"] = signals_df["position"].shift(-1)
            signals_df["next_2_position"] = signals_df["next_1_position"].shift(-1)
            signals_df["prev_1_position"] = signals_df["position"].shift(1)

            final_signal_df = signals_df[
                (signals_df["prev_1_position"] != signals_df["position"])
                & (signals_df["position"] == signals_df["next_1_position"])
            ]

            temp_Signal_df = final_signal_df[["position", "Datetime", "Close"]]
            temp_Signal_df.reset_index(inplace=True, drop=True)
            # print(temp_Signal_df)

            temp_Signal_df["Strategy"] = "Options_Chain_Volume"
            temp_Signal_df["Stock"] = stock
            temp_Signal_df = temp_Signal_df[
                ["Strategy", "Stock", "position", "Datetime", "Close"]
            ]

            temp_Signal_df.columns = [
                "Strategy",
                "Stock",
                "Signal",
                "Datetime",
                "Value",
            ]

            if not temp_Signal_df.empty:
                satisfied_df = temp_Signal_df.tail(1)

                satisfied_df.reset_index(inplace=True, drop=True)
                # print(satisfied_df)

                Signal_df.loc[increment, "Strategy"] = satisfied_df.loc[0, "Strategy"]
                Signal_df.loc[increment, "Stock"] = satisfied_df.loc[0, "Stock"]
                Signal_df.loc[increment, "Signal"] = satisfied_df.loc[0, "Signal"]
                Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[
                    0, "Datetime"
                ].strftime("%Y-%m-%d %H:%M:%S")
                Signal_df.loc[increment, "Value"] = round(
                    satisfied_df.loc[0, "Value"], 2
                )

                increment = increment + 1

    except Exception as e:
        print("options_chain_volume_breakout failed: {}".format(e))



us_stocks_data = pd.read_csv("/Users/apple/Downloads/Reddy_Stocks_Application/data/US - 30 Stocks.csv")

nse_data = pd.DataFrame(
    [["BANKNIFTY", "%5ENSEBANK", "BANKNIFTY-EQ"], ["Nifty50", "%5ENSEI", "Nifty50-EQ"]],
    columns=["Symbol", "Yahoo_Symbol", "TradingSymbol"],
)

strategies = [
    "sweths_violation",
    "cowboy",
    "reds_rocket",
    "reds_brahmos",
    "blackout",
    "gap_up",
    "gap_down",
    "volume_breakout",
    "abc_5_cand",
    "abc_3_cand",
]


bot_token = "1931575614:AAFhtU1xieFDqC9WAAzw15G4KdB8rdzrif4"
# chat_id = ["535162272","714628563", "1808943433","844935609","996359001","846794885","1623124565","1088161376","1612368682", "507042774","473977639","488310125","373868886","1594535460","960024014","1080210611","1710009819","1542490708","971366033","997884717","1677198821","910359081","323226633"]
chat_id = ["535162272"]

increment = 0
# nifty_support = 17531
# nifty_resistance = 17531
# bank_nifty_support = 40777
# bank_nifty_resistance = 41446


server_api = ServerApi("1")

client = MongoClient(
    "mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE",
    server_api=server_api,
)

# db = client.titania_trading

db = client["United_States_Titania_Trading"]


def closest_value(input_list, input_value):
    difference = lambda input_list: abs(input_list - input_value)
    res = min(input_list, key=difference)
    print("Nearest support/resistance : ", str(res))
    price_pct_diff = abs(input_value - res) / res

    if price_pct_diff >= 0.001:
        return True
    else:
        return False


Signal_df = pd.DataFrame(columns=["Strategy", "Stock", "Signal", "Datetime", "Value"])

for stk in range(0,len(us_stocks_data)):
    print(us_stocks_data.loc[stk,"Symbol"])
    stock = us_stocks_data.loc[stk,"Symbol"]
    nifty_support_data = (
        db["support_and_resistance"]
        .find({"Stock": str(us_stocks_data.loc[stk,"Symbol"])})
        .sort([("Execution_date", -1)])
        .limit(1)
    )
    nifty_support_data = pd.DataFrame(list(nifty_support_data))
    
#     print(nifty_support_data)
    
    nifty_arima = nifty_support_data.loc[0, "arima_pivot_point"]
    nifty_arima_resistance_1 = nifty_support_data.loc[0, "arima_resistance_1"]
    nifty_arima_resistance_2 = nifty_support_data.loc[0, "arima_resistance_2"]
    nifty_arima_support_1 = nifty_support_data.loc[0, "arima_support_1"]
    nifty_arima_support_2 = nifty_support_data.loc[0, "arima_support_2"]

    nifty_levels = [
        nifty_arima_support_2,
        nifty_arima_support_1,
        nifty_arima,
        nifty_arima_resistance_1,
        nifty_arima_resistance_2,
    ]
    
    data = yf.download(tickers=str(us_stocks_data.loc[stk,"Symbol"]), period="1d", interval="5m")

    data = pd.DataFrame(data)

    data.reset_index(inplace=True)
    data = data.rename(columns = {'index':'Datetime'})
    
#     print(data)
    
    latest_data = data.tail(1)
    latest_data.reset_index(level=0, inplace=True, drop=True)

    current_close = latest_data.loc[0, "Close"]

    print(current_close)

    nearest_value = closest_value(nifty_levels, current_close)
    
    print(nearest_value)
    
    if nearest_value == True:
        
        for stra in range(0, len(strategies)):
            now = datetime.now() 
            current_time = now.strftime("%H:%M:%S")
            if strategies[stra] == "sweths_violation":
                sweths_violation(stock, data)
            elif strategies[stra] == "cowboy":
                cowboy(stock, data)
            elif strategies[stra] == "reds_rocket":
                reds_rocket(stock, data)
            elif strategies[stra] == "reds_brahmos":
                reds_brahmos(stock, data)
            elif strategies[stra] == "blackout":
                blackout(stock, data)
            elif strategies[stra] == "gap_up":
                gap_up(stock, data)
            elif strategies[stra] == "gap_down":
                gap_down(stock, data)
            elif strategies[stra] == "abc_5_cand":
                abc_5_cand(stock, data)
            elif strategies[stra] == "abc_3_cand":
                abc_3_cand(stock, data)
            elif strategies[stra] == "volume_breakout":
                volume_breakout(stock, data)
            elif strategies[stra] == "options_chain_volume_breakout":
                options_chain_volume_breakout(stock)
            else:
                pass
            
    else:
        print("At Support/Resistance Zone")


Signal_df = Signal_df.sort_values(by=["Datetime"], ascending=False)
Signal_df.reset_index(inplace=True, drop=True)


# print(Signal_df)

pysqldf = lambda q: sqldf(q, globals())

if not Signal_df.empty:

    Signal_df["Datetime"] = pd.to_datetime(Signal_df["Datetime"])
    Signal_df["Datetime"] = pd.to_datetime(Signal_df["Datetime"], errors="coerce")
    Signal_df["Datetime"] = Signal_df["Datetime"].dt.round("5min")
    Signal_df['Datetime'] = pd.to_datetime(Signal_df['Datetime'],format = '%Y-%m-%d %H:%M:%S').dt.tz_localize(None)

#     print(Signal_df)

    technical_collection = db["technical_indicator_5_minutes"]
    orders_raw_data = db["orders_raw_data"]

    technical_data = technical_collection.find({})
    technical_data = pd.DataFrame(list(technical_data))
    technical_data["Datetime"] = technical_data["Datetime"] - timedelta(hours=5)
    # print(technical_data)
    # print("Before the pysqldf")

    # print(Signal_df.columns)
    # print(technical_data.columns)

    technical_data = technical_data[
        [
            "Stock",
            "Datetime",
            "SMA_Call",
            "RSI_Call",
            "MACD_Call",
            "Pivot_Call",
            "PCR_Call",
            "BB_Call",
            "VWAP_Call",
            "SuperTrend_Call",
            "buy_probability",
            "sell_probability",
        ]
    ]
    
#     print(technical_data)

    Signal_df = pysqldf(
        """select t1.Strategy as Strategy,
                                    t1.Stock as Stock,
                                    t1.Signal as Signal,
                                    t1.Datetime as Datetime,
                                    t1.Value as Value,
                                    t2.SMA_Call as SMA_Call,
                                    t2.RSI_Call as RSI_Call,
                                    t2.MACD_Call as MACD_Call,
                                    t2.Pivot_Call as Pivot_Call,
                                    t2.PCR_Call as PCR_Call,
                                    t2.BB_Call as BB_Call,
                                    t2.VWAP_Call as VWAP_Call,
                                    t2.SuperTrend_Call as SuperTrend_Call,
                                    t2.buy_probability as buy_probability,
                                    t2.sell_probability as sell_probability

                            from Signal_df t1 
                            left join technical_data t2 on t1.Stock = t2.Stock and t1.Datetime = t2.Datetime"""
    )

    # print("After the pysqldf")

    Signal_df = Signal_df.sort_values(by=["Datetime"], ascending=True)
    Signal_df.reset_index(inplace=True, drop=True)

    stop_loss = 1
    target = 1

    Capital = 10000

    Signal_df["StopLoss"] = np.where(
        Signal_df["Signal"] == "Buy",
        Signal_df["Value"] - ((stop_loss * Signal_df["Value"]) / 100),
        (stop_loss * Signal_df["Value"]) / 100 + Signal_df["Value"],
    )
    Signal_df["Target"] = np.where(
        Signal_df["Signal"] == "Buy",
        Signal_df["Value"] + ((target * Signal_df["Value"]) / 100),
        Signal_df["Value"] - (target * Signal_df["Value"]) / 100,
    )

    Signal_df["Qty"] = abs(
        (20 / 100)
        * Capital
        / (Signal_df["Target"].astype(int) - Signal_df["StopLoss"].astype(int))
    ).round(decimals=0)

    Signal_df["expiry"] = expiry_date_char
    Signal_df["exec_rnk"] = Signal_df["Datetime"].rank(ascending=True)
    Signal_df["execution_date"] = (datetime.now()).strftime(
        "%Y-%m-%d"
    )

    # Delete the existing orders pushed for today
    x = orders_raw_data.delete_many(
        {"execution_date": (datetime.now()).strftime("%Y-%m-%d")}
    )

    print(x.deleted_count, " documents deleted.")

    orders_raw_data.insert_many(Signal_df.to_dict("records"))

    Signal_df["Spot_Price"] = 0

    Signal_df["Strike_Buy_Price"] = 0
    Signal_df["premium_StopLoss"] = 0
    Signal_df["premium_Target"] = 0
    Signal_df["lotsize"] = 0
    Signal_df["premium_Qty"] = 0
    Signal_df["historic_profit"] = 0
    Signal_df["current_script"] = ""
    Signal_df["token"] = 0

    # Signal_df['order_place'] = 0
    # Signal_df['order_id'] = 0
    # Signal_df['target_order_id'] = 0

    # Signal_df['stop_loss_order_id'] = 0
    # Signal_df['robo_order_id'] = 0
    # Signal_df['cancel_order_id'] = 0
    # Signal_df['final_order_id'] = 0
    Signal_df["conclusion"] = ""
    # Signal_df['target_hit']  = ""
    # Signal_df['avg_buy_price']  = ""
    # Signal_df['avg_sell_price']  = ""
    # Signal_df['avg_qty']  = ""
    # Signal_df['adjusted_target']  = ""
    # Signal_df['adjusted_stoploss']  = ""

    # Signal_df["execution_time"] = datetime.now(timezone("Asia/Kolkata"))
    Signal_df["execution_date"] = (datetime.now() ).strftime(
        "%Y-%m-%d"
    )

    print(Signal_df)

import robin_stocks

robin_stocks.robinhood.authentication.login(username="saitejareddy123@gmail.com",
         password="Mahadev_143",
         expiresIn=86400,
         by_sms=True)

def nearest(list, target):
    return min(list, key=lambda x: abs(x - target))

if not Signal_df.empty:

    final_orders_raw_data = db["final_orders_raw_data"]
    Signal_df.reset_index(inplace=True, drop=True)
    
    for i in range(0, len(Signal_df)):
        print(Signal_df.loc[i,])
        
        side_dir = "call" if Signal_df.loc[i, "Signal"] == "Buy" else "put"
        
        try:
            tradable_options = robin_stocks.robinhood.options.find_tradable_options(Signal_df.loc[i, "Stock"],expirationDate=expiry_date_char,optionType=side_dir)
            tradable_options = pd.DataFrame(tradable_options)

            strike_prices = list(set(tradable_options['strike_price']))

            integer_numbers = [int(float(x)) for x in strike_prices]

            nearest_number = nearest(integer_numbers, Signal_df.loc[i, "Value"])

            print(nearest_number)
            Signal_df.loc[i, "Spot_Price"] = nearest_number
            
            
            option_data = robin_stocks.robinhood.options.get_option_historicals(Signal_df.loc[i, "Stock"],
                                  expiry_date_char,
                                  str(nearest_number),
                                  side_dir,
                                  interval='5minute',
                                  span='week',
                                  bounds='regular')
            option_data = pd.DataFrame(option_data)
            
            option_data['begins_at'] = pd.to_datetime(option_data['begins_at'], format='%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=5)
            option_data['begins_at'] = option_data['begins_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
#             print(option_data)
            
            current_time = datetime.strptime(Signal_df.loc[i, 'Datetime'], '%Y-%m-%d %H:%M:%S.%f')
            current_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(current_time)
            
            filtered_data = option_data[option_data['begins_at'] == current_time]
            print(filtered_data)
            
            filtered_data.reset_index(inplace=True,drop=True)
            
            Signal_df.loc[i, "Strike_Buy_Price"] = float(filtered_data.loc[0,'close_price'])
            
            stop_loss = 10

            target = 10

            risk_on_capital = 0.05

            capital = 10000

            lot_size = Capital * risk_on_capital / float(filtered_data.loc[0,'close_price'])

            Signal_df.loc[i, "premium_StopLoss"] = round(
                Signal_df.loc[i, "Strike_Buy_Price"]
                - ((stop_loss * Signal_df.loc[i, "Strike_Buy_Price"]) / 100),
                2,
            )
            Signal_df.loc[i, "premium_Target"] = round(
                Signal_df.loc[i, "Strike_Buy_Price"]
                + ((target * Signal_df.loc[i, "Strike_Buy_Price"]) / 100),
                2,
            )
            
#             Signal_df.loc[i, "current_script"] = lookup_symbol
#             Signal_df.loc[i, "token"] = token
            Signal_df.loc[i, "conclusion"] = "New Order"
            
            
        except Exception as e:
            print(str(e))
            print("Missing the latest option expiry")
            
    ## Delete the existing orders pushed for today
    x = final_orders_raw_data.delete_many(
        {"execution_date": (datetime.now().strftime("%Y-%m-%d"))}
    )

    print(x.deleted_count, " documents deleted.")

    final_orders_raw_data.insert_many(Signal_df.to_dict("records"))
    

end_time = datetime.now() 

print(end_time)

print("Duration: {}".format(end_time - start_time))
