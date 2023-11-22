import pandas as pd
import os
from datetime import datetime, timedelta
from smartapi import SmartConnect
from pandasql import sqldf
import math
from pandas.io.json import json_normalize
import json
import numpy as np
import yfinance as yf
import pandas_ta as ta
import time
import datetime as dt
from pytz import timezone
import pyotp
import requests

# import mysql.connector as mysql
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
from pymongo import MongoClient
from pymongo.server_api import ServerApi


pd.set_option("display.max_columns", None)

start_time = datetime.now(timezone("Asia/Kolkata"))
print("Script execution started")
print(start_time)


angel_script = pd.read_csv(
    "/home/sjonnal3/Hate_Speech_Detection/Trading_Application/angel_script.csv"
)

# client_data = pd.read_csv("/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Client_Details.csv")


# engine = create_engine("mysql+pymysql://root:Mahadev_143@localhost/titania_trading")
# print(engine)


# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()


# sql = "select * from titania_trading.client_details where client_id in ('J95213','S1604557','G304915','K256027')"
# print(sql)
# client_data = pd.read_sql(sql,con=engine)


server_api = ServerApi("1")

client = MongoClient(
    "mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE",
    server_api=server_api,
)
db = client["titania_trading"]

# client_details = db["client_details"].find({'client_id':{'$in' :['J95213','S1604557','G304915','K256027','M591295','M181705']}})
client_details = db["client_details"].find(
    {"client_id": {"$in": ["J95213", "M591295"]}}
)
client_data = pd.DataFrame(list(client_details))

print(client_data)

bot_token = '1931575614:AAFhtU1xieFDqC9WAAzw15G4KdB8rdzrif4'
chat_id = ["535162272"]


def place_stoploss_market(symbol, token, price, quantity, buy_price):

    # order_params = {
    #                 "variety":"STOPLOSS",
    #                 "tradingsymbol":str(symbol),
    #                 "symboltoken":str(token),
    #                 "transactiontype":"SELL",
    #                 "exchange":"NFO",
    #                 "ordertype":"STOPLOSS_MARKET",
    #                 "producttype":"CARRYFORWARD",
    #                 "duration":"DAY",
    #                 "triggerprice":str(price),
    #                 "price":"0",
    #                 "quantity":str(quantity)
    #             }

    print("This is to understand the iv based targets ")
    print("inside place_stoploss_market")
    print(str(symbol))
    print(str(token))
    print(str(price))
    price = int(round(price))
    print(str(price))
    print(str(avg_order_price))
    print(str(quantity))
    print(int(float(price)) + 0.5)

    #     BANKNIFTY15DEC2243300CE
    #     45739
    #     332.8
    #     387.55
    #     25

    order_params = {
        "variety": "STOPLOSS",
        "tradingsymbol": str(symbol),
        "symboltoken": str(token),
        "transactiontype": "SELL",
        "exchange": "NFO",
        "ordertype": "STOPLOSS_LIMIT",
        "producttype": "CARRYFORWARD",
        "duration": "DAY",
        "price": str(price),
        "squareoff": str(avg_order_price),
        "stoploss": str(price),
        "quantity": str(quantity),
        "triggerprice": int(float(price)) + 0.5,
    }

    order_place_sl = obj.placeOrder(order_params)

    chat_message = (
                    "************** Placing the target order for client id :"
                    + str(user_id)
                    + "\n  Symbol : "
                    + str(symbol)
                    + "\n Target Price :"
                    + str(price)
                    + "\n Quantity :"
                    + str(quantity)
                    + "\n Order ID :"
                    + str(order_place_sl)
                )
    print("Sending the message")
    for cht in chat_id:
        message = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + str(cht) + '&parse_mode=Markdown&text=' + str(chat_message)
        send = requests.post(message)

    return order_place_sl


def place_target_order(symbol, token, price, quantity):

    print("This is to understand the iv based targets ")
    print("inside place_stoploss_market")
    print(str(symbol))
    print(str(token))
    print(str(price))
    print(str(quantity))

    order_params = {
        "variety": "NORMAL",
        "tradingsymbol": str(symbol),
        "symboltoken": str(token),
        "transactiontype": "SELL",
        "exchange": "NFO",
        "ordertype": "LIMIT",
        "producttype": "CARRYFORWARD",
        "duration": "DAY",
        "price": str(price),
        "squareoff": 0,
        "stoploss": 0,
        "quantity": str(quantity),
    }
    order_place = obj.placeOrder(order_params)

    chat_message = (
                    "************** Placing the target order for client id :"
                    + str(user_id)
                    + "\n  Symbol : "
                    + str(symbol)
                    + "\n Target Price :"
                    + str(price)
                    + "\n Quantity :"
                    + str(quantity)
                    + "\n Order ID :"
                    + str(order_place)
                )

    print("Sending the message")
    for cht in chat_id:
        message = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + str(cht) + '&parse_mode=Markdown&text=' + str(chat_message)
        send = requests.post(message)

    return order_place


current_time = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S")

if current_time >= "09:15:00":

    for i in range(0, len(client_data)):
        try:
            user_id = str(client_data.loc[i, "client_id"])
            print("Running script for ", str(client_data.loc[i, "client_name"]))

            previous_orders = pd.DataFrame()

            if os.path.isfile(
                "/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Orders_Data/"
                + str(client_data.loc[i, "client_id"])
                + "/Updated_Targets/IV_Based_"
                + datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
                + ".csv"
            ):
                print("File Exists")
                previous_orders = pd.read_csv(
                    "/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Orders_Data/"
                    + str(client_data.loc[i, "client_id"])
                    + "/Updated_Targets/IV_Based_"
                    + datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
                    + ".csv"
                )
            else:
                print("Previous Orders is Null")
                previous_orders = pd.DataFrame(
                    columns=[
                        "symboltoken",
                        "symbolname",
                        "instrumenttype",
                        "priceden",
                        "pricenum",
                        "genden",
                        "gennum",
                        "precision",
                        "multiplier",
                        "boardlotsize",
                        "exchange",
                        "producttype",
                        "tradingsymbol",
                        "symbolgroup",
                        "strikeprice",
                        "optiontype",
                        "expirydate",
                        "lotsize",
                        "cfbuyqty",
                        "cfsellqty",
                        "cfbuyamount",
                        "cfsellamount",
                        "buyavgprice",
                        "sellavgprice",
                        "avgnetprice",
                        "netvalue",
                        "netqty",
                        "totalbuyvalue",
                        "totalsellvalue",
                        "cfbuyavgprice",
                        "cfsellavgprice",
                        "totalbuyavgprice",
                        "totalsellavgprice",
                        "netprice",
                        "buyqty",
                        "sellqty",
                        "buyamount",
                        "sellamount",
                        "pnl",
                        "realised",
                        "unrealised",
                        "ltp",
                        "close",
                        "target_order_id",
                        "stop_loss_order_id",
                        "final_order_id",
                        "cancel_order_id",
                        "max_profit_percent",
                        "min_profit_percent",
                        "current_stoploss_percent",
                        "current_prob",
                        "implied_volatility_time",
                        "implied_volatility",
                        "IV_Stoploss",
                        "IV_Target",
                    ]
                )

            obj = SmartConnect(api_key=client_data.loc[i, "client_api_key"])
            totp = pyotp.TOTP(client_data.loc[i, "totp_code"])
            print("pyotp", totp.now())
            attempts = 5
            while attempts > 0:
                attempts = attempts - 1
                #         data = obj.generateSession(client_data.loc[i,"client_id"],client_data.loc[i,"client_password"], totp.now())
                data = obj.generateSession(
                    client_data.loc[i, "client_id"], client_data.loc[i, "m_pin"], totp.now()
                )
                if data["status"]:
                    break
                # tt.sleep(2)

            print(data)

            my_positons_data = obj.position()
            current_position_data = pd.DataFrame(my_positons_data["data"])

            my_orders = obj.orderBook()
            order_data = pd.DataFrame(my_orders["data"])

            pysqldf = lambda q: sqldf(q, globals())

            #     print(current_position_data)

            if len(current_position_data) > 0:
                if not "current_prob" in previous_orders.columns:
                    previous_orders["current_prob"] = 40

                current_position_data = pysqldf(
                    """
                                    select t1.symboltoken,
                                        t1.symbolname,
                                        t1.instrumenttype,
                                        t1.priceden,
                                        t1.pricenum,
                                        t1.genden,
                                        t1.gennum,
                                        t1.precision,
                                        t1.multiplier,
                                        t1.boardlotsize,
                                        t1.exchange,
                                        t1.producttype,
                                        t1.tradingsymbol,
                                        t1.symbolgroup,
                                        t1.strikeprice,
                                        t1.optiontype,
                                        t1.expirydate,
                                        t1.lotsize,
                                        t1.cfbuyqty,
                                        t1.cfsellqty,
                                        t1.cfbuyamount,
                                        t1.cfsellamount,
                                        t1.buyavgprice,
                                        t1.sellavgprice,
                                        t1.avgnetprice,
                                        t1.netvalue,
                                        t1.netqty,
                                        t1.totalbuyvalue,
                                        t1.totalsellvalue,
                                        t1.cfbuyavgprice,
                                        t1.cfsellavgprice,
                                        t1.totalbuyavgprice,
                                        t1.totalsellavgprice,
                                        t1.netprice,
                                        t1.buyqty,
                                        t1.sellqty,
                                        t1.buyamount,
                                        t1.sellamount,
                                        t1.pnl,
                                        t1.realised,
                                        t1.unrealised,
                                        t1.ltp,
                                        t1.close,
                                        t2.target_order_id,
                                        t2.stop_loss_order_id,
                                        t2.final_order_id,
                                        t2.cancel_order_id,
                                        t2.netqty as prev_netqty,
                                        COALESCE(t2.max_profit_percent,0) as max_profit_percent,
                                        COALESCE(t2.min_profit_percent,0) as min_profit_percent,
                                        COALESCE(t2.current_stoploss_percent,0) as current_stoploss_percent,
                                        COALESCE(t2.current_prob,0) as middle_prob,
                                        t2.implied_volatility_time,
                                        COALESCE(t2.implied_volatility,0) as implied_volatility,
                                        COALESCE(t2.IV_Stoploss,0) as IV_Stoploss,
                                        COALESCE(t2.IV_Target,0) as  IV_Target

              from current_position_data t1 
              left join previous_orders t2 on t1.tradingsymbol = t2.tradingsymbol
                                """
                )

                current_position_data["netqty"] = current_position_data["netqty"].fillna(0)
                current_position_data["prev_netqty"] = current_position_data[
                    "prev_netqty"
                ].fillna(0)

                #         print(current_position_data)

                #         print(order_data)

                if len(order_data) > 0:

                    completed_orders = order_data[
                        order_data["status"] == "complete"
                    ].sort_values(by="exchorderupdatetime")

                    completed_orders["row_number_bygroup"] = (
                        completed_orders.groupby(
                            ["tradingsymbol", "transactiontype"]
                        ).cumcount()
                        + 1
                    )

                    print("Completed orders")
                    print(completed_orders["tradingsymbol"])

                pysqldf = lambda q: sqldf(q, globals())

                if len(current_position_data) > 0:
                    for ord in range(0, len(current_position_data)):
                        print(
                            "The Current Order is :",
                            str(current_position_data.loc[ord, "tradingsymbol"]),
                        )

                        if (
                            current_position_data.loc[ord, "tradingsymbol"]
                            == "WIPRO28JUL22410CE"
                            or current_position_data.loc[ord, "tradingsymbol"]
                            == "NIFTY11AUG2217200PE"
                            or current_position_data.loc[ord, "tradingsymbol"]
                            == "NIFTY22SEP2217800PE"
                        ):
                            print("Wipro order")
                        else:
                            if int(float(current_position_data.loc[ord, "netprice"])) > 0:
                                current_strike_completed = ""
                                if len(order_data) > 0:
                                    current_strike_completed = completed_orders[
                                        completed_orders["tradingsymbol"]
                                        == current_position_data.loc[ord, "tradingsymbol"]
                                    ].sort_values(by="exchorderupdatetime")

                                avg_order_price = 0

                                print("Open Positions")

                                open_position = pd.DataFrame(
                                    current_position_data.loc[
                                        ord,
                                    ]
                                )
                                open_position = open_position.T
                                open_position.reset_index(inplace=True, drop=True)

                                print(open_position.loc[0, "tradingsymbol"])

                                executed_orders = pd.DataFrame()

                                if len(order_data) > 0:

                                    executed_orders = order_data[
                                        (
                                            order_data["tradingsymbol"]
                                            == open_position.loc[0, "tradingsymbol"]
                                        )
                                        & (order_data["status"] == "complete")
                                    ]

                                    executed_orders.reset_index(inplace=True, drop=True)

                                #                             print("executed_order")
                                #                             print(executed_orders)

                                stock = ""
                                Signal = ""
                                initial_prob = 0
                                current_prob = 0
                                middle_prob = current_position_data.loc[ord, "middle_prob"]

                                today_order = True

                                collection = db.Stocks_data_5_minutes
                                technical_collection = db.technical_indicator_5_minutes

                                live_data = pd.DataFrame()
                                technical_data = pd.DataFrame()

                                if executed_orders.empty:
                                    print("Not Todays Order")

                                    today_order = False

                                    #                             opt_sql = ""
                                    #                             technical_sql = ""

                                    if "BANKNIFTY" in open_position.loc[0, "tradingsymbol"]:
                                        print("BankNifty Order")
                                        stock = "%5ENSEBANK"
                                        live_data = collection.find(
                                            {
                                                "instrumenttype": "FUTIDX",
                                                "Stock": "BankNifty",
                                            }
                                        )
                                        technical_data = (
                                            technical_collection.find(
                                                {"Stock": "BankNifty"}
                                            )
                                            .sort("Datetime", -1)
                                            .limit(1)
                                        )

                                    #                                 opt_sql = "select distinct * from Stocks_data_5_minutes where instrumenttype = 'OPTIDX' and Stock = 'BankNifty' order by Datetime asc"
                                    #                                 technical_sql = "select * from technical_indicator_5_minutes where stock ='BankNifty' order by Datetime desc limit 1"
                                    elif "NIFTY" in open_position.loc[0, "tradingsymbol"]:
                                        stock = "%5ENSEI"
                                        print("Nifty order")
                                        live_data = collection.find(
                                            {"instrumenttype": "FUTIDX", "Stock": "Nifty"}
                                        )
                                        technical_data = (
                                            technical_collection.find({"Stock": "Nifty"})
                                            .sort("Datetime", -1)
                                            .limit(1)
                                        )
                                    #                                 opt_sql = "select distinct * from Stocks_data_5_minutes where instrumenttype = 'OPTIDX' and Stock = 'Nifty' order by Datetime asc"
                                    #                                 technical_sql = "select * from technical_indicator_5_minutes where stock ='Nifty' order by Datetime desc limit 1"
                                    else:
                                        print("Other Symbol")

                                    if (
                                        open_position.loc[0, "tradingsymbol"][
                                            len(open_position.loc[0, "tradingsymbol"]) - 2 :
                                        ]
                                        == "CE"
                                    ):
                                        Signal = "BUY"
                                    else:
                                        Signal = "SELL"

                                    # print(Signal)

                                    #                             data = pd.read_sql(opt_sql,con=engine)
                                    #                             data.reset_index(level=0, inplace=True,drop = True)

                                    #                             technical_data = pd.read_sql(technical_sql,con=engine)
                                    #                             technical_data.reset_index(level=0, inplace=True,drop = True)

                                    #                             print(data)

                                    #                             data['Datetime'] = pd.to_datetime(data['Datetime'])

                                    global nifty_realtime_data
                                    global banknifty_realtime_data
                                    now = datetime.now(timezone("Asia/Kolkata"))

                                elif len(executed_orders) == 1:
                                    print("This is todays order")
                                    if (
                                        "BANKNIFTY"
                                        in executed_orders.loc[0, "tradingsymbol"]
                                    ):
                                        print("BankNifty Order")
                                        stock = "%5ENSEBANK"

                                        live_data = collection.find(
                                            {
                                                "instrumenttype": "FUTIDX",
                                                "Stock": "BankNifty",
                                            }
                                        )
                                        technical_data = technical_collection.find(
                                            {"Stock": "BankNifty"}
                                        ).sort("Datetime", -1)

                                    #                                 opt_sql = "select distinct * from Stocks_data_5_minutes where instrumenttype = 'OPTIDX' and Stock = 'BankNifty' order by Datetime asc"
                                    #                                 technical_sql = "select * from technical_indicator_5_minutes where stock ='BankNifty' order by Datetime desc"
                                    elif "NIFTY" in open_position.loc[0, "tradingsymbol"]:
                                        stock = "%5ENSEI"
                                        print("Nifty order")

                                        live_data = collection.find(
                                            {"instrumenttype": "FUTIDX", "Stock": "Nifty"}
                                        )
                                        technical_data = technical_collection.find(
                                            {"Stock": "Nifty"}
                                        ).sort("Datetime", -1)
                                    #                                 opt_sql = "select distinct * from Stocks_data_5_minutes where instrumenttype = 'OPTIDX' and Stock = 'Nifty' order by Datetime asc"
                                    #                                 technical_sql = "select * from technical_indicator_5_minutes where stock ='Nifty' order by Datetime desc"
                                    else:
                                        print("Other Symbol")

                                    if (
                                        executed_orders.loc[0, "tradingsymbol"][
                                            len(executed_orders.loc[0, "tradingsymbol"])
                                            - 2 :
                                        ]
                                        == "CE"
                                    ):
                                        Signal = "BUY"
                                    else:
                                        Signal = "SELL"

                                    # data = pd.read_sql(opt_sql,con=engine)
                                    # data.reset_index(level=0, inplace=True,drop = True)

                                    #                             technical_data = pd.read_sql(technical_sql,con=engine)
                                    #                             technical_data.reset_index(level=0, inplace=True,drop = True)

                                    live_data = pd.DataFrame(list(live_data))
                                    technical_data = pd.DataFrame(list(technical_data))

                                    live_data["Datetime"] = live_data[
                                        "Datetime"
                                    ] + timedelta(hours=5, minutes=30)

                                    #                             print("Technical Data")
                                    #                             print(technical_data)

                                    cr_date = executed_orders.loc[0, "exchorderupdatetime"]
                                    cr_date = datetime.strptime(
                                        cr_date, "%d-%b-%Y %H:%M:%S"
                                    )

                                    hms = dt.timedelta(
                                        hours=cr_date.hour,
                                        minutes=cr_date.minute,
                                        seconds=cr_date.second,
                                    )

                                    resolution = dt.timedelta(minutes=5)

                                    time_sub = dt.timedelta(
                                        seconds=hms.seconds % resolution.seconds
                                    )

                                    signal_time = cr_date - time_sub

                                    print("signal time")
                                    print(signal_time.strftime("%Y-%m-%d %H:%M:%S"))

                                    print(technical_data.tail(5))

                                    signal_indicator = technical_data[
                                        technical_data["Datetime"]
                                        <= signal_time.strftime("%Y-%m-%d %H:%M:%S")
                                    ]
                                    signal_indicator = signal_indicator.sort_values(
                                        by="Datetime"
                                    )
                                    signal_indicator = signal_indicator.tail(1)
                                    print(signal_indicator)
                                    current_indicator = technical_data[
                                        technical_data["Datetime"]
                                        == max(technical_data["Datetime"])
                                    ]

                                    signal_indicator.reset_index(
                                        level=0, inplace=True, drop=True
                                    )
                                    current_indicator.reset_index(
                                        level=0, inplace=True, drop=True
                                    )

                                    print("Signal Indicator")
                                    print(signal_indicator)

                                    initial_prob = 0
                                    current_prob = 0

                                    if Signal == "BUY":
                                        initial_prob = signal_indicator.loc[
                                            0, "buy_probability"
                                        ]
                                        current_prob = current_indicator.loc[
                                            0, "buy_probability"
                                        ]
                                    else:
                                        initial_prob = signal_indicator.loc[
                                            0, "sell_probability"
                                        ]
                                        current_prob = current_indicator.loc[
                                            0, "sell_probability"
                                        ]

                                    print("Initial Probability : ", str(initial_prob))
                                    print("Current Probability : ", str(current_prob))

                                else:
                                    print("Carry further calculation")

                                latest_update_time = ""
                                #                         print("current_strike_completed")
                                #                         print(current_strike_completed)
                                if len(current_strike_completed) > 0:
                                    print("inside current_strike_completed")
                                    n_occur = pd.DataFrame(
                                        current_strike_completed[
                                            "row_number_bygroup"
                                        ].value_counts()
                                    )
                                    n_occur["index"] = n_occur.index
                                    open_orders = current_strike_completed[
                                        current_strike_completed["row_number_bygroup"].isin(
                                            n_occur[n_occur["row_number_bygroup"] == 1][
                                                "index"
                                            ]
                                        )
                                    ]

                                    open_orders.reset_index(inplace=True, drop=True)

                                    print(open_orders)
                                    latest_update_time = max(open_orders["updatetime"])

                                    #                             avg_order_price = open_orders['averageprice'].mean()
                                    avg_order_price = open_position.loc[0, "netprice"]
                                else:
                                    #                             print("debugging")
                                    avg_order_price = current_position_data.loc[
                                        ord, "netprice"
                                    ]
                                #                             latest_average = open_position.loc[0,'netprice']

                                print(avg_order_price)

                                current_close = current_position_data.loc[ord, "ltp"]

                                print("current_close : ", str(current_close))

                                price_diff = (
                                    (float(avg_order_price) - float(current_close))
                                    * 100
                                    / float(avg_order_price)
                                )

                                print("price_diff : ", str(price_diff))

                                print(
                                    "Running the analysis for ",
                                    client_data.loc[i, "client_id"],
                                )

                                print("Average buy price : ", avg_order_price)

                                stop_loss_order_id = open_position.loc[
                                    0, "stop_loss_order_id"
                                ]
                                final_order_id = open_position.loc[0, "final_order_id"]
                                target_order_id = open_position.loc[0, "target_order_id"]

                                profit_pct = 0
                                min_profit_pct = 0

                                # if the current_price is less than
                                # ## For target as 10%

                                if current_position_data.loc[ord, "max_profit_percent"] > 0:
                                    profit_pct = max(
                                        current_position_data.loc[
                                            ord, "max_profit_percent"
                                        ],
                                        0
                                        if price_diff >= -3
                                        else 3
                                        if (price_diff > -3 and price_diff <= -7)
                                        else 7
                                        if (price_diff > -7 and price_diff <= -10)
                                        else 10
                                        if (price_diff >= -20 and price_diff <= -10)
                                        else 20,
                                    )
                                else:

                                    if price_diff <= -3:
                                        profit_pct = max(
                                            current_position_data.loc[
                                                ord, "max_profit_percent"
                                            ],
                                            0
                                            if price_diff >= -3
                                            else 3
                                            if (price_diff > -3 and price_diff <= -7)
                                            else 7
                                            if (price_diff > -7 and price_diff <= -10)
                                            else 10
                                            if (price_diff >= -20 and price_diff <= -10)
                                            else 20,
                                        )
                                    else:
                                        profit_pct = 0
                                if current_position_data.loc[ord, "min_profit_percent"] > 0:
                                    min_profit_pct = min(
                                        current_position_data.loc[
                                            ord, "min_profit_percent"
                                        ],
                                        0
                                        if min_profit_pct >= -3
                                        else 3
                                        if (min_profit_pct > -3 and min_profit_pct <= -7)
                                        else 7
                                        if (min_profit_pct > -7 and min_profit_pct <= -10)
                                        else 10
                                        if (min_profit_pct >= -20 and min_profit_pct <= -10)
                                        else 20,
                                    )

                                else:

                                    if price_diff <= -3:
                                        min_profit_pct = min(
                                            current_position_data.loc[
                                                ord, "min_profit_percent"
                                            ],
                                            0
                                            if min_profit_pct >= -3
                                            else 3
                                            if (
                                                min_profit_pct > -3 and min_profit_pct <= -7
                                            )
                                            else 7
                                            if (
                                                min_profit_pct > -7
                                                and min_profit_pct <= -10
                                            )
                                            else 10
                                            if (
                                                min_profit_pct >= -20
                                                and min_profit_pct <= -10
                                            )
                                            else 20,
                                        )
                                    else:
                                        min_profit_pct = 0

                                current_position_data.loc[
                                    ord, "max_profit_percent"
                                ] = profit_pct
                                current_position_data.loc[
                                    ord, "min_profit_percent"
                                ] = min_profit_pct

                                print("orders check : ")

                                orders_check = ""
                                target_pct = 0

                                print(final_order_id)
                                print(target_order_id)
                                print(stop_loss_order_id)

                                if final_order_id is None or np.isnan(final_order_id):
                                    orders_check = "place_initial_order"
                                    print("No Order Placed")
                                elif (
                                    target_order_id is not None
                                    and target_order_id == final_order_id
                                ):
                                    orders_check = "placed_target_order"
                                    print("Target Order Placed")
                                elif (
                                    stop_loss_order_id is not None
                                    and stop_loss_order_id == final_order_id
                                ):
                                    orders_check = "placed_stoploss_order"
                                    print("Stop Loss Order Placed")
                                else:
                                    print("Understand the order")

                                #                         print(current_position_data)

                                stock = current_position_data.loc[ord, "symbolname"]
                                curr_stock = ""
                                if stock == "NIFTY":
                                    curr_stock = "Nifty"
                                else:
                                    curr_stock = "BankNifty"

                                side_dir = current_position_data.loc[ord, "optiontype"]

                                expiry = current_position_data.loc[ord, "expirydate"]

                                expiry_date = datetime.strptime(expiry, "%d%b%Y").strftime(
                                    "%d-%m-%Y"
                                )
                                formatted_expiry_date = datetime.strptime(
                                    expiry, "%d%b%Y"
                                ).strftime("%d-%b-%Y")

                                strike_price = current_position_data.loc[ord, "strikeprice"]

                                current_script = current_position_data.loc[
                                    ord, "tradingsymbol"
                                ]

                                print(stock)
                                print(side_dir)
                                print(expiry)
                                print(expiry_date)
                                print(strike_price)
                                print(current_script)

                                order_identifier = (
                                    "OPTIDX"
                                    + str(stock.upper())
                                    + str(expiry_date)
                                    + str(side_dir)
                                    + str(strike_price)
                                    + "0"
                                )

                                print(order_identifier)

                                current_data = pd.read_csv(
                                    "/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/"
                                    + str(curr_stock)
                                    + "/"
                                    + str(formatted_expiry_date)
                                    + "/"
                                    + str(int(float(strike_price)))
                                    + ".csv",
                                    index_col=False,
                                )

                                #                         print(current_data.tail())

                                current_data = current_data.loc[
                                    current_data["identifier"] == order_identifier
                                ]
                                current_data.reset_index(inplace=True, drop=True)

                                #                         print(current_data.tail())

                                current_data["time"] = pd.to_datetime(
                                    current_data.time.str.strip(),
                                    errors="coerce",
                                    format="%Y-%m-%d %H:%M:%S",
                                    utc=True,
                                )

                                current_data["time"] = current_data["time"] + timedelta(
                                    hours=5, minutes=30
                                )

                                current_data["time"] = current_data.time.dt.strftime(
                                    "%Y-%m-%d %H:%M:00"
                                )

                                current_data["time"] = current_data["time"].astype(
                                    "datetime64[ns]"
                                )

                                current_data = current_data.sort_values(by="time")

                                trade_implied_volatility = 0

                                if not (today_order):
                                    print("Not order")
                                    print(latest_update_time)
                                    latest_filtered_data = current_data.tail(1)

                                    latest_filtered_data.reset_index(
                                        inplace=True, drop=True
                                    )
                                    trade_implied_volatility = latest_filtered_data.loc[
                                        0, "impliedvolatility"
                                    ]
                                    current_position_data.loc[
                                        ord, "implied_volatility_time"
                                    ] = latest_filtered_data.loc[0, "time"]

                                else:
                                    print("Consider latest IV logic")
                                    print(latest_update_time)
                                    filtered_data_below = current_data.loc[
                                        current_data["time"] <= latest_update_time
                                    ]

                                    filtered_data_below = filtered_data_below.tail(1)
                                    filtered_data_below.reset_index(inplace=True, drop=True)

                                    filtered_data_above = current_data.loc[
                                        current_data["time"] >= latest_update_time
                                    ]

                                    filtered_data_above = filtered_data_above.head(1)
                                    filtered_data_above.reset_index(inplace=True, drop=True)

                                    #                             print(filtered_data_below)
                                    #                             print(filtered_data_above)

                                    if (
                                        len(filtered_data_below) >= 1
                                        and len(filtered_data_above) >= 1
                                    ):
                                        print("The latest should be considered")
                                        print(filtered_data_below.loc[0, "time"])
                                        print(filtered_data_above.loc[0, "time"])

                                        latest_update_time = datetime.strptime(
                                            latest_update_time, "%d-%b-%Y %H:%M:%S"
                                        )

                                        below_time_diff = (
                                            latest_update_time
                                            - filtered_data_below.loc[0, "time"]
                                        )
                                        above_time_diff = (
                                            filtered_data_above.loc[0, "time"]
                                            - latest_update_time
                                        )

                                        if above_time_diff <= below_time_diff:
                                            print("Consider above time")
                                            trade_implied_volatility = (
                                                filtered_data_above.loc[
                                                    0, "impliedvolatility"
                                                ]
                                            )
                                            current_position_data.loc[
                                                ord, "implied_volatility_time"
                                            ] = filtered_data_above.loc[0, "time"]
                                        elif below_time_diff <= above_time_diff:
                                            print("Consider the below time")
                                            trade_implied_volatility = (
                                                filtered_data_below.loc[
                                                    0, "impliedvolatility"
                                                ]
                                            )
                                            current_position_data.loc[
                                                ord, "implied_volatility_time"
                                            ] = filtered_data_below.loc[0, "time"]
                                        else:
                                            ("Latest think !!")
                                    elif len(filtered_data_above) >= 1:
                                        print("Consider the Above IV")
                                        trade_implied_volatility = filtered_data_above.loc[
                                            0, "impliedvolatility"
                                        ]
                                        current_position_data.loc[
                                            ord, "implied_volatility_time"
                                        ] = filtered_data_above.loc[0, "time"]
                                    elif len(filtered_data_below) >= 1:
                                        print("Consider the below IV")
                                        trade_implied_volatility = filtered_data_below.loc[
                                            0, "impliedvolatility"
                                        ]
                                        current_position_data.loc[
                                            ord, "implied_volatility_time"
                                        ] = filtered_data_below.loc[0, "time"]
                                    else:
                                        print("Think about it")

                                print(trade_implied_volatility)

                                current_position_data.loc[ord, "implied_volatility"] = int(
                                    trade_implied_volatility
                                )

                                current_position_data.loc[ord, "IV_Stoploss"] = round(
                                    int(float(avg_order_price))
                                    - (
                                        int(trade_implied_volatility)
                                        * int(float(avg_order_price))
                                    )
                                    / 100,
                                    1,
                                )
                                #                         current_position_data.loc[ord,"IV_Target"] = round(int(float(avg_order_price)) + 1.5 * (int(trade_implied_volatility) * int(float(avg_order_price)))/100,2)
                                current_position_data.loc[ord, "IV_Target"] = round(
                                    int(float(avg_order_price))
                                    + (
                                        int(trade_implied_volatility)
                                        * int(float(avg_order_price))
                                    )
                                    / 100,
                                    1,
                                )

                                net_qty = int(current_position_data.loc[ord, "netqty"])
                                #                         net_qty = "1"

                                if price_diff > 0:
                                    print("In Loss")

                                    print(current_position_data)

                                    stop_loss_price = current_position_data.loc[
                                        ord, "IV_Stoploss"
                                    ]

                                    if orders_check == "place_initial_order":
                                        print("place stop loss order")

                                        order_place_sl = place_stoploss_market(
                                            str(open_position.loc[0, "tradingsymbol"]),
                                            str(open_position.loc[0, "symboltoken"]),
                                            stop_loss_price,
                                            net_qty,
                                            avg_order_price,
                                        )
                                        #                                 order_place_sl = 1
                                        print(
                                            "Placed the stoploss order : ", order_place_sl
                                        )
                                        current_position_data.loc[
                                            ord, "stop_loss_order_id"
                                        ] = order_place_sl
                                        current_position_data.loc[
                                            ord, "final_order_id"
                                        ] = order_place_sl
                                    elif orders_check == "placed_target_order":
                                        print("Cancel the target order as it went to loss")

                                        cancel_id = obj.cancelOrder(
                                            order_id=str(int(float(final_order_id))),
                                            variety="NORMAL",
                                        )

                                        cancel_id = cancel_id["data"]["orderid"]
                                        #                                 cancel_id= 2
                                        print("Cancelled the target order : ", cancel_id)

                                        current_position_data.loc[
                                            ord, "cancel_order_id"
                                        ] = cancel_id

                                        ## Place the Stoploss order

                                        # order_place_sl = place_stoploss_market(str(open_position.loc[0,"tradingsymbol"]),str(open_position.loc[0,"symboltoken"]),current_close,net_qty,avg_order_price)
                                        order_place_sl = place_stoploss_market(
                                            str(open_position.loc[0, "tradingsymbol"]),
                                            str(open_position.loc[0, "symboltoken"]),
                                            stop_loss_price,
                                            net_qty,
                                            avg_order_price,
                                        )
                                        #                                 order_place_sl = 1
                                        print(
                                            "Placed the stoploss order : ", order_place_sl
                                        )
                                        current_position_data.loc[
                                            ord, "stop_loss_order_id"
                                        ] = order_place_sl
                                        current_position_data.loc[
                                            ord, "final_order_id"
                                        ] = order_place_sl

                                    elif orders_check == "placed_stoploss_order":
                                        print("Stoploss Check continues")
                                    else:
                                        print("Think about it")

                                else:
                                    print("In Profit")

                                    print(current_position_data)

                                    target_price = current_position_data.loc[
                                        ord, "IV_Target"
                                    ]

                                    if orders_check == "place_initial_order":
                                        #                                 target_price = current_position_data.loc[ord,'IV_Target']
                                        print(target_price)
                                        print(net_qty)
                                        order_place = place_target_order(
                                            str(open_position.loc[0, "tradingsymbol"]),
                                            str(open_position.loc[0, "symboltoken"]),
                                            target_price,
                                            net_qty,
                                        )

                                        print(order_place)
                                        #                                 order_place = 1
                                        current_position_data.loc[
                                            ord, "target_order_id"
                                        ] = order_place
                                        current_position_data.loc[
                                            ord, "final_order_id"
                                        ] = order_place
                                    elif orders_check == "placed_target_order":
                                        print("Target Check continues")
                                    elif orders_check == "placed_stoploss_order":
                                        print(
                                            "Cancel the stoploss order as it went to profit"
                                        )

                                        cancel_id = obj.cancelOrder(
                                            order_id=str(int(float(final_order_id))),
                                            variety="NORMAL",
                                        )
                                        cancel_id = cancel_id["data"]["orderid"]
                                        #                                 cancel_id = 2
                                        print("Cancelled the stoploss order: ", cancel_id)

                                        current_position_data.loc[
                                            ord, "cancel_order_id"
                                        ] = cancel_id

                                        order_place = place_target_order(
                                            str(open_position.loc[0, "tradingsymbol"]),
                                            str(open_position.loc[0, "symboltoken"]),
                                            target_price,
                                            net_qty,
                                        )

                                        #                                 order_place = 1
                                        print("Placed the target order : ", order_place)

                                        current_position_data.loc[
                                            ord, "target_order_id"
                                        ] = order_place
                                        current_position_data.loc[
                                            ord, "final_order_id"
                                        ] = order_place
                                    else:
                                        print("Think about it")
                    print(current_position_data)

                    if len(current_position_data) > 0:
                        table = "target_and_stoploss_orders"
                        destination_table = "Titania_Dataset." + table
                        current_position_data["Client_Id"] = str(user_id)
                        current_position_data["execution_date"] = (
                            datetime.now(timezone("Asia/Kolkata"))
                        ).strftime("%Y-%m-%d")

                        target_and_stoploss_tbl = db[str(table)]

                        x = target_and_stoploss_tbl.delete_many(
                            {
                                "Client_Id": str(user_id),
                                "execution_date": (
                                    datetime.now(timezone("Asia/Kolkata"))
                                ).strftime("%Y-%m-%d"),
                            }
                        )

                        print(x.deleted_count, " documents deleted.")

                        final_order_data = current_position_data.astype(str)

                        target_and_stoploss_tbl.insert_many(
                            final_order_data.to_dict("records")
                        )

                        print("Order Data updated")
                    current_position_data.to_csv(
                        "/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Orders_Data/"
                        + str(user_id)
                        + "/Updated_Targets/IV_Based_"
                        + str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d"))
                        + ".csv"
                    )
        except Exception as e:
            print(str(e))
else:
    print("Market is closed")

end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print("Duration: {}".format(end_time - start_time))
