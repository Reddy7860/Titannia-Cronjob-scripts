import pandas as pd
import robin_stocks as rs
from datetime import timedelta
import pytz
import yfinance as yf
from pymongo import MongoClient
from pandasql import sqldf

import uuid
import numpy as np
from datetime import datetime
from pytz import timezone

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# get the current date and time in the UTC timezone
now_utc = datetime.now(pytz.utc)

# convert to the desired timezone (e.g. Eastern Standard Time)
tz = pytz.timezone('US/Eastern')
now_est = now_utc.astimezone(tz)

# format the date as a string in the desired format
today = now_est.strftime("%Y-%m-%d")

print(today)

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/United_States_Titania_Trading?ssl=true&ssl_cert_reqs=CERT_NONE")
db = client.United_States_Titania_Trading
collection = db['final_orders_raw_data']


orders_df = collection.find({"execution_date": today})
# orders_df = collection.find({"execution_date":"2023-03-16"})
orders_df = pd.DataFrame(list(orders_df))


if len(orders_df) > 0:
    # Create orders dataframe
    orders_df = orders_df[['Strategy','Stock','Datetime','Signal','Value','Target','StopLoss','buy_probability','sell_probability','Strike_Buy_Price','premium_StopLoss','premium_Target','current_script','token','conclusion']]
    orders_df['Quantity'] = np.ones(len(orders_df))
    orders_df['Trigger_Price'] = 0
    orders_df['Product_Type'] = 'DELIVERY'
    orders_df['Type'] = 'LIMIT'
    orders_df['Validity'] = 'DAY'
    orders_df['Buy_timestamp'] = ''
    orders_df['Sell_timestamp'] = ''
    orders_df['final_timestamp'] = ''

    # Fetching data from MongoDB
    orders_place_data = db["us_paper_trading_orders_place"].find({"execution_date":today})
    # # orders_place_data = db["us_paper_trading_orders_place"].find({"execution_date": "2023-03-16"})

    # paper_trading_orders_place = db["us_paper_trading_orders_place"]

    # Converting MongoDB cursor to Dataframe
    orders_place_data = pd.DataFrame(list(orders_place_data))

    print(orders_place_data.head())
    # print(orders_place_data)

    print(orders_place_data.columns)

    # Setting the columns of the Dataframe
    if len(orders_place_data) > 0:
        if 'target_order_id' in orders_place_data.columns and 'stoploss_order_id' in orders_place_data.columns:
            print("target_order_id in index")
            orders_place_data = orders_place_data[['Strategy', 'Stock', 'Datetime', 'Signal', 'Value','Target','StopLoss', 'buy_probability', 'sell_probability', 'Strike_Buy_Price', 'premium_StopLoss', 'premium_Target', 'current_script', 'token', 'conclusion', 'Quantity', 'Trigger_Price', 'Type', 'Product_Type', 'Validity', 'execution_date', 'order_id', 'target_order_id', 'stoploss_order_id', 'final_order_id', 'Buy_timestamp', 'Sell_timestamp', 'final_timestamp']]
        else:
            orders_place_data['target_order_id'] = ""
            orders_place_data['stoploss_order_id'] = ""
            orders_place_data = orders_place_data[['Strategy', 'Stock', 'Datetime', 'Signal', 'Value','Target','StopLoss', 'buy_probability', 'sell_probability', 'Strike_Buy_Price', 'premium_StopLoss', 'premium_Target', 'current_script', 'token', 'conclusion', 'Quantity', 'Trigger_Price', 'Type', 'Product_Type', 'Validity', 'execution_date', 'order_id', 'target_order_id', 'stoploss_order_id', 'final_order_id', 'Buy_timestamp', 'Sell_timestamp', 'final_timestamp']]
    else:
        orders_place_data = pd.DataFrame(columns = ['Strategy', 'Stock', 'Datetime', 'Signal', 'Value','Target','StopLoss','buy_probability', 'sell_probability', 'Strike_Buy_Price','premium_StopLoss', 'premium_Target', 'current_script', 'token','conclusion', 'Quantity', 'Trigger_Price','Type', 'Product_Type','Validity', 'execution_date','order_id','target_order_id','stoploss_order_id','final_order_id','Buy_timestamp','Sell_timestamp','final_timestamp'])

    # Changing datatype of 'Datetime' column to datetime
    orders_place_data["Datetime"] = pd.to_datetime(orders_place_data["Datetime"])

    # orders_place_data

    pysqldf = lambda q: sqldf(q, globals())

    orders_df.drop_duplicates(inplace=True)   
    orders_place_data.drop_duplicates(inplace=True)   

    final_raw_ordes_data = pysqldf(""" select * from
            (
            select *,row_number()over(partition by Strategy,Stock,Datetime order by Datetime ) as row_num from orders_df
            ) where row_num = 1

            """)

    # print(orders_df)



    orders_df['Datetime'] = pd.to_datetime(orders_df['Datetime'])
    orders_place_data['Datetime'] = pd.to_datetime(orders_place_data['Datetime'])

    # Subtract 4 hours from Datetime where conclusion is Buy Successful
    orders_place_data.loc[orders_place_data['conclusion'] == 'Buy Successful', 'Datetime'] = pd.to_datetime(orders_place_data.loc[orders_place_data['conclusion'] == 'Buy Successful', 'Datetime']) - pd.Timedelta(hours=4)
    orders_place_data.loc[orders_place_data['conclusion'].isin(['Stoploss Placed', 'Target Placed']), 'Datetime'] = pd.to_datetime(orders_place_data.loc[orders_place_data['conclusion'].isin(['Stoploss Placed', 'Target Placed']), 'Datetime']) - pd.Timedelta(hours=4)


    print(orders_df['Datetime'].min())
    print(orders_place_data['Datetime'].min())
    print(orders_df.head())
    print(orders_place_data.head())



    final_raw_ordes_data = pysqldf("""
                                select t1.Strategy,
                                        t1.Stock,
                                        t1.Datetime,
                                        t1.Signal,
                                        t1.Value,
                                        t1.Target,
                                        t1.StopLoss,
                                        t1.buy_probability,
                                        t1.sell_probability,
                                        t1.Strike_Buy_Price,
                                        t1.premium_StopLoss,
                                        t1.premium_Target,
                                        t1.current_script,
                                        t1.token,
                                        COALESCE(t2.conclusion,t1.conclusion) as conclusion,
                                        t1.Quantity,
                                        t1.Trigger_Price,
                                        COALESCE(t2.Product_Type,t1.Product_Type) as Product_Type,
                                        COALESCE(t2.order_id,'') as order_id,
                                        COALESCE(t2.Type,'LIMIT') as Type,
                                        COALESCE(t2.Validity,'DAY') as Validity,
                                        COALESCE(t2.target_order_id,'') as target_order_id,
                                        COALESCE(t2.stoploss_order_id,'') as stoploss_order_id,
                                        COALESCE(t2.final_order_id,'') as final_order_id,
                                        COALESCE(t2.Buy_timestamp,t1.Buy_timestamp) as Buy_timestamp,
                                        COALESCE(t2.Sell_timestamp,t1.Sell_timestamp) as Sell_timestamp,
                                        COALESCE(t2.final_timestamp,t1.final_timestamp) as final_timestamp


                                from orders_df t1 
                                left join orders_place_data t2 on t1.Strategy = t2.Strategy and t1.Stock = t2.Stock and t1.Datetime = t2.Datetime
                                """)

    print(final_raw_ordes_data.head())  

    print(orders_df.head())
    print(orders_place_data.head())

    print(len(final_raw_ordes_data[final_raw_ordes_data['conclusion'] == "Buy Successful"]))
    print(len(final_raw_ordes_data[final_raw_ordes_data['conclusion'] == "Stoploss Placed"]))
    print(len(final_raw_ordes_data[final_raw_ordes_data['conclusion'] == "Target Placed"]))


    if len(final_raw_ordes_data) > 0:

        for idx in range(0,len(final_raw_ordes_data)):

    #         print(final_raw_ordes_data.loc[idx,])

            ## Fetch data from Big query and merge both. First preference is for Big query and later the orders_df

            if final_raw_ordes_data.loc[idx,"conclusion"] == "New Order":

                print("This is latest order")       
                final_raw_ordes_data.loc[idx,'conclusion'] = "Limit Order Placed"
                final_raw_ordes_data.loc[idx,'order_id'] = final_raw_ordes_data.loc[idx,'final_order_id']= str(uuid.uuid4())

            else:
                print("other")

    #             symbol = final_raw_ordes_data.loc[idx,"Stock"]
    #             interval = '5minute'  # Change interval to minute
    #             span = 'day'
    #             bounds = 'regular'

    #             data = rs.robinhood.stocks.get_stock_historicals(symbol, interval=interval, span=span, bounds=bounds, info=None)

    #             hist_df = pd.DataFrame(data)
    #             hist_df['begins_at'] = pd.to_datetime(hist_df['begins_at'],format='%Y-%m-%dT%H:%M:%SZ') - timedelta(hours=4)
    #             hist_df.drop(columns=['session', 'interpolated'], inplace=True)
    #             hist_df.rename(columns={'begins_at':'Datetime','open_price': 'Open', 'close_price': 'Close', 'high_price': 'High', 'low_price': 'Low', 'volume': 'Volume'}, inplace=True)

                symbol = final_raw_ordes_data.loc[idx, 'Stock']
                ticker = yf.Ticker(symbol)

                start_time = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')
                end_time = pd.Timestamp.now(tz='US/Eastern').strftime('%Y-%m-%d')+ ' 16:00:00'

                hist_df = ticker.history(start=start_time, end=pd.Timestamp(end_time, tz='US/Eastern'), interval='1m')

                hist_df.rename(columns={'Open': 'Open', 'Close': 'Close', 'High': 'High', 'Low': 'Low', 'Volume': 'Volume'}, inplace=True)
                hist_df.index.name = 'Datetime'
                hist_df.reset_index(inplace=True)

                hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'])
                final_raw_ordes_data['Datetime'] = pd.to_datetime(final_raw_ordes_data['Datetime'])
                try:
                    final_raw_ordes_data['Datetime'] = final_raw_ordes_data['Datetime'].dt.tz_convert('America/New_York')
                except Exception as e:
                    final_raw_ordes_data['Datetime'] = final_raw_ordes_data['Datetime'].dt.tz_localize('America/New_York')

    #             print(hist_df)

                if str(final_raw_ordes_data.loc[idx,'order_id']) == str(final_raw_ordes_data.loc[idx,'final_order_id']):
                    if final_raw_ordes_data.loc[idx,'conclusion'] == "Limit Order Placed":
                        print("Place Target or Stoploss Order")
                        filter_df = hist_df[hist_df['Datetime'] > final_raw_ordes_data.loc[idx,"Datetime"] + pd.Timedelta(minutes=5) ]

                        filter_df.reset_index(inplace=True,drop=True)

            #             Check with the options data

                        ## Place the limit order


                        buy_price = final_raw_ordes_data.loc[idx,"Value"]
    #                     print(buy_price)

                        ## Check if the limit order has been placed 
                        for row in range(0,len(filter_df)):



    #                         print(final_raw_ordes_data.loc[idx,"Signal"])
    #                         print(float(filter_df.loc[row,"Low"]))
    #                         print(float(buy_price))
    #                         print("========")

                            if ((final_raw_ordes_data.loc[idx,"Signal"] == "Buy" and float(filter_df.loc[row,"Low"]) < float(buy_price)) or
                                (final_raw_ordes_data.loc[idx,"Signal"] == "Sell" and float(filter_df.loc[row,"High"]) > float(buy_price))):

    #                         if float(filter_df.loc[row,"Low"]) < float(buy_price):

                                # Convert to datetime objects in the Asia/Kolkata timezone
                                date_timestamp = pd.to_datetime(final_raw_ordes_data.loc[idx,'Datetime'])
                                timestamp1 = date_timestamp.tz_convert('America/New_York')

                                # Convert to datetime objects in the America/New_York timezone
                                timestamp2 = pd.to_datetime(filter_df.loc[row,'Datetime']).tz_convert('America/New_York')

                                # Calculate time delta
                                time_delta = timestamp2 - timestamp1

    #                             print(final_raw_ordes_data.loc[idx,'Datetime'])
    #                             print(filter_df.loc[idx,'Datetime'])

    #                             date_timestamp = pd.to_datetime(final_raw_ordes_data.loc[idx,'Datetime'])

    #                             timestamp1 = date_timestamp.tz_localize('America/New_York')

    #                             time_delta =  filter_df.loc[row,'Datetime'] - timestamp1
                                time_delta_mins = time_delta.total_seconds() / 60

                                print(time_delta_mins)

                                if time_delta_mins >= 30:
                                    final_raw_ordes_data.loc[idx,'conclusion'] = "Cancelled Order as time passed"
                                    final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                    break
                                else:
                                    final_raw_ordes_data.loc[idx,'conclusion'] = "Buy Successful"
                                    final_raw_ordes_data.loc[idx,'Buy_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                    break

                    elif final_raw_ordes_data.loc[idx,'conclusion'] == "Buy Successful":

                        print(final_raw_ordes_data.loc[idx,"Buy_timestamp"])

                        filter_df = hist_df[hist_df['Datetime'] > final_raw_ordes_data.loc[idx,"Buy_timestamp"]]

                        filter_df.reset_index(inplace=True,drop=True)

    #                     if final_orders_data.loc[idx,'final_order_id'] == "":

                        print("No Target or Stoploss placed")

    #                     latest_close = filter_df.loc[len(filter_df)-1,'Close']
    #                     latest_time = filter_df.loc[len(filter_df)-1,'Datetime']

                        latest_close = filter_df.loc[0,'Close']
                        latest_time = filter_df.loc[0,'Datetime']

                        if float(latest_close) >= float(final_raw_ordes_data.loc[idx,'Value']):

                            final_raw_ordes_data.loc[idx,'conclusion'] = "Target Placed"
                            final_raw_ordes_data.loc[idx,'target_order_id'] = final_raw_ordes_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                            final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(latest_time))

                        else:
                            final_raw_ordes_data.loc[idx,'conclusion'] = "Stoploss Placed"
                            final_raw_ordes_data.loc[idx,'stoploss_order_id'] = final_raw_ordes_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                            final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(latest_time))

                elif ((str(final_raw_ordes_data.loc[idx,'target_order_id']) == str(final_raw_ordes_data.loc[idx,'final_order_id'])) or (str(final_raw_ordes_data.loc[idx,'stoploss_order_id']) == str(final_raw_ordes_data.loc[idx,'final_order_id']))):

                    if final_raw_ordes_data.loc[idx,'conclusion'] == "Order Completed":
                        print("Order Complete")
                    else:
                        print(final_raw_ordes_data.loc[idx,"Buy_timestamp"])
                        filter_df = hist_df[hist_df['Datetime'] > final_raw_ordes_data.loc[idx,"Buy_timestamp"]]

                        filter_df.reset_index(inplace=True,drop=True)
                        for row in range(0,len(filter_df)):

                            print(row)
                            print(filter_df.loc[row,])
                            print(final_raw_ordes_data.loc[idx,'conclusion'])

                            current_low = filter_df.loc[row,'Low']
                            current_high = filter_df.loc[row,'High']
                            current_close = filter_df.loc[row,'Close']

    #                         print(final_raw_ordes_data.loc[idx,"Value"])
    #                         print(current_close)

                            price_diff = (float(final_raw_ordes_data.loc[idx,"Value"]) - float(current_close))*100 / float(final_raw_ordes_data.loc[idx,"Value"])

                            print(price_diff)

                            if final_raw_ordes_data.loc[idx,'conclusion'] == "Target Placed":
                                if ((final_raw_ordes_data.loc[idx,"Signal"] == "Buy" and price_diff > 0) or (final_raw_ordes_data.loc[idx,"Signal"] == "Sell" and price_diff < 0)):
                                    print("In Loss")
                                    final_raw_ordes_data.loc[idx,'conclusion'] = "Stoploss Placed"
                                    final_raw_ordes_data.loc[idx,'stoploss_order_id'] = final_raw_ordes_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                                    final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                else:
                                    if ((final_raw_ordes_data.loc[idx,"Signal"] == "Sell") and (float(current_low) <=  float(final_raw_ordes_data.loc[idx,"Target"]))):
    #                                     print("Called Sell from target")
    #                                 if float(filter_df.loc[row,'High']) >= float(final_raw_ordes_data.loc[idx,"Target"]):
                                        final_raw_ordes_data.loc[idx,'conclusion'] = "Order Completed"
                                        final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                        break
                                    elif((final_raw_ordes_data.loc[idx,"Signal"] == "Buy") and (float(current_high) >=  float(final_raw_ordes_data.loc[idx,"Target"]))):
    #                                     print("Called buy from target")
    #                                 if float(filter_df.loc[row,'High']) >= float(final_raw_ordes_data.loc[idx,"Target"]):
                                        final_raw_ordes_data.loc[idx,'conclusion'] = "Order Completed"
                                        final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                        break
                                    else:
                                        continue

                            elif final_raw_ordes_data.loc[idx,'conclusion'] == "Stoploss Placed":
                                if ((final_raw_ordes_data.loc[idx,"Signal"] == "Buy" and price_diff < 0) or (final_raw_ordes_data.loc[idx,"Signal"] == "Sell" and price_diff > 0)):
                                    print("In Profit")
                                    final_raw_ordes_data.loc[idx,'conclusion'] = "Target Placed"
                                    final_raw_ordes_data.loc[idx,'target_order_id'] = final_raw_ordes_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                                    final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                else:
                                    if (final_raw_ordes_data.loc[idx,"Signal"] == "Sell" and float(current_high) >=  float(final_raw_ordes_data.loc[idx,"StopLoss"])):
                                        print(final_raw_ordes_data.loc[idx,])
    #                                     print("Called Sell from stoploss")
    #                                 if float(filter_df.loc[row,'Low']) <= float(final_raw_ordes_data.loc[idx,"StopLoss"]):
                                        final_raw_ordes_data.loc[idx,'conclusion'] = "Order Completed"
                                        final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                        break
                                    elif (final_raw_ordes_data.loc[idx,"Signal"] == "Buy" and float(current_low) <=  float(final_raw_ordes_data.loc[idx,"StopLoss"])):
    #                                     print("Called buy from stoploss")
                                        final_raw_ordes_data.loc[idx,'conclusion'] = "Order Completed"
                                        final_raw_ordes_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                    else:
                                        continue


        final_raw_ordes_data.drop_duplicates(inplace=True)   
        print(final_raw_ordes_data)     

        paper_trading_orders_place = db["us_paper_trading_orders_place"]

        final_raw_ordes_data['execution_date'] = datetime.now(timezone("America/New_York")).strftime("%Y-%m-%d")

        ## Delete the existing orders pushed for today
        x = paper_trading_orders_place.delete_many(
            {"execution_date": (datetime.now(timezone("America/New_York")) ).strftime("%Y-%m-%d")}
        )

        print(x.deleted_count, " documents deleted.")

        paper_trading_orders_place.insert_many(final_raw_ordes_data.to_dict("records"))
            



