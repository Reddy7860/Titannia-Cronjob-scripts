import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime
from pytz import timezone
from pandas.io import gbq
import pandas_gbq
from pymongo import MongoClient
from smartapi import SmartConnect
import pyotp
from pandasql import sqldf
import time
import uuid
from google.oauth2 import service_account

warnings.filterwarnings('ignore')

today = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")
# today = datetime.now().strftime("%Y-%m-%d")
# today = "2023-01-27"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Connect to MongoDB
client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE")
db = client.titania_trading

# credentials = service_account.Credentials.from_service_account_file(
#     '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/client_secret.json',
# )
credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/ferrous-module-376519-7e08f583402d.json',
)

# Download and preprocess data
data = yf.download(tickers="%5ENSEI", period="5d", interval="5m")
data.reset_index(inplace=True)
data.rename(columns={'index':'Datetime'}, inplace=True)
latest_data = data[data['Datetime'] >= today][['Datetime','Open','High','Low','Close']]
latest_data.reset_index(inplace=True,drop=True)

# Generate session
obj = SmartConnect(api_key="Whpk7Rvg")
totp = pyotp.TOTP("2JN6BREQ5NXE2UHEHIFCVEKOWQ")
print("pyotp", totp.now())
attempts = 5
while attempts > 0:
    attempts -= 1
    data = obj.generateSession("M591295", "0808", totp.now())
    if data["status"]:
        break
    time.sleep(2)

# Read data from Bigquery
# sql = "select * FROM `reddy000-c898c.Titania_Dataset.candlestick_signals` where execution_date = '2023-01-27'"
sql = "select * FROM `ferrous-module-376519.Titania.new_candle_stick_signals` where execution_date = '"+str(today) + "'"
print(sql)
temp_df = pandas_gbq.read_gbq(sql, project_id='ferrous-module-376519',credentials=credentials)



# Create orders dataframe
orders_df = temp_df[['Strategy','Stock','Datetime','Signal','Value','buy_probability','sell_probability','Strike_Buy_Price','premium_StopLoss','premium_Target','current_script','token','conclusion']]
orders_df['Quantity'] = np.ones(len(orders_df))
orders_df['Trigger_Price'] = 0
orders_df['Product_Type'] = 'DELIVERY'
orders_df['Type'] = 'LIMIT'
orders_df['Validity'] = 'DAY'
orders_df['Buy_timestamp'] = ''
orders_df['Sell_timestamp'] = ''
orders_df['final_timestamp'] = ''





# Fetching data from MongoDB
orders_place_data = (db["paper_trading_orders_place"].find({"execution_date": (datetime.now(timezone("Asia/Kolkata")) ).strftime("%Y-%m-%d")}))

# Converting MongoDB cursor to Dataframe
orders_place_data = pd.DataFrame(list(orders_place_data))

# Setting the columns of the Dataframe
if len(orders_place_data) > 0:
    orders_place_data = orders_place_data[['Strategy', 'Stock', 'Datetime', 'Signal', 'Value','buy_probability', 'sell_probability', 'Strike_Buy_Price','premium_StopLoss', 'premium_Target', 'current_script', 'token','conclusion', 'Quantity', 'Trigger_Price', 'Type','Product_Type','Validity', 'execution_date','order_id','target_order_id','stoploss_order_id','final_order_id','Buy_timestamp','Sell_timestamp','final_timestamp']]
else:
    orders_place_data = pd.DataFrame(columns = ['Strategy', 'Stock', 'Datetime', 'Signal', 'Value','buy_probability', 'sell_probability', 'Strike_Buy_Price','premium_StopLoss', 'premium_Target', 'current_script', 'token','conclusion', 'Quantity', 'Trigger_Price','Type', 'Product_Type','Validity', 'execution_date','order_id','target_order_id','stoploss_order_id','final_order_id','Buy_timestamp','Sell_timestamp','final_timestamp'])

# Changing datatype of 'Datetime' column to datetime
orders_place_data["Datetime"] = pd.to_datetime(orders_place_data["Datetime"])

pysqldf = lambda q: sqldf(q, globals())

orders_df.drop_duplicates(inplace=True)   
orders_place_data.drop_duplicates(inplace=True)   

orders_df = pysqldf(""" select * from
        (
        select *,row_number()over(partition by Strategy,Stock,Datetime order by Datetime ) as row_num from orders_df
        ) where row_num = 1
        
        """)



final_orders_data = pysqldf("""
                            select t1.Strategy,
                                    t1.Stock,
                                    t1.Datetime,
                                    t1.Signal,
                                    t1.Value,
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
                            left join orders_place_data t2 on t1.Strategy = t2.Strategy and t1.current_script = t2.current_script and t1.Datetime = t2.Datetime
                            """)


if len(final_orders_data) > 0:

    for idx in range(0,len(final_orders_data)):
        
        print(final_orders_data.loc[idx,])
        
        ## Fetch data from Big query and merge both. First preference is for Big query and later the orders_df
        
        if final_orders_data.loc[idx,"conclusion"] == "New Order":
            
            print("This is latest order")       
            final_orders_data.loc[idx,'conclusion'] = "Limit Order Placed"
            final_orders_data.loc[idx,'order_id'] = final_orders_data.loc[idx,'final_order_id']= str(uuid.uuid4())
            
        else:
            
            print(final_orders_data.loc[idx,"token"])
            hist = {
                "exchange": "NFO",
                "symboltoken": str(final_orders_data.loc[idx,"token"]),
                "interval": "ONE_MINUTE",
                "fromdate": str(today) + " 09:15",
                "todate": str(today) + " 15:30",
            }

            resp = obj.getCandleData(hist)
            time.sleep(0.5)
            # print(resp)
            hist_df = pd.DataFrame.from_dict(resp["data"])

            hist_df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]

            hist_df["Datetime"] = pd.to_datetime(
                hist_df["Datetime"], format="%Y-%m-%d %H:%M:%S"
            )

            # print(hist_df)
            
            if str(final_orders_data.loc[idx,'order_id']) == str(final_orders_data.loc[idx,'final_order_id']):
                
                if final_orders_data.loc[idx,'conclusion'] == "Limit Order Placed":

                    print("Place Target or Stoploss Order")

                    filter_df = hist_df[hist_df['Datetime'] > final_orders_data.loc[idx,"Datetime"]]

                    filter_df.reset_index(inplace=True,drop=True)

        #             Check with the options data

                    ## Place the limit order


                    buy_price = final_orders_data.loc[idx,"Strike_Buy_Price"]

                    ## Check if the limit order has been placed 
                    for row in range(0,len(filter_df)):

                        if filter_df.loc[row,"Low"] < int(float(buy_price)):

        #                     print(final_orders_data.loc[idx,'Datetime'])
        #                     print(filter_df.loc[idx,'Datetime'])

                            date_timestamp = pd.to_datetime(final_orders_data.loc[idx,'Datetime'])

                            timestamp1 = date_timestamp.tz_localize('Asia/Kolkata')

                            time_delta =  filter_df.loc[row,'Datetime'] - timestamp1
                            time_delta_mins = time_delta.total_seconds() / 60

                            print(time_delta_mins)

                            if time_delta_mins >= 30:
                                final_orders_data.loc[idx,'conclusion'] = "Cancelled Order as time passed"
                                final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                break
                            else:
                                final_orders_data.loc[idx,'conclusion'] = "Buy Successful"
                                final_orders_data.loc[idx,'Buy_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                break

                elif final_orders_data.loc[idx,'conclusion'] == "Buy Successful":

                    filter_df = hist_df[hist_df['Datetime'] > final_orders_data.loc[idx,"Buy_timestamp"]]

                    filter_df.reset_index(inplace=True,drop=True)

#                     if final_orders_data.loc[idx,'final_order_id'] == "":

                    print("No Target or Stoploss placed")

#                     latest_close = filter_df.loc[len(filter_df)-1,'Close']
#                     latest_time = filter_df.loc[len(filter_df)-1,'Datetime']
            
                    latest_close = filter_df.loc[0,'Close']
                    latest_time = filter_df.loc[0,'Datetime']

                    if int(latest_close) >= int(float(final_orders_data.loc[idx,'Strike_Buy_Price'])):

                        final_orders_data.loc[idx,'conclusion'] = "Target Placed"
                        final_orders_data.loc[idx,'target_order_id'] = final_orders_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                        final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(latest_time))

                    else:
                        final_orders_data.loc[idx,'conclusion'] = "Stoploss Placed"
                        final_orders_data.loc[idx,'stoploss_order_id'] = final_orders_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                        final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(latest_time))
                        
            elif ((str(final_orders_data.loc[idx,'target_order_id']) == str(final_orders_data.loc[idx,'final_order_id'])) or (str(final_orders_data.loc[idx,'stoploss_order_id']) == str(final_orders_data.loc[idx,'final_order_id']))):
                
                if final_orders_data.loc[idx,'conclusion'] == "Order Completed":
                    print("Order Complete")
                else:
                    filter_df = hist_df[hist_df['Datetime'] > final_orders_data.loc[idx,"Buy_timestamp"]]

                    filter_df.reset_index(inplace=True,drop=True)
                    for row in range(0,len(filter_df)):
                        
                        current_low = filter_df.loc[row,'Low']
                        current_high = filter_df.loc[row,'High']
                        current_close = filter_df.loc[row,'Close']
                        
                        price_diff = (float(final_orders_data.loc[idx,"Strike_Buy_Price"]) - float(current_close))*100 / float(final_orders_data.loc[idx,"Strike_Buy_Price"])
                        
                        # print(price_diff)
                        
                        if final_orders_data.loc[idx,'conclusion'] == "Target Placed":
                            if price_diff > 0 :
                                print("In Loss")
                                final_orders_data.loc[idx,'conclusion'] = "Stoploss Placed"
                                final_orders_data.loc[idx,'stoploss_order_id'] = final_orders_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                                final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                            else:
                                if int(float(filter_df.loc[row,'High'])) >= int(float(final_orders_data.loc[idx,"premium_Target"])):
                                    final_orders_data.loc[idx,'conclusion'] = "Order Completed"
                                    final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                    break
                                else:
                                    continue
                                
                        elif final_orders_data.loc[idx,'conclusion'] == "Stoploss Placed":
                            if price_diff < 0 :
                                print("In Profit")
                                final_orders_data.loc[idx,'conclusion'] = "Target Placed"
                                final_orders_data.loc[idx,'target_order_id'] = final_orders_data.loc[idx,'final_order_id'] = str(uuid.uuid4())
                                final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                            else:
                                if int(float(filter_df.loc[row,'Low'])) <= int(float(final_orders_data.loc[idx,"premium_StopLoss"])):
                                    final_orders_data.loc[idx,'conclusion'] = "Order Completed"
                                    final_orders_data.loc[idx,'final_timestamp'] = str(pd.to_datetime(filter_df.loc[row,'Datetime']))
                                    break
                                else:
                                    continue
                            
  
    final_orders_data.drop_duplicates(inplace=True)   
    print(final_orders_data)     

    paper_trading_orders_place = db["paper_trading_orders_place"]

    final_orders_data['execution_date'] = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")

    ## Delete the existing orders pushed for today
    x = paper_trading_orders_place.delete_many(
        {"execution_date": (datetime.now(timezone("Asia/Kolkata")) ).strftime("%Y-%m-%d")}
    )

    print(x.deleted_count, " documents deleted.")

    paper_trading_orders_place.insert_many(final_orders_data.to_dict("records"))

