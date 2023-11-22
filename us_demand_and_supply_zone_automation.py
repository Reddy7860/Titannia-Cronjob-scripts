import pandas as pd
import os
import yfinance as yf
from pandas.tseries.offsets import BDay
from pymongo import MongoClient
from datetime import datetime, timedelta
from pytz import timezone 

start_time = datetime.now()

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/United_States_Titania_Trading?ssl=true&ssl_cert_reqs=CERT_NONE")
db = client.United_States_Titania_Trading
collection = db['demand_and_supply_zones_daily']


def supply_zone_detection(df,stock,df_supply_and_demand):
    for ind in range(1, df.shape[0]-1):
        if (df.iloc[ind-1]['Open'] < df.iloc[ind-1]['Close'] and # Green Candle
            (abs(df.iloc[ind-1]['Open'] - df.iloc[ind-1]['Close']) > 0.5 * (df.iloc[ind-1]['High'] - df.iloc[ind-1]['Low'])) and # Im Balance Candle
            (abs(df.iloc[ind]['Open'] - df.iloc[ind]['Close']) <= 0.3 * (df.iloc[ind]['High'] - df.iloc[ind]['Low'])) and
            (df.iloc[ind+1]['Open'] > df.iloc[ind+1]['Close']) and # Red Candle
            (abs(df.iloc[ind+1]['Open'] - df.iloc[ind+1]['Close']) > 0.5 * (df.iloc[ind+1]['High'] - df.iloc[ind+1]['Low']))
           ):

            df_supply_and_demand.loc[ind,'stock'] = stock
            df_supply_and_demand.loc[ind,'pattern'] = "Supply Reversal Pattern(R-B-D)"
            df_supply_and_demand.loc[ind,'Date'] = df.iloc[ind]['Date']
            df_supply_and_demand.loc[ind,'zone_1'] = round(df.iloc[ind]['Low'],2) 
            # df_supply_and_demand.loc[ind,'zone_2'] = round(max(df.iloc[ind]['Open'],df.iloc[ind]['Close']),2)
            df_supply_and_demand.loc[ind,'zone_2'] = round(df.iloc[ind]['High'],2)

            if(df.iloc[ind+1]['Open'] > df.iloc[ind]['Open']):
                df_supply_and_demand.loc[ind,'strength'] = "Strong"
            else:
                df_supply_and_demand.loc[ind,'strength'] = "Normal"

        elif ((df.iloc[ind-1]['Open'] > df.iloc[ind-1]['Close']) and # Red Candle
              (abs(df.iloc[ind-1]['Open'] - df.iloc[ind-1]['Close']) > 0.5*(df.iloc[ind-1]['High'] - df.iloc[ind-1]['Low'])) and # Im-Balance Candle
              (abs(df.iloc[ind]['Open'] - df.iloc[ind]['Close']) <= 0.3*(df.iloc[ind]['High'] - df.iloc[ind]['Low'])) and
              (df.iloc[ind+1]['Open'] > df.iloc[ind+1]['Close']) and # Red Candle
              (abs(df.iloc[ind+1]['Open'] - df.iloc[ind+1]['Close']) > 0.5*(df.iloc[ind+1]['High'] - df.iloc[ind+1]['Low']))  # Im-Balance Candle
             ):

            df_supply_and_demand.loc[ind,'stock'] = stock
            df_supply_and_demand.loc[ind,'pattern'] = "Supply Continuous Pattern(D-B-D)"
            df_supply_and_demand.loc[ind,'Date'] = df.iloc[ind]['Date']
            df_supply_and_demand.loc[ind,'zone_1'] = round(df.iloc[ind]['Low'],2)
            # df_supply_and_demand.loc[ind,'zone_2'] = round(max(df.iloc[ind]['Open'],df.iloc[ind]['Close']),2)
            df_supply_and_demand.loc[ind,'zone_2'] = round(df.iloc[ind]['High'],2)

            if(df.iloc[ind+1]['Open'] < df.iloc[ind]['Open']):
                df_supply_and_demand.loc[ind,'strength'] = "Strong"
            else:
                df_supply_and_demand.loc[ind,'strength'] = "Normal"
                
    return df_supply_and_demand

def demand_zone_detection(df,stock,df_supply_and_demand):
    for ind in range(1, df.shape[0]-1):
        if ((df.iloc[ind-1]['Open'] > df.iloc[ind-1]['Close']) and
            (abs(df.iloc[ind-1]['Open'] - df.iloc[ind-1]['Close']) > 0.5 * (df.iloc[ind-1]['High'] - df.iloc[ind-1]['Low'])) and
            (abs(df.iloc[ind]['Open'] - df.iloc[ind]['Close']) <= 0.3 * (df.iloc[ind]['High'] - df.iloc[ind]['Low'])) and
            (df.iloc[ind+1]['Open'] < df.iloc[ind+1]['Close']) and # Green Candle
            (abs(df.iloc[ind+1]['Open'] - df.iloc[ind+1]['Close']) > 0.5 * (df.iloc[ind+1]['High'] - df.iloc[ind+1]['Low']))
           ):

            df_supply_and_demand.loc[ind,'stock'] = stock
            df_supply_and_demand.loc[ind,'pattern'] = "Demand Reversal Pattern(D-B-R)"
            df_supply_and_demand.loc[ind,'Date'] = df.iloc[ind]['Date']
            df_supply_and_demand.loc[ind,'zone_1'] = round(df.iloc[ind]['High'],2)
            # df_supply_and_demand.loc[ind,'zone_2'] = round(min(df.iloc[ind]['Open'],df.iloc[ind]['Close']),2)
            df_supply_and_demand.loc[ind,'zone_2'] = round(df.iloc[ind]['Low'],2)

            if(df.iloc[ind+1]['Open'] > df.iloc[ind]['Open']):
                df_supply_and_demand.loc[ind,'strength'] = "Strong"
            else:
                df_supply_and_demand.loc[ind,'strength'] = "Normal"

        elif ((df.iloc[ind-1]['Open'] < df.iloc[ind-1]['Close']) and # Green Candle
              (abs(df.iloc[ind-1]['Open'] - df.iloc[ind-1]['Close']) > 0.5 * (df.iloc[ind-1]['High'] - df.iloc[ind-1]['Low'])) and # Im-Balance Candle
              (abs(df.iloc[ind]['Open'] - df.iloc[ind]['Close']) <= 0.3 * (df.iloc[ind]['High'] - df.iloc[ind]['Low'])) and
              (df.iloc[ind+1]['Open'] < df.iloc[ind+1]['Close']) and # Green Candle
              (abs(df.iloc[ind+1]['Open'] - df.iloc[ind+1]['Close']) > 0.5 * (df.iloc[ind+1]['High'] - df.iloc[ind+1]['Low']))  # Im-Balance Candle
             ):

            df_supply_and_demand.loc[ind,'stock'] = stock
            df_supply_and_demand.loc[ind,'pattern'] = "Demand Continuous Pattern(R-B-R)"
            df_supply_and_demand.loc[ind,'Date'] = df.iloc[ind]['Date']
            df_supply_and_demand.loc[ind,'zone_1'] = round(df.iloc[ind]['High'],2)
            # df_supply_and_demand.loc[ind,'zone_2'] = round(min(df.iloc[ind]['Open'],df.iloc[ind]['Close']),2)
            df_supply_and_demand.loc[ind,'zone_2'] = round(df.iloc[ind]['Low'],2)

            if(df.iloc[ind+1]['Open'] > df.iloc[ind]['Open']):
                df_supply_and_demand.loc[ind,'strength'] = "Strong"
            else:
                df_supply_and_demand.loc[ind,'strength'] = "Normal"
                

# Get the current date and time in the Asia/Kolkata timezone
now = datetime.now(timezone("Asia/Kolkata"))

# Subtract 4 months from the current date
four_months_ago = now - timedelta(days=30*4)

# Format the result as a string in the YYYY-MM-DD format
start_date = four_months_ago.strftime('%Y-%m-%d')
end_date = now.strftime('%Y-%m-%d')

df_supply_and_demand_final = pd.DataFrame(columns=["stock", "pattern", "strength", "Date", "zone_1", "zone_2"])
current_dir = os.getcwd()

print(current_dir)

# nifty_df = pd.read_csv("/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Nifty50_Stocks.csv")
nifty_df = pd.read_csv("/home/sjonnal3/Hate_Speech_Detection/Trading_Application/US_30_Stocks.csv")

for idx in range(0,len(nifty_df)):
    stock = nifty_df.loc[idx,"Yahoo Symbol"]
    print(stock)
    # Download the data from Yahoo Finance
    data = yf.download(stock, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    # Print the downloaded data
    print(data)

    df_supply_and_demand = pd.DataFrame(columns=["stock", "pattern", "strength", "Date", "zone_1", "zone_2"])

    supply_zone_df = supply_zone_detection(data,stock,df_supply_and_demand)

    if supply_zone_df is not None and len(supply_zone_df) > 0:
        df_supply_and_demand_final = pd.concat([df_supply_and_demand_final, supply_zone_df], axis=0, ignore_index=True)

    demand_zone_df = demand_zone_detection(data,stock,df_supply_and_demand)

    if demand_zone_df is not None and len(demand_zone_df) > 0:
        df_supply_and_demand_final = pd.concat([df_supply_and_demand_final, demand_zone_df], axis=0, ignore_index=True)

    if len(df_supply_and_demand_final) > 0 :
        print(df_supply_and_demand_final)
        df_supply_and_demand_final.reset_index(inplace=True,drop=True)


if len(df_supply_and_demand_final) > 0 :
    df_supply_and_demand_final["Voided_Time"] = ""
    df_supply_and_demand_final["Percentage Change"]= 0

for ind in range(df_supply_and_demand_final.shape[0]):
    stock = df_supply_and_demand_final.loc[ind, "stock"]
    call_date = df_supply_and_demand_final.loc[ind, "Date"]
    call_date = pd.to_datetime(call_date).date()
    print(stock)

    nextWorkingDay = (pd.to_datetime(call_date) + BDay(1)).date()
    nextWorkingDay = (pd.to_datetime(nextWorkingDay) + BDay(1)).date()

    try:
        if (nextWorkingDay - pd.to_datetime("today").date()).days <= -1:
            curr_stock_data = yf.download(stock, start=nextWorkingDay, end=pd.to_datetime("today")+pd.DateOffset(1))

            curr_stock_data = curr_stock_data.reset_index()

            max_zone = max(df_supply_and_demand_final.loc[ind, "zone_1"], df_supply_and_demand_final.loc[ind, "zone_2"])
            min_zone = min(df_supply_and_demand_final.loc[ind, "zone_1"], df_supply_and_demand_final.loc[ind, "zone_2"])

            df_supply_and_demand_final.loc[ind, "fit"] = "Active"


            current_close = curr_stock_data.tail(1)["Close"].values[0]

            if (df_supply_and_demand_final.loc[ind, "pattern"] == "Supply Reversal Pattern(R-B-D)") or (
                    df_supply_and_demand_final.loc[ind, "pattern"] == "Supply Continuous Pattern(D-B-D)"):

                df_supply_and_demand_final.loc[ind, "Percentage Change"] = round(
                    (((df_supply_and_demand_final.loc[ind, "zone_1"] - current_close) / df_supply_and_demand_final.loc[ind, "zone_1"]) * 100), 2)

                for row_ind in range(curr_stock_data.shape[0]):
                    if curr_stock_data.loc[row_ind, "Close"] > max_zone:
                        df_supply_and_demand_final.loc[ind, "fit"] = "Voided"
                        df_supply_and_demand_final.loc[ind, "Voided_Time"] = curr_stock_data.loc[row_ind, "Date"]
                        break

            else:
                df_supply_and_demand_final.loc[ind, "Percentage Change"] = round(
                    (((current_close - df_supply_and_demand_final.loc[ind, "zone_1"]) / df_supply_and_demand_final.loc[ind, "zone_1"]) * 100), 2)

                for row_ind in range(curr_stock_data.shape[0]):
                    if curr_stock_data.loc[row_ind, "Close"] < min_zone:
                        df_supply_and_demand_final.loc[ind, "fit"] = "Voided"
                        df_supply_and_demand_final.loc[ind, "Voided_Time"] = curr_stock_data.loc[row_ind, "Date"]
                        break

        else:
            df_supply_and_demand_final.loc[ind, "fit"] = "Active"

    except Exception as e:
        print("error for :")
        print(stock)

print(df_supply_and_demand_final)

# filtered_df = df_supply_and_demand_final[(df_supply_and_demand_final['strength'] == 'Strong') & 
#                                       (df_supply_and_demand_final['fit'] == 'Active') & 
#                                       (df_supply_and_demand_final['Voided_Time']=="")]

# voided_df = df_supply_and_demand_final[(df_supply_and_demand_final['Voided_Time']!="")]

# filtered_df = filtered_df.sort_values(by="Date", ascending=False)
# filtered_df.reset_index(inplace=True,drop=True)

# voided_df = voided_df.sort_values(by="Date", ascending=False)
# voided_df.reset_index(inplace=True,drop=True)

print(df_supply_and_demand_final)

df_supply_and_demand_final['Execution_date'] = datetime.now().strftime('%Y-%m-%d')

x = collection.delete_many({})
print(x.deleted_count, " documents deleted.")

collection.insert_many(df_supply_and_demand_final.to_dict('records'))

end_time = datetime.now()

print(end_time)

print('Duration: {}'.format(end_time - start_time))