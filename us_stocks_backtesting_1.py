import numpy as np
from datetime import datetime, timedelta
from pandasql import sqldf
import warnings
import yfinance as yf
import pandas as pd
from pymongo import MongoClient

# Ignore warnings
warnings.filterwarnings("ignore")

# CSCO, TSLA, AAPL, CAT, WMT -> titaniacluster.fwxe9u5.mongodb.net ## titaniatraders2@gmail.com
# GE, XOM, AXP, BA, DD, HD, CVX, GS -> cluster0.92qtk1m.mongodb.net ## titaniatraders3@gmail.com
# MCD, JPM, IBM, INTC, KO, MMM, JNJ -> cluster0.xm6ytet.mongodb.net ## titaniatraders4@gmail.com
# NKE, MRK, PG, TRV, UNH, MSFT , PFE -> cluster0.izbxvtf.mongodb.net ## titaniatraders5@gmail.com
# V -> cluster0.igqnlsy.mongodb.net ##titaniatraders6@gmail.com

#titaniatraders7@gmail.com -> mongodb storage space available
#titaniacluster.lbbopxc.mongodb.net -> titaniatraders8@gmail.com
#titania-cluster.oak6unc.mongodb.net -> titaniatraders1@gmail.com

# MongoDB connection details
username = 'Titania'
password = 'Mahadev'
cluster_url = 'cluster0.igqnlsy.mongodb.net'
database_name = 'stocks_5_mins_data'
collection_name = 'United_States_Market'
# MongoDB connection URI
uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority"
# Connect to the MongoDB client
client = MongoClient(uri)
# Access the database and collection
db = client[database_name]
collection = db[collection_name]

stock = ticker = 'V'

# Set the start and end dates for the data
start_date = "2008-01-01"
end_date = "2023-05-30"


# Fetch the data for stock = 'TSLA' and Date from 2021-01-01 to 2023-05-30
query = {'stock': stock, 'Date': {'$gte': start_date, '$lte': end_date}}
# query = { 'Date': {'$gte': '2008-01-01', '$lte': '2023-05-30'}}
data = collection.find(query)
# Convert the data to a pandas DataFrame and show the head of the data
import pandas as pd
df = pd.DataFrame(list(data))


df = df[['DateTime','Open','High','Low','Close','Volume','stock']]



# Define the ticker symbol for TSLA
# ticker = "TSLA"



# Download the data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Convert the data to a pandas DataFrame
stock_data = pd.DataFrame(data)

# Reset the index and convert "Date" to a regular column
stock_data.reset_index(inplace=True)

# Print the DataFrame
# print(stock_data)



# Cowboy strategy

print("Running Cowboy")

stock_data["Rider_Bullish"] = "No"
stock_data["Bullish_Level"] = 100000
stock_data["Rider_Bearish"] = "No"
stock_data["Bearish_Level"] = 0
stock_data["Max_High"] = 0
stock_data["Max_Low"] = 0

for i in range(3, len(stock_data)):
    if abs((stock_data.loc[i, "High"] - stock_data.loc[i-1, "High"]) / stock_data.loc[i-1, "High"] * 100) < 0.5:
        stock_data.loc[i, "Rider_Bullish"] = "Yes"
        stock_data.loc[i, "Bullish_Level"] = max(stock_data.loc[i, "High"], stock_data.loc[i-1, "High"])
        stock_data.loc[i, "Max_High"] = max(stock_data.loc[i, "High"], stock_data.loc[i-1, "High"])
        stock_data.loc[i, "Max_Low"] = min(stock_data.loc[i, "Low"], stock_data.loc[i-1, "Low"])
    else:
        stock_data.loc[i, "Rider_Bullish"] = "No"
        stock_data.loc[i, "Bullish_Level"] = 100000
    if abs((stock_data.loc[i, "Low"] - stock_data.loc[i-1, "Low"]) / stock_data.loc[i-1, "Low"] * 100) < 0.5:
        stock_data.loc[i, "Rider_Bearish"] = "Yes"
        stock_data.loc[i, "Bearish_Level"] = min(stock_data.loc[i, "Low"], stock_data.loc[i-1, "Low"])
        stock_data.loc[i, "Max_High"] = max(stock_data.loc[i, "High"], stock_data.loc[i-1, "High"])
        stock_data.loc[i, "Max_Low"] = min(stock_data.loc[i, "Low"], stock_data.loc[i-1, "Low"])
    else:
        stock_data.loc[i, "Rider_Bearish"] = "No"
        stock_data.loc[i, "Bearish_Level"] = 0


df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

final_5_min_stocks = stock_5_min_historic_data.sort_values(by='date').reset_index(drop=True)
query = '''
SELECT *,
       DENSE_RANK() OVER (ORDER BY date) as dns_rank
FROM stock_5_min_historic_data sm
'''
final_5_min_stocks = sqldf(query, locals())

increment = 0
Signal_df = pd.DataFrame(columns=["Strategy", "Stock", "Signal", "Datetime", "Value", "Date", "StopLoss", "Target","Potential_Target","Potential_Stoploss"])

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# print(final_5_min_stocks)

for i in range(2, max(final_5_min_stocks['dns_rank'])):
# for i in range(2, 10):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i - 1]

    current_date = current_data.iloc[0]["date"]
    previous_date = temp_data2.iloc[0]["date"]
    
    # print(current_date)
    # print(previous_date)

    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
        
    # print(temp_stock)

    if len(temp_stock) > 0:
        temp_stock = temp_stock.drop_duplicates()
        temp_stock.reset_index(drop=True, inplace=True)

        if temp_stock.iloc[0]['Rider_Bullish'] == "Yes":
            satisfied_df = pd.DataFrame()
            for i in range(len(current_data)):
                if current_data.iloc[i]['Close'] > temp_stock.iloc[0]['Bullish_Level']:
                    satisfied_df = satisfied_df.append(current_data.iloc[i])
                    potential_stoploss = temp_stock.iloc[0]['Max_Low']
                    satisfied_df['Potential_Stoploss'] = potential_stoploss
                    stoploss_points = current_data.iloc[i]['Close'] - potential_stoploss
                    target_points = current_data.iloc[i]['Close'] + (1 * stoploss_points)
                    satisfied_df['Potential_Target'] = target_points
                    break
                else:
                    continue
            if len(satisfied_df) == 0:
                continue
            else:
                satisfied_df.reset_index(drop=True, inplace=True)
                # print(satisfied_df)
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")
                

                if curr_hr == "15" and int(curr_min) >= 15:
                    continue
                else:
                    Signal_df.loc[increment, "Strategy"] = "Cowboy"
                    Signal_df.loc[increment, "Stock"] = ticker
                    Signal_df.loc[increment, "Signal"] = "Buy"
                    Signal_df.loc[increment, "Datetime"] = satisfied_df.iloc[0]['DateTime']
                    Signal_df.loc[increment, "Value"] = satisfied_df.iloc[0]['Close']
                    Signal_df.loc[increment, "Potential_Stoploss"] = satisfied_df.iloc[0]['Potential_Stoploss']
                    Signal_df.loc[increment, "Potential_Target"] = satisfied_df.iloc[0]['Potential_Target']
                    increment += 1

        else:
            continue

        if temp_stock.iloc[0]['Rider_Bearish'] == "Yes":
            satisfied_df = pd.DataFrame()
            for i in range(len(current_data)):
                if current_data.iloc[i]['Close'] < temp_stock.iloc[0]['Bearish_Level']:
                    satisfied_df = satisfied_df.append(current_data.iloc[i])

                    potential_stoploss = temp_stock.iloc[0]['Max_High']
                    satisfied_df['Potential_Stoploss'] = potential_stoploss
                    stoploss_points = potential_stoploss - current_data.iloc[i]['Close']
                    target_points = current_data.iloc[i]['Close'] - (1 * stoploss_points)
                    satisfied_df['Potential_Target'] = target_points

                    break
                else:
                    continue
            if len(satisfied_df) == 0:
                continue
            else:
                satisfied_df.reset_index(drop=True, inplace=True)
                # print("satisfied_df")
                # print(satisfied_df)
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")

                if curr_hr == "15" and int(curr_min) >= 15:
                    continue
                else:
                    Signal_df.loc[increment, "Strategy"] = "Cowboy"
                    Signal_df.loc[increment, "Stock"] = ticker
                    Signal_df.loc[increment, "Signal"] = "Sell"
                    Signal_df.loc[increment, "Datetime"] = satisfied_df.iloc[0]['DateTime']
                    Signal_df.loc[increment, "Value"] = satisfied_df.iloc[0]['Close']
                    Signal_df.loc[increment, "Potential_Stoploss"] = satisfied_df.iloc[0]['Potential_Stoploss']
                    Signal_df.loc[increment, "Potential_Target"] = satisfied_df.iloc[0]['Potential_Target']
                    increment += 1

# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

#Sweths Violation

# Signal_df = pd.DataFrame(columns=["Strategy", "Stock", "Signal", "Datetime", "Value"])
# increment = 1

# stock = 'TSLA'

print("Running Violation")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

# Perform SQL query using sqldf
final_5_min_stocks = sqldf("SELECT *, dense_rank() OVER (ORDER BY date) AS dns_rank FROM stock_5_min_historic_data")

# Signal_df = pd.DataFrame(columns=["Strategy", "Stock", "Signal", "Datetime", "Value"])
# increment = 0


for i in range(1, max(final_5_min_stocks["dns_rank"]) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
    current_data.reset_index(drop=True, inplace=True)
    current_date = current_data.loc[0, "date"]

    trigger_price = 0
    stage = ""

    if current_data.loc[0, "Close"] > current_data.loc[0, "Open"] and abs(current_data.loc[0, "Close"] - current_data.loc[0, "Open"]) >= 0.7 * abs(current_data.loc[0, "High"] - current_data.loc[0, "Low"]):
        trigger_price = current_data.loc[0, "Low"]
        stage = "Green"
    elif current_data.loc[0, "Close"] < current_data.loc[0, "Open"] and abs(current_data.loc[0, "Close"] - current_data.loc[0, "Open"]) >= 0.7 * abs(current_data.loc[0, "High"] - current_data.loc[0, "Low"]):
        trigger_price = current_data.loc[0, "High"]
        stage = "Red"
    else:
        continue

    satisfied_df = pd.DataFrame()

    for j in range(4, len(current_data)):
        if stage == "Green":
            if current_data.loc[j, "Close"] < trigger_price:
                satisfied_df = satisfied_df.append(current_data.loc[j])
                call = "Sell"
        elif stage == "Red":
            if current_data.loc[j, "Close"] > trigger_price:
                satisfied_df = satisfied_df.append(current_data.loc[j])
                call = "Buy"
        else:
            continue

    if len(satisfied_df) == 0:
        continue
    else:
        satisfied_df.reset_index(drop=True, inplace=True)
        # print("satisfied_df")
        # print(satisfied_df)
        datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
        datetime = pd.to_datetime(datetime_str)
        curr_hr = datetime.strftime("%H")
        curr_min = datetime.strftime("%M")

        if curr_hr == 15 and curr_min >= 15:
            continue
        else:
            # Signal_df.loc[increment] = ["Sweths Violation",stock, call, satisfied_df.loc[0, "DateTime"], satisfied_df.loc[0, "Close"]]
            Signal_df.loc[increment, "Strategy"] = "Sweths Violation"
            Signal_df.loc[increment, "Stock"] = stock
            Signal_df.loc[increment, "Signal"] = call
            Signal_df.loc[increment, "Datetime"] = satisfied_df.loc[0, "DateTime"]
            Signal_df.loc[increment, "Value"] = satisfied_df.loc[0, "Close"]
            increment += 1

# print(Signal_df)

# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

# Reds Rocket

print("Running Reds Rocket")

# final_levels_df = pd.DataFrame(columns=["Date", "Stock", "Reds_High", "Reds_Low"])

stock_data["Reds_High"] = 0
stock_data["Reds_Low"] = 0
stock_data["Reds_Satisfied"] = "No"

inc = 0

# stock = 'TSLA'

for i in range(3, len(stock_data)):
    current_date = stock_data.loc[i, "Date"]

    l1_day_range = abs(stock_data.loc[i, "High"] - stock_data.loc[i, "Low"])
    l2_day_range = abs(stock_data.loc[i-1, "High"] - stock_data.loc[i-1, "Low"])
    l3_day_range = abs(stock_data.loc[i-2, "High"] - stock_data.loc[i-2, "Low"])
    l4_day_range = abs(stock_data.loc[i-3, "High"] - stock_data.loc[i-3, "Low"])

    l2_day_high = stock_data.loc[i-1, "High"]
    l1_day_high = stock_data.loc[i, "High"]

    l2_day_low = stock_data.loc[i-1, "Low"]
    l1_day_low = stock_data.loc[i, "Low"]

    if (l1_day_range < l2_day_range) and (l1_day_range < l3_day_range) and (l1_day_range < l4_day_range):
        if l1_day_low > l2_day_low and l1_day_high < l2_day_high:
            # final_levels_df.loc[inc, "Date"] = current_date
            # final_levels_df.loc[inc, "Stock"] = stock
            # final_levels_df.loc[inc, "Reds_High"] = l1_day_high
            # final_levels_df.loc[inc, "Reds_Low"] = l1_day_low

            stock_data.loc[i, "Reds_High"] = l1_day_high
            stock_data.loc[i, "Reds_Low"] = l1_day_low
            stock_data.loc[i,"Reds_Satisfied"] = "Yes"
            inc += 1
    else:
        continue

# stock_data = final_levels_df


df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

# Perform SQL query using sqldf
final_5_min_stocks = sqldf("SELECT *,dense_rank() OVER (ORDER BY date) AS dns_rank FROM stock_5_min_historic_data sm")

# Create Signal_df DataFrame
# Signal_df = pd.DataFrame({'Strategy': [],
#                           'Stock': [],
#                           'Signal': [],
#                           'Datetime': [],
#                           'Value': []})

# increment = 1

# print(stock_data)
                    
    
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

for i in range(2, max(final_5_min_stocks['dns_rank']) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks['dns_rank'] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks['dns_rank'] == i - 1]
    
    current_date = current_data.iloc[0]['date']
    previous_date = temp_data2.iloc[0]['date']
    
    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
    
    if len(temp_stock) > 0:
        temp_stock.reset_index(drop=True, inplace=True)

        if temp_stock.iloc[0]['Reds_Satisfied'] == "Yes":
            
            current_data['Call'] = ""
            
            satisfied_df = pd.DataFrame(columns=['dates', 'Open', 'High', 'Low', 'Close', 'Volume', 'Call'])
            
            for j in range(len(current_data)):
                if current_data.iloc[j]['Close'] > temp_stock.iloc[0]['Reds_High']:
                    satisfied_df = satisfied_df.append(current_data.iloc[j])
                    satisfied_df.loc[satisfied_df.index[-1], 'Call'] = "Buy"
                elif current_data.iloc[j]['Close'] < temp_stock.iloc[0]['Reds_Low']:
                    satisfied_df = satisfied_df.append(current_data.iloc[j])
                    satisfied_df.loc[satisfied_df.index[-1], 'Call'] = "Sell"
            
            if len(satisfied_df) == 0:
                continue
            else:
                satisfied_df.reset_index(drop=True, inplace=True)
                # print("satisfied_df")
                # print(satisfied_df)
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")
                
                if curr_hr == 15 and curr_min >= 15:
                    continue
                else:
                    Signal_df.loc[increment, 'Strategy'] = "Reds Rocket"
                    Signal_df.loc[increment, 'Stock'] = stock
                    Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
                    Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
                    Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']
                    
                    increment += 1

# print(Signal_df)
# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

# Reds Brahmos

print("Running Reds Brahmos")

stock_data["Reds_High"] = 0
stock_data["Reds_Low"] = 0
stock_data["Reds_Satisfied"] = "No"

# stock = "TSLA"

# inc = 0

for i in range(5, len(stock_data)):
    current_date = stock_data.loc[i, "Date"]
    
    l1_day_range = abs(stock_data.loc[i, "High"] - stock_data.loc[i, "Low"])
    l2_day_range = abs(stock_data.loc[i-1, "High"] - stock_data.loc[i-1, "Low"])
    l3_day_range = abs(stock_data.loc[i-2, "High"] - stock_data.loc[i-2, "Low"])
    l4_day_range = abs(stock_data.loc[i-3, "High"] - stock_data.loc[i-3, "Low"])
    l5_day_range = abs(stock_data.loc[i-4, "High"] - stock_data.loc[i-4, "Low"])
    l6_day_range = abs(stock_data.loc[i-5, "High"] - stock_data.loc[i-5, "Low"])
    
    l2_day_high = stock_data.loc[i-1, "High"]
    l1_day_high = stock_data.loc[i, "High"]
    
    l2_day_low = stock_data.loc[i-1, "Low"]
    l1_day_low = stock_data.loc[i, "Low"]
    
    if (l1_day_range < l2_day_range) and (l1_day_range < l3_day_range) and (l1_day_range < l4_day_range) and (l1_day_range < l5_day_range) and (l1_day_range < l6_day_range):
        # final_levels_df.loc[inc, "Date"] = current_date
        # final_levels_df.loc[inc, "Stock"] = stock
        # final_levels_df.loc[inc, "Reds_High"] = l1_day_high
        # final_levels_df.loc[inc, "Reds_Low"] = l1_day_low
        stock_data.loc[i, "Reds_High"] = l1_day_high
        stock_data.loc[i, "Reds_Low"] = l1_day_low
        stock_data.loc[i,"Reds_Satisfied"] = "Yes"
        
        inc += 1
        
# stock_data = final_levels_df

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
            dense_rank() OVER (ORDER BY date) as dns_rank
        FROM stock_5_min_historic_data sm
        """

final_5_min_stocks = sqldf(query, locals())

# Signal_df = pd.DataFrame({
#     'Strategy': [],
#     'Stock': [],
#     'Signal': [],
#     'Datetime': [],
#     'Value': []
# })

# increment = 1

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

for i in range(2, max(final_5_min_stocks['dns_rank']) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks['dns_rank'] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks['dns_rank'] == i-1]
    
    current_date = current_data.iloc[0]['date']
    previous_date = temp_data2.iloc[0]['date']
    
    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
    
    if len(temp_stock) > 0:
        temp_stock = temp_stock.reset_index(drop=True)

        if temp_stock.iloc[0]['Reds_Satisfied'] == "Yes":
            current_data['Call'] = ""
            
            satisfied_df = pd.DataFrame(columns=['dates', 'Open', 'High', 'Low', 'Close', 'Volume', 'Call'])
            
            for j in range(len(current_data)):
                if current_data.iloc[j]['Close'] > temp_stock.iloc[0]['Reds_High']:
                    satisfied_df = satisfied_df.append(current_data.iloc[j])
                    satisfied_df.reset_index(drop=True, inplace=True)
                    satisfied_df.at[satisfied_df.index[-1], 'Call'] = "Buy"
                elif current_data.iloc[j]['Close'] < temp_stock.iloc[0]['Reds_Low']:
                    satisfied_df = satisfied_df.append(current_data.iloc[j])
                    satisfied_df.reset_index(drop=True, inplace=True)
                    satisfied_df.at[satisfied_df.index[-1], 'Call'] = "Sell"
                else:
                    continue
            
            if len(satisfied_df) == 0:
                continue
            else:
                satisfied_df = satisfied_df.head(1)
                satisfied_df.reset_index(inplace=True,drop=True)
                
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")
                
                if curr_hr == 15 and curr_min >= 15:
                    continue
                else:
                    Signal_df.loc[increment, "Strategy"] = "Reds Brahmos"
                    Signal_df.loc[increment, "Stock"] = stock
                    Signal_df.loc[increment, "Signal"] = satisfied_df.iloc[0]['Call']
                    Signal_df.loc[increment, "Datetime"] = satisfied_df.iloc[0]['DateTime']
                    Signal_df.loc[increment, "Value"] = satisfied_df.iloc[0]['Close']
                    
                    increment += 1
# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))


print("Running Blackout")
# final_levels_df = pd.DataFrame(columns=["Date", "Stock", "target", "stage"])
inc = 0
# stock = "TSLA"

stock_data["target"] = 0
stock_data["stage"] = ""

for i in range(3, len(stock_data)):
    current_date = stock_data.loc[i, "Date"]
    
    l4_high = stock_data.loc[i-3, "High"]
    l3_high = stock_data.loc[i-2, "High"]
    l2_high = stock_data.loc[i-1, "High"]
    l1_high = stock_data.loc[i, "High"]
    
    l4_low = stock_data.loc[i-3, "Low"]
    l3_low = stock_data.loc[i-2, "Low"]
    l2_low = stock_data.loc[i-1, "Low"]
    l1_low = stock_data.loc[i, "Low"]
    
    if (l1_low > l2_low) and (l1_high > l2_high) and (l2_low > l3_low) and (l2_high > l3_high) and (l3_low > l4_low) and (l3_high > l4_high):
        l1_open = stock_data.iloc[i, 1]
        l1_close = stock_data.iloc[i, 4]
        real_body = abs(l1_open - l1_close)
        body_high = max(l1_open, l1_close)
        
        if (l1_high - body_high) > 2 * real_body:
            # final_levels_df.loc[inc, "Stock"] = stock
            # final_levels_df.loc[inc, "target"] = l1_low
            # final_levels_df.loc[inc, "stage"] = "Short"
            # final_levels_df.loc[inc, "Date"] = current_date
            stock_data.loc[i,"target"] = l1_low
            stock_data.loc[i, "stage"] = "Short"
            inc += 1
        
    elif (l1_low < l2_low) and (l1_high < l2_high) and (l2_low < l3_low) and (l2_high < l3_high) and (l3_low < l4_low) and (l3_high < l4_high):
        l1_open = stock_data.iloc[i, 1]
        l1_close = stock_data.iloc[i, 4]
        real_body = abs(l1_open - l1_close)
        body_low = min(l1_open, l1_close)
        
        if (l1_low - body_low) > 2 * real_body:
            # final_levels_df.loc[inc, "Stock"] = stock
            # final_levels_df.loc[inc, "target"] = l1_high
            # final_levels_df.loc[inc, "stage"] = "Long"
            # final_levels_df.loc[inc, "Date"] = current_date
            stock_data.loc[i,"target"] = l1_high
            stock_data.loc[i, "stage"] = "Long"
            inc += 1

# stock_data = final_levels_df


df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
            dense_rank() OVER (ORDER BY date) as dns_rank
        FROM stock_5_min_historic_data sm
        """

final_5_min_stocks = sqldf(query, locals())

# Signal_df = pd.DataFrame({
#     'Strategy': [],
#     'Stock': [],
#     'Signal': [],
#     'Datetime': [],
#     'Value': []
# })

# increment = 1

stock_data['Date'] = pd.to_datetime(stock_data['Date'])


for i in range(2, max(final_5_min_stocks["dns_rank"])+1):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i - 1]

    current_date = current_data.iloc[0]["date"]
    previous_date = temp_data2.iloc[0]["date"]

    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
    
    if temp_stock.shape[0] > 0:
        temp_stock = temp_stock.iloc[[0], :]
        stage = temp_stock.iloc[0, :]["stage"]
        if stage != "":
            target_value = temp_stock.iloc[0, :]["target"]
            current_data["Call"] = ""
            
            satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])
            
            if stage == "Short":
                for j in range(current_data.shape[0]):
                    if current_data.iloc[j, :]["Close"] < target_value:
                        satisfied_df = satisfied_df.append(current_data.iloc[j, :])
    #                     satisfied_df.iloc[-1, :]["Call"] = "Sell"
                        satisfied_df.at[satisfied_df.index[-1], 'Call'] = "Sell"
            else:
                for j in range(current_data.shape[0]):
                    if current_data.iloc[j, :]["Close"] > target_value:
                        satisfied_df = satisfied_df.append(current_data.iloc[j, :])
    #                     satisfied_df.iloc[-1, :]["Call"] = "Buy"
                        satisfied_df.at[satisfied_df.index[-1], 'Call'] = "Buy"
            
            if satisfied_df.shape[0] == 0:
                continue
            else:
                satisfied_df = satisfied_df.head(1)
                
                satisfied_df.reset_index(inplace=True,drop=True)
                
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")
                
                if curr_hr == 15 and curr_min >= 15:
                    continue
                else:
                    Signal_df.loc[increment, "Strategy"] = "Blackout"
                    Signal_df.loc[increment, "Stock"] = stock
                    Signal_df.loc[increment, "Signal"] = satisfied_df.iloc[0, :]["Call"]
                    Signal_df.loc[increment, "Datetime"] = satisfied_df.iloc[0, :]["DateTime"]
                    Signal_df.loc[increment, "Value"] = satisfied_df.iloc[0, :]["Close"]
                    
                    increment += 1
# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

print("Running Gap up")
                

stock_data["Stock"] = stock
# stock_data["Previous_Open"] = stock_data["Open"].shift(1)
# stock_data["Previous_High"] = stock_data["High"].shift(1)
# stock_data["Previous_Low"] = stock_data["Low"].shift(1)
# stock_data["Previous_Close"] = stock_data["Close"].shift(1)

stock_data = stock_data.dropna().reset_index(drop=True)

stock_data = stock_data.drop_duplicates().reset_index(drop=True)

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_5_min_historic_data
        """

final_5_min_stocks = sqldf(query)

stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# print("stock_data")
# print(stock_data)

for i in range(2, max(final_5_min_stocks["dns_rank"]) + 1):
# for i in range(2, 10):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i - 1]

    current_date = current_data.iloc[0]["date"]
    previous_date = temp_data2.iloc[0]["date"]
    
    # print(current_date)
    # print(previous_date)

    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
        
    # print(temp_stock)
    # print(current_data)
    
    if len(temp_stock) > 0:
        temp_stock.reset_index(inplace=True,drop=True)
        current_data.reset_index(inplace=True,drop=True)
        high_price = temp_stock.loc[0,"High"]
        previous_close = temp_stock.loc[0,"Close"]

        current_data["Call"] = ""

        satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

        open_price = current_data.loc[0, 'Open']
        close_price = current_data.loc[0, 'Close']
        
        # print(open_price)
        # print(close_price)

        if open_price > previous_close:
            for j in range(4, current_data.shape[0]):
                current_date = current_data.loc[j, 'DateTime']

                day_high = max(
                    max(current_data.loc[0:j-1, 'Close'].dropna()),
                    max(current_data.loc[0:j-1, 'Open'].dropna())
                )

                day_low = min(
                    min(current_data.loc[0:j-1, 'Close'].dropna()),
                    min(current_data.loc[0:j-1, 'Open'].dropna())
                )

                low_range = min(
                    current_data.loc[j-1, 'Low'],
                    current_data.loc[j-2, 'Low'],
                    current_data.loc[j-3, 'Low'],
                    current_data.loc[j-4, 'Low']
                )

                high_range = max(
                    current_data.loc[j-1, 'High'],
                    current_data.loc[j-2, 'High'],
                    current_data.loc[j-3, 'High'],
                    current_data.loc[j-4, 'High']
                )

                current_close = current_data.loc[j, 'Close']

                if (abs(high_range - low_range) / low_range * 100 < 0.4) and (current_close >= high_price) and (current_close >= day_high):
                    satisfied_df = satisfied_df.append(current_data.loc[j], ignore_index=True)
                    satisfied_df.loc[satisfied_df.shape[0]-1, 'Call'] = "Buy"

                elif (abs(high_range - low_range) / low_range * 100 < 0.4) and (current_close <= close_price) and (current_close <= day_low):
                    satisfied_df = satisfied_df.append(current_data.loc[j], ignore_index=True)
                    satisfied_df.loc[satisfied_df.shape[0]-1, 'Call'] = "Sell"

            if len(satisfied_df) == 0:
                pass
            else:
                satisfied_df = satisfied_df.head(1)
                satisfied_df.reset_index(inplace=True,drop=True)
                
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")

                if curr_hr == 15 and curr_min >= 15:
                    pass
                else:
                    Signal_df.loc[increment, 'Strategy'] = "Gap_up"
                    Signal_df.loc[increment, 'Stock'] = stock
                    Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
                    Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
                    Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

                    increment += 1
# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))
    
#Gap Down

print("Running Gap down")
stock_data["Stock"] = stock
# stock_data["Previous_Open"] = stock_data["Open"].shift(1)
# stock_data["Previous_High"] = stock_data["High"].shift(1)
# stock_data["Previous_Low"] = stock_data["Low"].shift(1)
# stock_data["Previous_Close"] = stock_data["Close"].shift(1)

stock_data = stock_data.dropna().reset_index(drop=True)

stock_data = stock_data.drop_duplicates().reset_index(drop=True)

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_5_min_historic_data
        """

final_5_min_stocks = sqldf(query)


stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# print("stock_data")
# print(stock_data)

for i in range(2, max(final_5_min_stocks["dns_rank"]) + 1):
# for i in range(2, 10):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
    temp_data2 = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i - 1]

    current_date = current_data.iloc[0]["date"]
    previous_date = temp_data2.iloc[0]["date"]
    
    # print(current_date)
    # print(previous_date)

    temp_stock = stock_data[stock_data["Date"].dt.date.astype(str) == previous_date]
        
    # print(temp_stock)
    # print(current_data)
    
    if len(temp_stock) > 0:
        temp_stock.reset_index(inplace=True,drop=True)
        current_data.reset_index(inplace=True,drop=True)
        high_price = temp_stock.loc[0,"High"]
        previous_close = temp_stock.loc[0,"Close"]
        prev_low_price = temp_stock.loc[0, "Low"]

        current_data["Call"] = ""

        satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

        open_price = current_data.loc[0, 'Open']
        close_price = current_data.loc[0, 'Close']

        if open_price < previous_close:

            for j in range(4, current_data.shape[0]):
                current_date = current_data.loc[j, 'DateTime']

                day_high = max(
                    max(current_data.loc[0:j-1, 'Close'].dropna()),
                    max(current_data.loc[0:j-1, 'Open'].dropna())
                )

                day_low = min(
                    min(current_data.loc[0:j-1, 'Close'].dropna()),
                    min(current_data.loc[0:j-1, 'Open'].dropna())
                )

                low_range = min(
                    current_data.loc[j-1, 'Low'],
                    current_data.loc[j-2, 'Low'],
                    current_data.loc[j-3, 'Low'],
                    current_data.loc[j-4, 'Low']
                )

                high_range = max(
                    current_data.loc[j-1, 'High'],
                    current_data.loc[j-2, 'High'],
                    current_data.loc[j-3, 'High'],
                    current_data.loc[j-4, 'High']
                )

                current_close = current_data.loc[j, 'Close']

                if ((abs(high_range - low_range) / low_range * 100 < 0.4) and (current_close >= high_price) and (current_close >= day_high)):
                    satisfied_df = satisfied_df.append(current_data.loc[j], ignore_index=True)
                    satisfied_df.loc[satisfied_df.shape[0]-1, 'Call'] = "Buy"

                elif (abs(high_range - low_range) / low_range * 100 < 0.4) and (current_close <= prev_low_price) and (current_close <= day_low):
                    satisfied_df = satisfied_df.append(current_data.loc[j], ignore_index=True)
                    satisfied_df.loc[satisfied_df.shape[0]-1, 'Call'] = "Sell"

            if len(satisfied_df) == 0:
                pass
            else:
                satisfied_df = satisfied_df.head(1)
                satisfied_df.reset_index(inplace=True,drop=True)
                
                datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
                datetime = pd.to_datetime(datetime_str)
                curr_hr = datetime.strftime("%H")
                curr_min = datetime.strftime("%M")

                if curr_hr == 15 and curr_min >= 15:
                    pass
                else:
                    Signal_df.loc[increment, 'Strategy'] = "Gap_down"
                    Signal_df.loc[increment, 'Stock'] = stock
                    Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
                    Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
                    Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

                    increment += 1

# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

# 5 candle ABC

print("Running 5 Candle ABC")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_5_min_historic_data
        """

final_5_min_stocks = sqldf(query)

for i in range(1, max(final_5_min_stocks["dns_rank"]) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]

    current_date = current_data.iloc[0]["date"]
    current_data.reset_index(inplace=True,drop=True)

    satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

    for j in range(5, current_data.shape[0]):
        if current_data.loc[j, "Close"] > current_data.loc[j, "Open"]:
            # Check if the prior candles are in the reversal trend
            if (current_data.loc[j - 1, "Low"] < current_data.loc[j - 2, "Low"]) and (
                current_data.loc[j - 2, "Low"] < current_data.loc[j - 3, "Low"]
            ):
                # Get the breakout max in the reversal i.e., B Point
                reversal_high = max(
                    current_data.loc[j - 1, "High"],
                    current_data.loc[j - 2, "High"],
                    current_data.loc[j - 3, "High"],
                )

                # Get the breakout min in the reversal i.e., C point
                reversal_low = min(
                    current_data.loc[j - 1, "Low"],
                    current_data.loc[j - 2, "Low"],
                    current_data.loc[j - 3, "Low"],
                )

                # Check if the before reversal is a uptrend
                if (
                    current_data.loc[j - 3, "High"] > current_data.loc[j - 4, "High"]
                    and current_data.loc[j - 4, "High"] > current_data.loc[j - 5, "High"]
                ):
                    # Get the starting point of the trend i.e., A point
                    trend_low = min(current_data.loc[j - 4, "Low"], current_data.loc[j - 5, "Low"])

                    # Check if the ABC pattern is completely followed
                    if (current_data.loc[j, "Close"] > reversal_high and reversal_low > trend_low):
                        temp_call_data = current_data.loc[j,]
                        temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
                        satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

        else:
            # Check if the prior candles are in the reversal trend
            if (
                current_data.loc[j - 1, "High"] > current_data.loc[j - 2, "High"]
                and current_data.loc[j - 2, "High"] > current_data.loc[j - 3, "High"]
            ):

                # Get the breakout max in the reversal i.e., B Point
                reversal_high = min(
                    current_data.loc[j - 1, "Low"],
                    current_data.loc[j - 2, "Low"],
                    current_data.loc[j - 3, "Low"],
                )

                # Get the breakout min in the reversal i.e., C point
                reversal_low = max(
                    current_data.loc[j - 1, "High"],
                    current_data.loc[j - 2, "High"],
                    current_data.loc[j - 3, "High"],
                )

                # Check if the before reversal is a uptrend
                if (
                    current_data.loc[j - 3, "Low"] < current_data.loc[j - 4, "Low"]
                    and current_data.loc[j - 4, "Low"] < current_data.loc[j - 5, "Low"]
                ):
                    # Get the starting point of the trend i.e., A point
                    trend_low = max(
                        current_data.loc[j - 4, "High"], current_data.loc[j - 5, "High"]
                    )

                    # Check if the ABC pattern is completely followed
                    if (
                        current_data.loc[j, "Close"] < reversal_high
                        and reversal_low < trend_low
                    ):
                        temp_call_data = current_data.loc[j,]
                        temp_call_data = temp_call_data.append(pd.Series("Sell", index=["Call"]))
                        satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

    if len(satisfied_df) == 0:
        pass
    else:
        satisfied_df = satisfied_df.head(1)
        satisfied_df.reset_index(inplace=True,drop=True)
        
        datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
        datetime = pd.to_datetime(datetime_str)
        curr_hr = datetime.strftime("%H")
        curr_min = datetime.strftime("%M")

        if curr_hr == 15 and curr_min >= 15:
            pass
        else:
            Signal_df.loc[increment, 'Strategy'] = "5_Cand_ABC"
            Signal_df.loc[increment, 'Stock'] = stock
            Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
            Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
            Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

            increment += 1

# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

# 3 candle ABC
print("Running 3 Candle ABC")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_5_min_historic_data
        """

final_5_min_stocks = sqldf(query)

for i in range(1, max(final_5_min_stocks["dns_rank"]) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]

    current_date = current_data.iloc[0]["date"]
    current_data.reset_index(inplace=True,drop=True)

    satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

    for j in range(2, current_data.shape[0]):

        if ((current_data.loc[j, "Close"] > current_data.loc[j, "Open"])
            and (current_data.loc[j - 1, "Close"] < current_data.loc[j - 1, "Open"])
            and (current_data.loc[j - 2, "Close"] > current_data.loc[j - 2, "Open"])
            ):

            if ((current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"])
                and (current_data.loc[j, "Close"] > current_data.loc[j - 2, "High"])
                and (current_data.loc[j - 1, "High"] < current_data.loc[j - 2, "High"])
                ):

                first_range = (
                    current_data.loc[j - 2, "High"] - current_data.loc[j - 2, "Low"]
                )
                second_range = (
                    current_data.loc[j - 1, "High"] - current_data.loc[j - 1, "Low"]
                )

                if first_range / second_range >= 2:
                    temp_call_data = current_data.loc[j,]
                    temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
                    satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

        elif ((current_data.loc[j, "Close"] < current_data.loc[j, "Open"])
            and (current_data.loc[j - 1, "Close"] > current_data.loc[j - 1, "Open"])
            and (current_data.loc[j - 2, "Close"] < current_data.loc[j - 2, "Open"])
            ):

            if ((current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"])
                and (current_data.loc[j, "Close"] < current_data.loc[j - 2, "Low"])
                and (current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"])
                ):
                first_range = (
                    current_data.loc[j - 2, "High"] - current_data.loc[j - 2, "Low"]
                )
                second_range = (
                    current_data.loc[j - 1, "High"] - current_data.loc[j - 1, "Low"]
                )
                if first_range / second_range >= 2:
                    temp_call_data = current_data.loc[j,]
                    temp_call_data = temp_call_data.append(pd.Series("Sell", index=["Call"]))
                    satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

    if len(satisfied_df) == 0:
        pass
    else:
        satisfied_df = satisfied_df.head(1)
        satisfied_df.reset_index(inplace=True,drop=True)
        
        datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
        datetime = pd.to_datetime(datetime_str)
        curr_hr = datetime.strftime("%H")
        curr_min = datetime.strftime("%M")

        if curr_hr == 15 and curr_min >= 15:
            pass
        else:
            Signal_df.loc[increment, 'Strategy'] = "3_Cand_ABC"
            Signal_df.loc[increment, 'Stock'] = stock
            Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
            Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
            Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

            increment += 1

# Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

# # 3 candle ABC
# print("Running ABCD")


# df['DateTime'] = pd.to_datetime(df['DateTime'])
# df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

# df = df[df['DateTime'].between(start_date, end_date)]
# stock_5_min_historic_data = df

# stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

# stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

# query = """
#         SELECT *,
#         dense_rank() OVER (ORDER BY date) AS dns_rank
#         FROM stock_5_min_historic_data
#         """

# final_5_min_stocks = sqldf(query)

# # Define the required Fibonacci ratios
# ABCD_Ratios = {
#     'AB': 0.382,
#     'BC': 0.618,
#     'CD': 1.272
# }

# # Iterate over each rank in the data
# for i in range(1, max(final_5_min_stocks["dns_rank"]) + 1):
#     current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]
#     current_date = current_data.iloc[0]["date"]
#     current_data.reset_index(inplace=True, drop=True)

#     satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

#     # Iterate over each candle in the current rank
#     for j in range(2, current_data.shape[0]):
#         if (
#             current_data.loc[j, "Close"] > current_data.loc[j, "Open"] and
#             current_data.loc[j - 1, "Close"] < current_data.loc[j - 1, "Open"] and
#             current_data.loc[j - 2, "Close"] > current_data.loc[j - 2, "Open"]
#         ):
#             if (
#                 current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"] and
#                 current_data.loc[j, "Close"] > current_data.loc[j - 2, "High"] and
#                 current_data.loc[j - 1, "High"] < current_data.loc[j - 2, "High"]
#             ):
#                 first_range = current_data.loc[j - 2, "High"] - current_data.loc[j - 2, "Low"]
#                 second_range = current_data.loc[j - 1, "High"] - current_data.loc[j - 1, "Low"]

#                 if first_range / second_range >= ABCD_Ratios['AB']:
#                     CD_range = ABCD_Ratios['CD'] * second_range
                    
#                     if current_data.loc[j, "Close"] > current_data.loc[j - 2, "High"] + CD_range:
#                         temp_call_data = current_data.loc[j]
#                         temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
#                         satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

#                         if curr_hr != "15" or (curr_hr == "15" and curr_min < "15"):
#                             Signal_df.loc[increment, 'Strategy'] = "ABCD"
#                             Signal_df.loc[increment, 'Stock'] = satisfied_df.loc[0, 'stock']
#                             Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
#                             Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
#                             Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']
#                             increment += 1

#         elif (
#             current_data.loc[j, "Close"] < current_data.loc[j, "Open"] and
#             current_data.loc[j - 1, "Close"] > current_data.loc[j - 1, "Open"] and
#             current_data.loc[j - 2, "Close"] < current_data.loc[j - 2, "Open"]
#         ):
#             if (
#                 current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"] and
#                 current_data.loc[j, "Close"] < current_data.loc[j - 2, "Low"] and
#                 current_data.loc[j - 1, "Low"] > current_data.loc[j - 2, "Low"]
#             ):
#                 first_range = current_data.loc[j - 2, "High"] - current_data.loc[j - 2, "Low"]
#                 second_range = current_data.loc[j - 1, "High"] - current_data.loc[j - 1, "Low"]

#                 if first_range / second_range >= ABCD_Ratios['AB']:
#                     CD_range = ABCD_Ratios['CD'] * second_range
                    
#                     if current_data.loc[j, "Close"] < current_data.loc[j - 2, "Low"] - CD_range:
#                         temp_call_data = current_data.loc[j]
#                         temp_call_data = temp_call_data.append(pd.Series("Sell", index=["Call"]))
#                         satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

#     if len(satisfied_df) == 0:
#         pass
#     else:
#         satisfied_df = satisfied_df.head(1)
#         satisfied_df.reset_index(inplace=True,drop=True)
        
#         datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
#         datetime = pd.to_datetime(datetime_str)
#         curr_hr = datetime.strftime("%H")
#         curr_min = datetime.strftime("%M")

#         if curr_hr == 15 and curr_min >= 15:
#             pass
#         else:
#             Signal_df.loc[increment, 'Strategy'] = "ABCD"
#             Signal_df.loc[increment, 'Stock'] = stock
#             Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
#             Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
#             Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

#             increment += 1


print("Running Intraday Buying Past 15 Min")

df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume','stock']]

df = df[df['DateTime'].between(start_date, end_date)]
stock_5_min_historic_data = df

stock_5_min_historic_data = pd.DataFrame(stock_5_min_historic_data)

stock_5_min_historic_data['date'] = pd.to_datetime(stock_5_min_historic_data['DateTime']).dt.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_5_min_historic_data
        """

final_5_min_stocks = sqldf(query)

for i in range(1, max(final_5_min_stocks["dns_rank"]) + 1):
    current_data = final_5_min_stocks[final_5_min_stocks["dns_rank"] == i]

    current_date = current_data.iloc[0]["date"]
    current_data.reset_index(inplace=True,drop=True)

    satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

    for j in range(15, current_data.shape[0]):
        sma_volume_15_periods_1 = current_data.loc[j-16:j-2, 'Volume'].mean()
        sma_volume_15_periods_2 = current_data.loc[j-17:j-3, 'Volume'].mean()
        sma_volume_15_periods_3 = current_data.loc[j-18:j-4, 'Volume'].mean()

        price_increase = current_data.loc[j, "Close"] > current_data.loc[j-1, "Close"] > current_data.loc[j-2, "Close"] > current_data.loc[j-3, "Close"]
        volume_above_sma_1 = current_data.loc[j, "Volume"] > sma_volume_15_periods_1
        volume_above_sma_2 = current_data.loc[j-1, "Volume"] > sma_volume_15_periods_2
        volume_above_sma_3 = current_data.loc[j-2, "Volume"] > sma_volume_15_periods_3

        if price_increase and volume_above_sma_1 and volume_above_sma_2 and volume_above_sma_3:
            temp_call_data = current_data.loc[j]
            temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
            satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

    if len(satisfied_df) == 0:
        pass
    else:
        satisfied_df = satisfied_df.head(1)
        satisfied_df.reset_index(inplace=True,drop=True)
        
        datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
        datetime = pd.to_datetime(datetime_str)
        curr_hr = datetime.strftime("%H")
        curr_min = datetime.strftime("%M")

        if curr_hr == 15 and curr_min >= 15:
            pass
        else:
            Signal_df.loc[increment, 'Strategy'] = "Intraday_Buying_Past_15_Min"
            Signal_df.loc[increment, 'Stock'] = stock
            Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
            Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
            Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

            increment += 1
print(len(Signal_df))



print("Running 15 Minute Breakout")

df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df[df['DateTime'].between(start_date, end_date)]

df.set_index('DateTime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume','stock']]


# Resampling the data to 15 minute intervals
df_resampled = df.resample('15Min').agg({'Open': 'first', 
                                          'High': 'max', 
                                          'Low': 'min', 
                                          'Close': 'last', 
                                          'Volume': 'sum',
                                          'stock': 'first'})

df_resampled.dropna(inplace=True)

stock_15_min_historic_data = df_resampled

stock_15_min_historic_data = pd.DataFrame(stock_15_min_historic_data)

stock_15_min_historic_data['date'] = stock_15_min_historic_data.index.date

query = """
        SELECT *,
        dense_rank() OVER (ORDER BY date) AS dns_rank
        FROM stock_15_min_historic_data
        """

final_15_min_stocks = sqldf(query)

for i in range(1, max(final_15_min_stocks["dns_rank"]) + 1):
    current_data = final_15_min_stocks[final_15_min_stocks["dns_rank"] == i]

    current_date = current_data.iloc[0]["date"]
    current_data.reset_index(inplace=True,drop=True)

    satisfied_df = pd.DataFrame(columns=["dates", "Open", "High", "Low", "Close", "Volume", "Call"])

    for j in range(20, current_data.shape[0]):
        max_last_20_periods = max(current_data.loc[j-20:j-1, 'Close'])
        min_last_20_periods = min(current_data.loc[j-20:j-1, 'Close'])
        sma_volume_20_periods = current_data.loc[j-20:j-1, 'Volume'].mean()

        if current_data.loc[j, "Close"] > max_last_20_periods and current_data.loc[j, "Volume"] > sma_volume_20_periods:
            temp_call_data = current_data.loc[j]
            temp_call_data = temp_call_data.append(pd.Series("Buy", index=["Call"]))
            satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)
        
        elif current_data.loc[j, "Close"] < min_last_20_periods and current_data.loc[j, "Volume"] < sma_volume_20_periods:
            temp_call_data = current_data.loc[j]
            temp_call_data = temp_call_data.append(pd.Series("Sell", index=["Call"]))
            satisfied_df = satisfied_df.append(temp_call_data, ignore_index=True)

    if len(satisfied_df) == 0:
        pass
    else:
        satisfied_df = satisfied_df.head(1)
        satisfied_df.reset_index(inplace=True,drop=True)
        
        datetime_str = satisfied_df.iloc[0]['DateTime'].replace("+00:00", "")
        datetime = pd.to_datetime(datetime_str)
        curr_hr = datetime.strftime("%H")
        curr_min = datetime.strftime("%M")

        if curr_hr == 15 and curr_min >= 15:
            pass
        else:
            Signal_df.loc[increment, 'Strategy'] = "15_Min_Breakout"
            Signal_df.loc[increment, 'Stock'] = stock
            Signal_df.loc[increment, 'Signal'] = satisfied_df.loc[0, 'Call']
            Signal_df.loc[increment, 'Datetime'] = satisfied_df.loc[0, 'DateTime']
            Signal_df.loc[increment, 'Value'] = satisfied_df.loc[0, 'Close']

            increment += 1

print(len(Signal_df))


Signal_df.reset_index(inplace=True,drop=True)
print(len(Signal_df))

Signal_df.to_csv('V_5_minutes_signals.csv')


