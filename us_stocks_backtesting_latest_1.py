import numpy as np
from datetime import datetime, timedelta
from pandasql import sqldf
import warnings
import yfinance as yf
import pandas as pd
import pytz
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import joblib
import numpy as np
import pandas_ta as ta
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.inspection import permutation_importance
from tensorflow.keras.metrics import AUC
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier

# Ignore warnings
warnings.filterwarnings("ignore")


stock = ticker = 'V'

# log_file = f'US_Stock_Models/{stock}/logfile.txt'
log_file = f'US_Stock_Models/{stock}/logfile.log'
# Configure the logging module to append log messages
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s', filemode='a')

start_date = "2008-01-01"
end_date = "2023-05-30"

from pymongo import MongoClient
# MongoDB connection details
username = 'Titania'
password = 'Mahadev'

cluster_url = 'cluster0.igqnlsy.mongodb.net'
database_name = 'stocks_5_mins_data'
collection_name = 'United_States_Market'

# MongoDB connection URI
uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority"

# Create a MongoClient instance
client = MongoClient(uri)

# Access the database
db = client[database_name]

# Access the collection
collection = db[collection_name]

# Define the query to filter stocks
query = {
    'stock': {'$in': ['nasdaq', 's_and_p', 'Dow_Jones']}
}

# Define the projection to include only the required fields
projection = {'_id': 0,'DateTime': 1,'Date': 1,'Open': 1,'High': 1,'Low': 1,'Close': 1,'stock': 1}

# Fetch the data from the collection
cursor = collection.find(query, projection)

# Convert the cursor to a DataFrame
df_pivot = pd.DataFrame(list(cursor))

# Pivot the DataFrame to create the desired columns
df_pivot = df_pivot.pivot(index=['DateTime','Date'], columns='stock', values=['Open', 'High', 'Low', 'Close'])

# Flatten the column names
df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

# Reset the index
df_pivot.reset_index(inplace=True)

# Rename the DateTime column
df_pivot.rename(columns={'DateTime': 'DateTime','Date':'Date'}, inplace=True)


# Define the ticker symbols for S&P 500, Dow Jones, and Nasdaq
s_and_p_ticker = "^GSPC"
dow_jones_ticker = "^DJI"
nasdaq_ticker = "^IXIC"

data = yf.download([s_and_p_ticker, dow_jones_ticker, nasdaq_ticker], start=start_date, end=end_date)

# Extract the necessary columns from the data
extracted_data = data[["Open", "High", "Low", "Close"]]

# Rename the columns
extracted_data.columns = ["s_and_p_Open", "s_and_p_High", "s_and_p_Low", "s_and_p_Close",
                          "dow_jones_Open", "dow_jones_High", "dow_jones_Low", "dow_jones_Close",
                          "nasdaq_Open", "nasdaq_High", "nasdaq_Low", "nasdaq_Close"]

# Reset the index to get the date as a separate column
extracted_data.reset_index(inplace=True)

extracted_data["s_and_p_Previous_1_Open"] = extracted_data["s_and_p_Open"].shift(1)
extracted_data["s_and_p_Previous_2_Open"] = extracted_data["s_and_p_Open"].shift(2)
extracted_data["s_and_p_Previous_1_High"] = extracted_data["s_and_p_High"].shift(1)
extracted_data["s_and_p_Previous_2_High"] = extracted_data["s_and_p_High"].shift(2)
extracted_data["s_and_p_Previous_1_Low"] = extracted_data["s_and_p_Low"].shift(1)
extracted_data["s_and_p_Previous_2_Low"] = extracted_data["s_and_p_Low"].shift(2)
extracted_data["s_and_p_Previous_1_Close"] = extracted_data["s_and_p_Close"].shift(1)
extracted_data["s_and_p_Previous_2_Close"] = extracted_data["s_and_p_Close"].shift(2)

extracted_data["dow_jones_Previous_1_Open"] = extracted_data["dow_jones_Open"].shift(1)
extracted_data["dow_jones_Previous_2_Open"] = extracted_data["dow_jones_Open"].shift(2)
extracted_data["dow_jones_Previous_1_High"] = extracted_data["dow_jones_High"].shift(1)
extracted_data["dow_jones_Previous_2_High"] = extracted_data["dow_jones_High"].shift(2)
extracted_data["dow_jones_Previous_1_Low"] = extracted_data["dow_jones_Low"].shift(1)
extracted_data["dow_jones_Previous_2_Low"] = extracted_data["dow_jones_Low"].shift(2)
extracted_data["dow_jones_Previous_1_Close"] = extracted_data["dow_jones_Close"].shift(1)
extracted_data["dow_jones_Previous_2_Close"] = extracted_data["dow_jones_Close"].shift(2)

extracted_data["nasdaq_Previous_1_Open"] = extracted_data["nasdaq_Open"].shift(1)
extracted_data["nasdaq_Previous_2_Open"] = extracted_data["nasdaq_Open"].shift(2)
extracted_data["nasdaq_Previous_1_High"] = extracted_data["nasdaq_High"].shift(1)
extracted_data["nasdaq_Previous_2_High"] = extracted_data["nasdaq_High"].shift(2)
extracted_data["nasdaq_Previous_1_Low"] = extracted_data["nasdaq_Low"].shift(1)
extracted_data["nasdaq_Previous_2_Low"] = extracted_data["nasdaq_Low"].shift(2)
extracted_data["nasdaq_Previous_1_Close"] = extracted_data["nasdaq_Close"].shift(1)
extracted_data["nasdaq_Previous_2_Close"] = extracted_data["nasdaq_Close"].shift(2)

# Calculate previous day's price movement percentage
extracted_data["s_and_p_Prev_Percentage"] = extracted_data["s_and_p_Previous_1_Close"].pct_change() * 100
extracted_data["dow_jones_Prev_Percentage"] = extracted_data["dow_jones_Previous_1_Close"].pct_change() * 100
extracted_data["nasdaq_Prev_Percentage"] = extracted_data["nasdaq_Previous_1_Close"].pct_change() * 100

# Calculate log price movement
extracted_data["s_and_p_Log_Previous_1_Price_Movement"] = np.log(extracted_data["s_and_p_Previous_1_Close"] / extracted_data["s_and_p_Previous_1_Close"].shift(1))
extracted_data["dow_jones_Log_Previous_1_Price_Movement"] = np.log(extracted_data["dow_jones_Previous_1_Close"] / extracted_data["dow_jones_Previous_1_Close"].shift(1))
extracted_data["nasdaq_Log_Previous_1_Price_Movement"] = np.log(extracted_data["nasdaq_Previous_1_Close"] / extracted_data["nasdaq_Previous_1_Close"].shift(1))

# cluster_url = 'titaniacluster.fwxe9u5.mongodb.net'
# cluster_url = 'cluster0.igqnlsy.mongodb.net'
# cluster_url = 'cluster0.92qtk1m.mongodb.net'
cluster_url = 'cluster0.izbxvtf.mongodb.net'

database_name = 'stocks_5_mins_data'
collection_name = 'United_States_Market'
# MongoDB connection URI
uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority"
# Connect to the MongoDB client
client = MongoClient(uri)
# Access the database and collection
db = client[database_name]
collection = db[collection_name]
query = {'stock': stock, 'Date': {'$gte': start_date, '$lte': end_date}}
data = collection.find(query)
df = pd.DataFrame(list(data))

df = df[['DateTime','Open','High','Low','Close','Volume','stock']]

df['stock'].value_counts()

data = yf.download(ticker, start=start_date, end=end_date)
stock_data = pd.DataFrame(data)
stock_data.reset_index(inplace=True)

# Print the DataFrame
print(stock_data)

file_name = f'{stock}_5_minutes_signals.csv'
print(file_name)
final_Signal_df = pd.read_csv(file_name)

print(final_Signal_df['Strategy'].value_counts())

logging.info(final_Signal_df['Strategy'].value_counts())

def target_and_sl(Signal_df):
    if Signal_df.shape[0] == 0:
        print("break")
    else:
        Signal_df["Value"] = Signal_df["Value"].astype(float).round(2)
        Signal_df["Date"] = pd.to_datetime(Signal_df["Datetime"]).dt.date
        Capital = 100000
        target_selection = "Percentage"
        # targets = [1, 1, 1.5, 1.5]  # List of target percentages
        # stop_losses = [1, 0.5, 1, 0.5]  # List of stop-loss percentages
        targets = [1]  # List of target percentages
        stop_losses = [1]  # List of stop-loss percentages
        
        # Create empty lists to store the expanded rows
        expanded_rows = []
        
        for i in range(len(targets)):
            target = targets[i]
            stop_loss = stop_losses[i]
            
            temp_df = Signal_df.copy()
            
            temp_df["StopLoss"] = np.where(temp_df["Signal"] == "Buy", temp_df["Value"] - ((stop_loss * temp_df["Value"]) / 100), ((stop_loss * temp_df["Value"]) / 100) + temp_df["Value"])
            temp_df["Target"] = np.where(temp_df["Signal"] == "Buy", temp_df["Value"] + ((target * temp_df["Value"]) / 100), temp_df["Value"] - ((target * temp_df["Value"]) / 100))
            temp_df["Potential_Target"] = temp_df["Target"]
            temp_df["Potential_Stoploss"] = temp_df["StopLoss"]
            
            expanded_rows.append(temp_df)
        
        # Concatenate the expanded rows into a single DataFrame
        Signal_df = pd.concat(expanded_rows, ignore_index=True)
        
    return Signal_df

strategies = ['Cowboy','Sweths Violation','Reds Rocket','Reds Brahmos','Blackout','Gap_up','Gap_down','5_Cand_ABC','3_Cand_ABC','Intraday_Buying_Past_15_Min','15_Min_Breakout']
scalar_names = ['cowboy_scaler','violation_scaler','rocket_scaler','brahmos_scaler','blackout_scaler','gap_up_scaler','gap_down_scaler','5_cand_ABC_scaler','3_cand_ABC_scaler','Intraday_Buying_Past_15_Min_scalar','15_Min_Breakout_scalar']
model_names = ['cowboy_best_model','violation_best_model','rocket_best_model','brahmos_best_model','blackout_best_model','gap_up_best_model','gap_down_best_model','5_cand_ABC_best_model','3_cand_ABC_best_model','Intraday_Buying_Past_15_Min_best_model','15_Min_Breakout_best_model']

# strategies = ['Gap_up','Gap_down','5_Cand_ABC','3_Cand_ABC']
# scalar_names = ['5_cand_ABC_scaler','3_cand_ABC_scaler']
# model_names = ['gap_up_best_model','gap_down_best_model','5_cand_ABC_best_model','3_cand_ABC_best_model']


for strat in range(0,len(strategies)):
    print(strat)
    log_message = f"{strategies[strat]}\n"
    print(log_message)
    logging.info(log_message)

    Signal_df = final_Signal_df[final_Signal_df['Strategy'] == strategies[strat]]
    Signal_df.reset_index(inplace=True,drop=True)
    Signal_df = target_and_sl(Signal_df)

    logging.info(Signal_df.tail())

    final_signal_df = pd.DataFrame(columns=["Strategy", "Call_time", "Call", "stock", "Target", "SL", "achieved_ts", "points", "Value"])

    def check_final_calls(Signal_df):
        # final_signal_df = pd.DataFrame(columns=["Strategy", "Call_time", "Call", "stock", "Target", "SL", "achieved_ts", "points", "Value","Potential_Target","Potential_Stoploss"])
        final_signal_df = pd.DataFrame(columns=["Strategy", "Call_time", "Call", "stock", "Target", "SL", "achieved_ts", "points", "Value"])
        
        
        response_data = df
            
        response_data['DateTime'] = pd.to_datetime(response_data['DateTime'])
        final_data = response_data
        # print(final_data)
        final_data = final_data[["DateTime", "Open", "High", "Low", "Close", "Volume"]]
        stock_5_min_historic_data = final_data
        
        
        Signal_df.reset_index(drop=True, inplace=True)

        print(Signal_df.head())
        
        for i in range(Signal_df.shape[0]):
            stock = Signal_df.loc[i, "Stock"]
            call_time = Signal_df.loc[i, "Datetime"]
            call_time = pd.to_datetime(call_time)  # Convert to the same data type as Datetime column
            
            signal_val = Signal_df.loc[i, "Signal"]
            call_val = Signal_df.loc[i, "Value"]
            StopLoss = Signal_df.loc[i, "StopLoss"]
            Target = Signal_df.loc[i, "Target"]
            Strategy = Signal_df.loc[i, "Strategy"]
            current_date = Signal_df.loc[i, "Date"]
            current_signal = Signal_df.loc[i, "Signal"]
            # potential_target = Signal_df.loc[i, "Potential_Target"]
            # potential_stoploss = Signal_df.loc[i, "Potential_Stoploss"]
            
            # Convert 'Datetime' column to datetime if it's not already
            stock_5_min_historic_data['DateTime'] = pd.to_datetime(stock_5_min_historic_data['DateTime'])

            # Create new 'date' column
            stock_5_min_historic_data['date'] = stock_5_min_historic_data['DateTime'].dt.date

            # Sort dataframe by 'date' and assign dense rank
            stock_5_min_historic_data.sort_values('date', inplace=True)
            stock_5_min_historic_data['dns_rank'] = stock_5_min_historic_data['date'].rank(method='dense')

            final_5_min_stocks = stock_5_min_historic_data.copy()

            current_data = pd.DataFrame()
            
            eod_square_off = "no"
            
            if eod_square_off == "yes":
                current_data = final_5_min_stocks[final_5_min_stocks['date'] == current_date]

                sub_data = current_data[current_data['DateTime'] > call_time]

                sub_data = sub_data.sort_values('DateTime')
                
                satisfied_df = pd.DataFrame(columns=['Strategy', 'Call_time', 'Call', 'stock', 'Target', 'SL', 'achieved_ts', 'points', 'Value'])
                # satisfied_df = pd.DataFrame(columns=['Strategy', 'Call_time', 'Call', 'stock', 'Target', 'SL', 'achieved_ts', 'points', 'Value','Potential_Target','Potential_Stoploss'])

                if not sub_data.empty:
                    sub_data.reset_index(drop=True, inplace=True)

                    if signal_val == "Buy":
                        for j, row in sub_data.iterrows():
                            curr_datetime = row['DateTime']
                            # curr_datetime = row['DateTime'] - datetime.timedelta(hours=4)
                            # curr_datetime = curr_datetime.astimezone(pytz.timezone('America/New_York'))
                            curr_hr = curr_datetime.hour
                            curr_min = curr_datetime.minute
                            if row['High'] >= Target or row['Low'] <= StopLoss or (curr_hr == 15 and curr_min == 15):
                            # if row['High'] >= potential_target or row['Low'] <= potential_stoploss or (curr_hr == 15 and curr_min == 15):
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': 'Yes' if row['High'] >= Target else '', 
                                    'SL': 'Yes' if row['Low'] <= StopLoss else '',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['High'], 2), 
                                    'Value': call_val
                                }
                                # new_row = {
                                #     'Strategy': Strategy, 
                                #     'stock': stock, 
                                #     'Call_time': call_time, 
                                #     'Call': current_signal, 
                                #     'Target': 'Yes' if row['High'] >= potential_target else '', 
                                #     'SL': 'Yes' if row['Low'] <= potential_stoploss else '',
                                #     'achieved_ts': row['DateTime'], 
                                #     'points': round(abs(row['High'] - call_val), 2), 
                                #     'Value': round(row['High'], 2)
                                # }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break
                    else:
                        for j, row in sub_data.iterrows():
                            curr_datetime = row['DateTime']
                            # curr_datetime = row['DateTime'] - datetime.timedelta(hours=4)
                            # curr_datetime = curr_datetime.astimezone(pytz.timezone('America/New_York'))
                            curr_hr = curr_datetime.hour
                            curr_min = curr_datetime.minute
                            if row['Low'] <= Target or row['High'] >= StopLoss or (curr_hr == 15 and curr_min == 15):
                            # if row['Low'] <= potential_target or row['High'] >= potential_stoploss or (curr_hr == 15 and curr_min == 15):
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': 'Yes' if row['Low'] <= Target else '', 
                                    'SL': 'Yes' if row['High'] >= StopLoss else '',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['Low'], 2), 
                                    'Value': call_val
                                }
                                # new_row = {
                                #     'Strategy': Strategy, 
                                #     'stock': stock, 
                                #     'Call_time': call_time, 
                                #     'Call': current_signal, 
                                #     'Target': 'Yes' if row['Low'] <= potential_target else '', 
                                #     'SL': 'Yes' if row['High'] >= potential_stoploss else '',
                                #     'achieved_ts': row['DateTime'], 
                                #     'points': round(abs(row['Low'] - call_val), 2), 
                                #     'Value': round(row['Low'], 2)
                                # }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break

                    if not satisfied_df.empty:
                        satisfied_df = satisfied_df.head(1)
                        final_signal_df = pd.concat([final_signal_df, satisfied_df], ignore_index=True)
            else:
                current_data = final_5_min_stocks[final_5_min_stocks['date'] >= current_date]

                current_data['DateTime'] = current_data['DateTime'].dt.tz_localize(None)
                sub_data = current_data[current_data['DateTime'] > call_time]

                sub_data = sub_data.sort_values('DateTime')

                satisfied_df = pd.DataFrame(columns=['Strategy', 'Call_time', 'Call', 'stock', 'Target', 'SL', 'achieved_ts', 'points', 'Value'])

                if not sub_data.empty:
                    sub_data.reset_index(drop=True, inplace=True)

                    for j, row in sub_data.iterrows():
                        curr_datetime = row['DateTime']
                        # curr_datetime = row['DateTime'] - timedelta(hours=4)
                        # curr_datetime = curr_datetime.tz_localize('UTC')  # Localize to UTC (or another timezone) first
                        # curr_datetime = curr_datetime.astimezone(pytz.timezone('America/New_York'))
                        curr_hr = curr_datetime.hour
                        curr_min = curr_datetime.minute

                        if signal_val == "Buy":
                            if row['High'] >= Target:
                            # if row['High'] >= potential_target:
                                
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': 'Yes', 
                                    'SL': '',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['High'], 2), 
                                    'Value': call_val
                                }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break
                            elif row['Low'] <= StopLoss:
                            # elif row['Low'] <= potential_stoploss:
                                
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': '', 
                                    'SL': 'Yes',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['Low'], 2), 
                                    'Value': call_val
                                }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break

                        elif signal_val != "Buy":
                            if row['Low'] <= Target:
                            # if row['Low'] <= potential_target:
                                
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': 'Yes', 
                                    'SL': '',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['Low'], 2), 
                                    'Value': call_val
                                }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break
                            elif row['High'] >= StopLoss:
                            # elif row['High'] >= potential_stoploss:
                                
                                new_row = {
                                    'Strategy': Strategy, 
                                    'stock': stock, 
                                    'Call_time': call_time, 
                                    'Call': current_signal, 
                                    'Target': '', 
                                    'SL': 'Yes',
                                    'achieved_ts': row['DateTime'], 
                                    'points': round(row['High'], 2), 
                                    'Value': call_val
                                }
                                satisfied_df = satisfied_df.append(new_row, ignore_index=True)
                                break

                    if not satisfied_df.empty:
                        satisfied_df = satisfied_df.head(1)
                        final_signal_df = pd.concat([final_signal_df, satisfied_df], ignore_index=True)

        return final_signal_df


    if Signal_df.shape[0] > 0:
        final_signal_df = check_final_calls(Signal_df)

    print(final_signal_df.tail())
    logging.info(final_signal_df.tail()) 

    # Calculate hit rate
    num_successful_trades = final_signal_df[(final_signal_df['Call'] == 'Buy') & (final_signal_df['Target'] == 'Yes')].shape[0] + final_signal_df[(final_signal_df['Call'] == 'Sell') & (final_signal_df['SL'] == 'Yes')].shape[0]
    num_total_trades = final_signal_df.shape[0]
    hit_rate = num_successful_trades / num_total_trades

    # Calculate P&L
    final_signal_df['PnL'] = final_signal_df['points'] * np.where(final_signal_df['Call'] == 'Buy', 1, -1)
    total_pnl = final_signal_df['PnL'].sum()

    # Calculate monthly revenue
    final_signal_df['Month'] = final_signal_df['Call_time'].dt.to_period('M')
    monthly_revenue = final_signal_df.groupby('Month')['PnL'].sum()

    # Calculate max drawdown
    final_signal_df['Cumulative PnL'] = final_signal_df['PnL'].cumsum()
    final_signal_df['Running Max'] = final_signal_df['Cumulative PnL'].cummax()
    final_signal_df['Drawdown'] = final_signal_df['Running Max'] - final_signal_df['Cumulative PnL']
    max_drawdown = final_signal_df['Drawdown'].max()

    # Calculate day of the week performance
    final_signal_df['Day of Week'] = final_signal_df['Call_time'].dt.dayofweek
    day_of_week_performance = final_signal_df.groupby('Day of Week')['PnL'].sum()

    # Calculate performance of buy and sell signals
    performance_buy_signals = final_signal_df[final_signal_df['Call'] == 'Buy']['PnL'].sum()
    performance_sell_signals = final_signal_df[final_signal_df['Call'] == 'Sell']['PnL'].sum()

    # Calculate Sharpe ratio
    daily_returns = final_signal_df['PnL'] / final_signal_df['Value']
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

    # Display the metrics
    hit_rate, total_pnl, monthly_revenue, max_drawdown, day_of_week_performance, performance_buy_signals, performance_sell_signals, sharpe_ratio

    # Calculate the unique signals
    unique_signals = final_signal_df['Call'].unique()
    # Calculate the number of buy signals
    buy_signals = final_signal_df[final_signal_df['Call'] == 'Buy'].shape[0]
    # Calculate the number of sell signals
    sell_signals = final_signal_df[final_signal_df['Call'] == 'Sell'].shape[0]

    # Convert 'Call_time' to datetime if it's not already
    final_signal_df['Call_time'] = pd.to_datetime(final_signal_df['Call_time'])
    # Extract the hour
    final_signal_df['Hour'] = final_signal_df['Call_time'].dt.hour

    signals_by_hour = final_signal_df.groupby('Hour')['Call'].value_counts()
    unique_signals, buy_signals, sell_signals, signals_by_hour

    final_signal_df.set_index('Call_time', inplace=True)


    # Calculate the monthly revenue if investing 10K in the strategy
    investment = 10000
    final_signal_df['Monthly Return'] = final_signal_df['PnL'] * investment / final_signal_df['Value'].shift()
    monthly_revenue = final_signal_df['Monthly Return'].resample('M').sum()
    # Calculate the max drawdown
    final_signal_df['Cumulative Max'] = final_signal_df['Cumulative PnL'].cummax()
    drawdown = final_signal_df['Cumulative Max'] - final_signal_df['Cumulative PnL']
    max_drawdown = drawdown.max()
    # Calculate the performance of buy signals and sell signals
    buy_performance = final_signal_df[final_signal_df['Call'] == 'Buy']['PnL'].sum()
    sell_performance = final_signal_df[final_signal_df['Call'] == 'Sell']['PnL'].sum()
    # Calculate the Sharpe ratio
    sharpe_ratio = final_signal_df['PnL'].mean() / final_signal_df['PnL'].std() * np.sqrt(252)
    monthly_revenue, max_drawdown, buy_performance, sell_performance, sharpe_ratio

    # Convert DateTime column to datetime type
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Create a separate date column
    df['date'] = df['DateTime'].dt.date

    # Sort the dataframe by DateTime
    df.sort_values('DateTime', inplace=True)

    # Initialize Day_High_Till_Time and Day_Low_Till_Time columns
    df['Day_High_Till_Time'] = df.groupby('date')['High'].cummax().shift()
    df['Day_Low_Till_Time'] = df.groupby('date')['Low'].cummin().shift()

    # # Add your lagged features
    # ml_latest_merged_df['Last_1_Close'] = ml_latest_merged_df['Close'].shift(1)
    # ml_latest_merged_df['Last_2_Close'] = ml_latest_merged_df['Close'].shift(2)

    # Forward fill the missing values
    df['Day_High_Till_Time'].ffill(inplace=True)
    df['Day_Low_Till_Time'].ffill(inplace=True)

    # Convert NaN values to empty string
    df['Day_High_Till_Time'] = df['Day_High_Till_Time'].fillna('').astype(str)
    df['Day_Low_Till_Time'] = df['Day_Low_Till_Time'].fillna('').astype(str)

    df.loc[df['Day_High_Till_Time'] == '', 'Day_High_Till_Time'] = df['High']
    df.loc[df['Day_Low_Till_Time'] == '', 'Day_Low_Till_Time'] = df['Low']

    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df['date'] = df['DateTime'].dt.date

    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    df['rsi'] = df.ta.rsi(close='Close',length = 14)
    df['sma_20'] = ta.sma(df["Close"], length=20)
    df.ta.bbands(close = 'Close', length=20, std=2,append = True)
    df['SMA_Call'] = df.apply(lambda x: 'Buy' if x['Close'] >= x['sma_20'] else 'Sell', axis=1)
    df['RSI_Call'] = df.apply(lambda x: 'Buy' if x['rsi'] >= 60 else 'Sell' if x['rsi'] <=40 else 'Neutral', axis=1)
    df['MACD_Call'] = df.apply(lambda x: 'Buy' if x['MACD_12_26_9'] >= x['MACDs_12_26_9'] else 'Sell', axis=1)
    df['Pivot_Call'] = ''
    df['PCR_Call'] = ''

    # Calculate Stochastic Oscillator
    df.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)

    # Calculate Average True Range (ATR)
    df.ta.atr(high='High', low='Low', close='Close', length=14, append=True)

    # Calculate On-Balance Volume (OBV)
    df.ta.obv(close='Close', volume='Volume', append=True)

    def calculate_classic_pivots(data,idx):
        pivot_data = data.tail(1)
        # print(pivot_data)
        pivot_data.reset_index(level=0, inplace=True)
        pivot_point = (pivot_data.loc[0,'High'] + pivot_data.loc[0,'Low'] + pivot_data.loc[0,'Close'])/3

        pivot_bc = (pivot_data.loc[0,'High'] + pivot_data.loc[0,'Low'])/2
        pivot_tc = 2* pivot_point - pivot_bc

        # print(pivot_bc)
        # print(pivot_tc)
        
        classic_support_1 = round((2*pivot_point) - pivot_data.loc[0,'High'],2)
        
        classic_resistance_1 = round((2*pivot_point) - pivot_data.loc[0,'Low'],2)
            
        classic_support_2 = round(pivot_point - (classic_resistance_1 - classic_support_1),2)

        classic_resistance_2 = round((pivot_point - classic_support_1 ) + classic_resistance_1,2)

        classic_resistance_3 = round((pivot_point - classic_support_2 ) + classic_resistance_2,2)

        classic_support_3 = round(pivot_point - (classic_resistance_2 - classic_support_2),2)
        
        
        price_difference = (pivot_data.loc[0,'High'] - pivot_data.loc[0,'Low'])

        fibonnaci_resistance_1 = round((38.2*price_difference/100) + pivot_point,2)

        fibonnaci_resistance_2 = round((61.8*price_difference/100) + pivot_point,2)

        fibonnaci_resistance_3 = round((100*price_difference/100) + pivot_point,2)

        fibonnaci_support_1 = round(pivot_point - (38.2*price_difference/100),2)

        fibonnaci_support_2 = round(pivot_point - (61.8*price_difference/100),2)

        fibonnaci_support_3 = round(pivot_point - (100*price_difference/100),2)
        
        
        daily_levels_final_data.loc[idx,"Date"] = pivot_data.loc[0,'Date']
        daily_levels_final_data.loc[idx,"Open"] = pivot_data.loc[0,'Open']
        daily_levels_final_data.loc[idx,"High"] = pivot_data.loc[0,'High']
        daily_levels_final_data.loc[idx,"Low"] = pivot_data.loc[0,'Low']
        daily_levels_final_data.loc[idx,"Close"] = pivot_data.loc[0,'Close']
        
        daily_levels_final_data.loc[idx,"pivot_point"] = round(pivot_point,2)
        daily_levels_final_data.loc[idx,"pivot_bc"] = round(pivot_bc,2)
        daily_levels_final_data.loc[idx,"pivot_tc"] = round(pivot_tc,2)

        daily_levels_final_data.loc[idx,"classical_support_1"] = classic_support_1
        
        daily_levels_final_data.loc[idx,"classical_resistance_1"] = classic_resistance_1
            
        daily_levels_final_data.loc[idx,"classical_support_2"] = classic_support_2

        daily_levels_final_data.loc[idx,"classical_resistance_2"] = classic_resistance_2

        daily_levels_final_data.loc[idx,"classical_resistance_3"] = classic_resistance_3

        daily_levels_final_data.loc[idx,"classical_support_3"] = classic_support_3
        
        
        price_difference = (pivot_data.loc[0,'High'] - pivot_data.loc[0,'Low'])

        daily_levels_final_data.loc[idx,"fibonnaci_resistance_1"] = round((38.2*price_difference/100) + pivot_point,2)

        daily_levels_final_data.loc[idx,"fibonnaci_resistance_2"] = round((61.8*price_difference/100) + pivot_point,2)

        daily_levels_final_data.loc[idx,"fibonnaci_resistance_3"] = round((100*price_difference/100) + pivot_point,2)

        daily_levels_final_data.loc[idx,"fibonnaci_support_1"] = round(pivot_point - (38.2*price_difference/100),2)

        daily_levels_final_data.loc[idx,"fibonnaci_support_2"] = round(pivot_point - (61.8*price_difference/100),2)

        daily_levels_final_data.loc[idx,"fibonnaci_support_3"] = round(pivot_point - (100*price_difference/100),2)

    daily_levels_final_data = pd.DataFrame()

    for idx in range(0,len(stock_data)):
        calculate_classic_pivots(pd.DataFrame(stock_data.loc[idx,]).transpose(),idx)

    # Ensure the date columns in both dataframes are in datetime format
    df['date'] = pd.to_datetime(df['date'])
    daily_levels_final_data['Date'] = pd.to_datetime(daily_levels_final_data['Date'])

    # Merge the dataframes on the date columns
    merged_df = pd.merge(df, daily_levels_final_data, how='inner', left_on='date', right_on='Date')

    # Merge the dataframes on the date columns
    merged_df = pd.merge(merged_df, extracted_data, how='inner')

    # Optionally, you can drop the duplicate 'Date' column
    merged_df = merged_df.drop(['Date'], axis=1)

    merged_df = merged_df.rename(columns={'Open_x': 'Open', 'High_x':'High','Low_x':'Low','Close_x':'Close', 'Open_y': 'day_open', 'High_y': 'day_high', 'Low_y': 'day_low', 'Close_y': 'day_close'})

    merged_df['Pivot_Call'] = merged_df.apply(lambda x: 'Buy' if x['Close'] >= x['pivot_bc'] else 'Sell', axis=1)
    hist_df = merged_df[['DateTime','Open', 'High','Low', 'Close','Volume']]
    hist_df.set_index(pd.DatetimeIndex(hist_df["DateTime"]), inplace=True)
    hist_df.ta.vwap(high='High', low='Low',close='Close',volume='Volume', append=True)
    hist_df.ta.supertrend(high='High',low='Low',close='Close',append=True)
    hist_df.reset_index(inplace=True,drop=True)
    print(hist_df.tail(5))
    print(merged_df.tail(5))
    result = pd.merge(merged_df, hist_df, on="DateTime")
    result.reset_index(inplace=True,drop=True)  

    result = result[['DateTime', 'Open_x', 'High_x', 'Low_x', 'Close_x','Volume_x',
                    'Day_High_Till_Time', 'Day_Low_Till_Time', 
                    'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                    'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV','VWAP_D', 'SUPERT_7_3.0',
                    'pivot_point', 'pivot_bc', 'pivot_tc',
                    'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3',
                    'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                    's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                    's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                    's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                    's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                    'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                    'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                    'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                    'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                    'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                    'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                    'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                    'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                    's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                    'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                    'dow_jones_Log_Previous_1_Price_Movement',
                    'nasdaq_Log_Previous_1_Price_Movement'
                    ]]
    result.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 
                    'Day_High_Till_Time', 'Day_Low_Till_Time',
                    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20','BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
                    'SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV', 'VWAP_D','supertrend',
                    'pivot_point', 'pivot_bc', 'pivot_tc',
                    'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3',
                    'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                    's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                    's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                    's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                    's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                    'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                    'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                    'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                    'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                    'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                    'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                    'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                    'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                    's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                    'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                    'dow_jones_Log_Previous_1_Price_Movement',
                    'nasdaq_Log_Previous_1_Price_Movement']
    # print(result)
    result['VWAP_D'] = result['VWAP_D'].replace(np.nan, 0)
    result['supertrend'] = result['supertrend'].replace(np.nan, 0)

    result['BB_Call'] = result.apply(lambda x: 'Buy' if x['Close'] <= x['BBL_20_2.0'] else 'Sell' if x['Close'] >= x['BBU_20_2.0'] else 'Neutral', axis=1)
    result['VWAP_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['VWAP_D'] else 'Sell',axis = 1)
    result['SuperTrend_Call'] = result.apply(lambda x:'Buy' if x['Close'] >= x['supertrend'] else 'Sell',axis = 1)
    result['date'] = pd.to_datetime(result['DateTime'], format='%Y-%m-%d')

    result = result[[ 'DateTime', 'Open', 'High', 'Low', 'Close','Volume',
                 'Day_High_Till_Time', 'Day_Low_Till_Time',
                'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0', 
                'SMA_Call', 'RSI_Call', 'MACD_Call','Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call', 'SuperTrend_Call','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV', 'VWAP_D','supertrend',
                'pivot_point', 'pivot_bc', 'pivot_tc',
                'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3', 
                'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
                's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                'dow_jones_Log_Previous_1_Price_Movement',
                'nasdaq_Log_Previous_1_Price_Movement'
                ]]


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

    result['date'] = pd.to_datetime(result['DateTime'])
    result['date'] = result['date'].dt.date

    daily_levels_final_data['Date'] = pd.to_datetime(daily_levels_final_data['Date'])

    # Convert the 'Date' column in stock_data and 'DateTime' column in merged_df to datetime type
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Previous_1_Open'] = stock_data['Open'].shift(1)
    stock_data['Previous_1_High'] = stock_data['High'].shift(1)
    stock_data['Previous_1_Low'] = stock_data['Low'].shift(1)
    stock_data['Previous_1_Close'] = stock_data['Close'].shift(1)
    stock_data['Previous_1_Volume'] = stock_data['Volume'].shift(1)
    stock_data['Previous_2_Open'] = stock_data['Open'].shift(2)
    stock_data['Previous_2_High'] = stock_data['High'].shift(2)
    stock_data['Previous_2_Low'] = stock_data['Low'].shift(2)
    stock_data['Previous_2_Close'] = stock_data['Close'].shift(2)
    stock_data['Previous_2_Volume'] = stock_data['Volume'].shift(2)

    # Convert the 'date' column in 'result' dataframe to datetime type
    result['date'] = pd.to_datetime(result['date'])

    # Convert the 'Date' column in 'stock_data' dataframe to datetime type
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Merge the dataframes based on the date columns
    result = result.merge(stock_data, left_on='date', right_on='Date', how='left')

    result = result.rename(columns={'Open_x': 'Open', 'High_x':'High','Low_x':'Low','Close_x':'Close','Volume_x':'Volume'})

    result = result.drop(['Open_y', 'High_y','Low_y','Close_y','Volume_y'], axis=1)

    final_signal_df.reset_index(inplace=True)

    new_df = final_signal_df[['Call_time', 'Strategy', 'Call', 'stock', 'Value', 'points']].copy()
    new_df['Target_SL'] = final_signal_df['Target'].apply(lambda x: 1 if x == 'Yes' else 0) - final_signal_df['SL'].apply(lambda x: 1 if x == 'Yes' else 0)

    latest_merged_df = new_df.merge(result, left_on='Call_time', right_on='DateTime', how='inner')

    # Optionally, you can drop the duplicate 'Date' column
    latest_merged_df = latest_merged_df.drop(['DateTime','date','Date'], axis=1)

    # Merge the dataframes on the date columns
    latest_merged_df = pd.merge(latest_merged_df, df_pivot, how='inner',left_on="Call_time",right_on="DateTime")

    latest_merged_df = latest_merged_df.drop(['DateTime','Date'], axis=1)

    latest_merged_df = latest_merged_df[['Call_time','Open','High','Low','Close','Volume','Day_High_Till_Time', 'Day_Low_Till_Time','MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0','BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0','Previous_1_Open','Previous_1_High', 'Previous_1_Low', 'Previous_1_Close','Previous_1_Volume','Previous_2_Open', 'Previous_2_High', 'Previous_2_Low','Previous_2_Close','Previous_2_Volume','Call','Value','points','Target_SL','SMA_Call','RSI_Call','MACD_Call','Pivot_Call','PCR_Call','BB_Call','VWAP_Call','SuperTrend_Call','buy_probability','sell_probability','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV', 'VWAP_D','supertrend','pivot_point', 'pivot_bc', 'pivot_tc','classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3', 'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3', 
                                    's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
                                    's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
                                    's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
                                    's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
                                    'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
                                    'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
                                    'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
                                    'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
                                    'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
                                    'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
                                    'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
                                    'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
                                    's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
                                    'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
                                    'dow_jones_Log_Previous_1_Price_Movement',
                                    'nasdaq_Log_Previous_1_Price_Movement',
                                    'Open_Dow_Jones','Open_nasdaq','Open_s_and_p','High_Dow_Jones','High_nasdaq','High_s_and_p','Low_Dow_Jones','Low_nasdaq','Low_s_and_p',
                                    'Close_Dow_Jones','Close_nasdaq','Close_s_and_p'
                                   
                                   ]]

    latest_merged_df.replace({"Buy": 1, "Sell": -1, "Neutral": 0}, inplace=True)
    latest_merged_df['PCR_Call'].replace({"": 0})

    latest_merged_df['Day'] = latest_merged_df['Call_time'].dt.day
    latest_merged_df['Month'] = latest_merged_df['Call_time'].dt.month
    latest_merged_df['Year'] = latest_merged_df['Call_time'].dt.year
    latest_merged_df['Hour'] = latest_merged_df['Call_time'].dt.hour
    latest_merged_df['Minute'] = latest_merged_df['Call_time'].dt.minute

    # Calculate day of the week in numbers (0: Monday, 1: Tuesday, ..., 6: Sunday)
    latest_merged_df['DayOfWeek'] = latest_merged_df['Call_time'].dt.dayofweek
    # Calculate quarter
    latest_merged_df['Quarter'] = latest_merged_df['Call_time'].dt.quarter

    ml_latest_merged_df = latest_merged_df.copy()

    ml_latest_merged_df.drop('Call_time', axis=1, inplace=True)

    ml_latest_merged_df.drop('PCR_Call', axis=1, inplace=True)

    print(ml_latest_merged_df['Target_SL'].value_counts())

    log_message = f"{ml_latest_merged_df['Target_SL'].value_counts()}\n"
    logging.info(log_message)

    nan_counts = ml_latest_merged_df.isna().sum()
    # print(nan_counts)

    ml_latest_merged_df = ml_latest_merged_df.dropna()

    ml_latest_merged_df['Target_SL'] = ml_latest_merged_df['Target_SL'].replace(-1, 0)

    # Dropping duplicates and updating the DataFrame
    ml_latest_merged_df = ml_latest_merged_df.drop_duplicates()

    # Define the features and target
    # features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day_High_Till_Time', 'Day_Low_Till_Time',
    #         'Previous_1_Open', 'Previous_1_High', 'Previous_1_Low', 'Previous_1_Close', 'Previous_1_Volume',
    #         'Previous_2_Open', 'Previous_2_High', 'Previous_2_Low', 'Previous_2_Close', 'Previous_2_Volume',
    #         'Call', 'Value', 'points', 'SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call',
    #         'BB_Call', 'VWAP_Call', 'SuperTrend_Call', 'buy_probability',
    #         'sell_probability', 'Day', 'Month', 'Year', 'Hour', 'Minute','DayOfWeek','Quarter', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    #         'ATRr_14', 'OBV']

    features = ['Open', 'High', 'Low', 'Close', 'Volume','Day_High_Till_Time', 'Day_Low_Till_Time',
            'Previous_1_Open', 'Previous_1_High', 'Previous_1_Low','Previous_1_Close', 'Previous_1_Volume',
            'Previous_2_Open', 'Previous_2_High', 'Previous_2_Low', 'Previous_2_Close', 'Previous_2_Volume', 
            'Call', 'Value', 'points',
            'buy_probability', 'sell_probability',
            'MACD_12_26_9','MACDh_12_26_9', 'MACDs_12_26_9', 'rsi', 'sma_20', 'BBL_20_2.0',
            'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ATRr_14', 'OBV', 'VWAP_D', 'supertrend',
            'pivot_point', 'pivot_bc', 'pivot_tc',
            'classical_support_1', 'classical_resistance_1', 'classical_support_2','classical_resistance_2', 'classical_resistance_3','classical_support_3', 
            'fibonnaci_resistance_1','fibonnaci_resistance_2', 'fibonnaci_resistance_3','fibonnaci_support_1', 'fibonnaci_support_2', 'fibonnaci_support_3',
            'Day', 'Month', 'Year', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
            's_and_p_Previous_1_Open', 's_and_p_Previous_2_Open',
            's_and_p_Previous_1_High', 's_and_p_Previous_2_High',
            's_and_p_Previous_1_Low', 's_and_p_Previous_2_Low',
            's_and_p_Previous_1_Close', 's_and_p_Previous_2_Close',
            'dow_jones_Previous_1_Open', 'dow_jones_Previous_2_Open',
            'dow_jones_Previous_1_High', 'dow_jones_Previous_2_High',
            'dow_jones_Previous_1_Low', 'dow_jones_Previous_2_Low',
            'dow_jones_Previous_1_Close', 'dow_jones_Previous_2_Close',
            'nasdaq_Previous_1_Open', 'nasdaq_Previous_2_Open',
            'nasdaq_Previous_1_High', 'nasdaq_Previous_2_High',
            'nasdaq_Previous_1_Low', 'nasdaq_Previous_2_Low',
            'nasdaq_Previous_1_Close', 'nasdaq_Previous_2_Close',
            's_and_p_Prev_Percentage', 'dow_jones_Prev_Percentage',
            'nasdaq_Prev_Percentage', 's_and_p_Log_Previous_1_Price_Movement',
            'dow_jones_Log_Previous_1_Price_Movement',
            'nasdaq_Log_Previous_1_Price_Movement',
            'Open_Dow_Jones','Open_nasdaq','Open_s_and_p',
            'High_Dow_Jones','High_nasdaq','High_s_and_p',
            'Low_Dow_Jones','Low_nasdaq','Low_s_and_p',
            'Close_Dow_Jones','Close_nasdaq','Close_s_and_p'
            ]
    
    target = 'Target_SL'


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    logging.info("------- Running Logistic Regression Min Max Scalar-------")

    # Perform min-max scaling on the training and testing data
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    min_max_scalar_file = f'US_Stock_Models/{stock}/{scalar_names[strat]}_min_max_scalar.pkl'

    joblib.dump(min_max_scaler, min_max_scalar_file)

    # Create a logistic regression classifier
    logreg_classifier = LogisticRegression(solver='liblinear')

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    }

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=logreg_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_min_max_scalar.pkl'

    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)
    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.coef_[0]
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{str(feature)}: {str(importance)}\n"
        logging.info(log_message)

    # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_min_max_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_min_max_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)


    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "logistic_regression",
        'Scalar File': min_max_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Convert the dictionary into a pandas DataFrame
    ml_df = pd.DataFrame([ml_data])

    print(ml_df)


    logging.info("------- Running Logistic Regression Max Abs Scalar-------")

    # Perform min-max scaling on the training and testing data
    max_abs_scaler = MaxAbsScaler()
    X_train_scaled = max_abs_scaler.fit_transform(X_train)
    X_test_scaled = max_abs_scaler.transform(X_test)

    max_abs_scalar_file = f'US_Stock_Models/{stock}/{scalar_names[strat]}_max_abs_scalar.pkl'
    joblib.dump(max_abs_scaler, max_abs_scalar_file)

    # Create a logistic regression classifier
    logreg_classifier = LogisticRegression(solver='liblinear')

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10]
    }

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=logreg_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_max_abs_scalar.pkl'

    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report) 

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.coef_[0]
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{str(feature)}: {str(importance)}\n"
        logging.info(log_message)

    
    # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_max_abs_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_logistic_regression_max_abs_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "logistic_regression",
        'Scalar File': max_abs_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)


    # logging.info("------- Running Random Forest -------")

    # # Calculate correlation matrix
    # correlation_matrix = ml_latest_merged_df[features].corr().abs()

    # # Set a correlation threshold
    # correlation_threshold = 0.8

    # # Create a mask to filter highly correlated features
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # correlation_filtered = correlation_matrix.mask(mask)

    # # Get the indices of features to keep
    # indices_to_keep = np.where(correlation_filtered < correlation_threshold)

    # # Filter the features based on the correlation threshold
    # selected_features = list(set([features[i] for i in indices_to_keep[1]]))

    # print(selected_features)
    # logging.info("selected features")
    # logging.info(selected_features)

    # # # Split the data into training and testing sets
    # # X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
    # #                                                     ml_latest_merged_df[target],
    # #                                                     test_size=0.2,
    # #                                                     random_state=42)

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[selected_features],
    #                                                     ml_latest_merged_df[target],
    #                                                     test_size=0.2,
    #                                                     random_state=42)

    # # Perform min-max scaling on the training and testing data
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Save the scaler
    # scalar_file = f'US_Stock_Models/{stock}/{scalar_names[strat]}_rf.pkl'
    # print(scalar_file)
    # joblib.dump(scaler, scalar_file)

    # # Save the selected features
    # selected_features_file = f'US_Stock_Models/{stock}/{scalar_names[strat]}_selected_features_rf.pkl'
    # joblib.dump(selected_features, selected_features_file)


    # # Define the parameter grid for hyperparameter tuning
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    # # Create a Random Forest classifier
    # rf_classifier = RandomForestClassifier()

    # # Perform grid search with stratified K-fold cross-validation
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)

    # # Get the best parameters and best model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print("Best Parameters:", best_params)
    # logging.info("Best Parameters: %s", best_params)

    # model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf.pkl'
    # print(model_file)
    # joblib.dump(best_model, model_file)
    # # Predict on the testing data using the best model
    # y_pred = best_model.predict(X_test_scaled)

    # # Print the classification report
    # report = classification_report(y_test, y_pred)
    # print("Test Classification Report:\n", report)

    # logging.info("Classification Report:\n")
    # logging.info(report)  

    # # Evaluate the accuracy of the best model
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    # logging.info("Test Accuracy:\n")
    # logging.info(accuracy)  

    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # sorted_indices = np.argsort(feature_importances)[::-1]
    # sorted_features = np.array(selected_features)[sorted_indices]

    # print("Feature Importances:")
    # for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
    #     print(feature, ":", importance)
    #     log_message = f"{feature}: {importance}\n"
    #     logging.info(log_message)

    logging.info("------- Running Random Forest Min Max Scalar-------")


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform min-max scaling on the training and testing data
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)


    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

    # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)


    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "random_forest",
        'Scalar File': min_max_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)


    # logging.info("------- Running Random Forest Min Max Scalar with RFE -------")

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
    #                                                     ml_latest_merged_df[target],
    #                                                     test_size=0.2,
    #                                                     random_state=42)

    # # Perform min-max scaling on the training and testing data
    # min_max_scaler = MinMaxScaler()
    # X_train_scaled = min_max_scaler.fit_transform(X_train)
    # X_test_scaled = min_max_scaler.transform(X_test)

    # # Define the parameter grid for hyperparameter tuning
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    # # Create a Random Forest classifier
    # rf_classifier = RandomForestClassifier()

    # # Perform grid search with stratified K-fold cross-validation
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)

    # # Get the best parameters and best model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print("Best Parameters:", best_params)
    # logging.info("Best Parameters: %s", best_params)
    # model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar.pkl'
    # print(model_file)
    # joblib.dump(best_model, model_file)

    # # Perform Recursive Feature Elimination with cross-validation
    # rfe = RFECV(estimator=best_model, cv=cv)
    # X_train_scaled_rfe = rfe.fit_transform(X_train_scaled, y_train)
    # X_test_scaled_rfe = rfe.transform(X_test_scaled)

    # # Get the selected feature indices
    # selected_feature_indices = rfe.get_support(indices=True)
    # selected_features = np.array(features)[selected_feature_indices]

    # # Fit the model on the selected features
    # best_model.fit(X_train_scaled_rfe, y_train)

    # # Predict on the training data using the best model
    # y_train_pred = best_model.predict(X_train_scaled_rfe)

    # # Predict on the testing data using the best model
    # y_test_pred = best_model.predict(X_test_scaled_rfe)

    # # Print the classification report for training data
    # train_report = classification_report(y_train, y_train_pred)
    # print("Training Classification Report:\n", train_report)

    # # Print the classification report
    # report = classification_report(y_test, y_test_pred)
    # print("Test Classification Report:\n", report)

    # logging.info("Classification Report:\n")
    # logging.info(report)

    # # Evaluate the accuracy of the best model
    # train_accuracy = accuracy_score(y_train, y_train_pred)
    # print("Train Accuracy:", train_accuracy)

    # # Evaluate the accuracy of the best model
    # test_accuracy = accuracy_score(y_test, y_test_pred)
    # print("Test Accuracy:", test_accuracy)

    # # logging.info("Accuracy:\n")
    # # logging.info(accuracy)  

    # # Calculate training precision
    # train_precision = precision_score(y_train, y_train_pred)
    # print("Training Precision:", train_precision)

    # # Calculate testing precision
    # test_precision = precision_score(y_test, y_test_pred)
    # print("Testing Precision:", test_precision)

    # # Calculate training recall
    # train_recall = recall_score(y_train, y_train_pred)
    # print("Training Recall:", train_recall)

    # # Calculate testing recall
    # test_recall = recall_score(y_test, y_test_pred)
    # print("Testing Recall:", test_recall)

    # # Calculate training AUC
    # train_auc = roc_auc_score(y_train, y_train_pred)
    # print("Training AUC:", train_auc)

    # # Calculate testing AUC
    # test_auc = roc_auc_score(y_test, y_test_pred)
    # print("Testing AUC:", test_auc)

    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # selected_feature_importances = feature_importances[selected_feature_indices]
    # sorted_indices = np.argsort(selected_feature_importances)[::-1]
    # sorted_features = selected_features[sorted_indices]

    # print("Feature Importances:")
    # for feature, importance in zip(sorted_features, selected_feature_importances[sorted_indices]):
    #     print(feature, ":", importance)
    #     log_message = f"{feature}: {importance}\n"
    #     logging.info(log_message)
    
    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # selected_feature_importances = [feature_importances[i] for i in selected_feature_indices if i < len(feature_importances)]
    # sorted_indices = np.argsort(selected_feature_importances)[::-1]
    # sorted_features = selected_features[sorted_indices]

    # # Create a dictionary to store the feature importances
    # feature_importance_dict = {
    #     "Feature": sorted_features.tolist(),
    #     "Importance": selected_feature_importances[sorted_indices].tolist()
    # }

    

    # # Add prediction and probability columns to the training dataset
    # X_train_with_preds = X_train.copy()
    # X_train_with_preds['Prediction'] = y_train_pred
    # X_train_with_preds['Target'] = y_train
    # train_probabilities = best_model.predict_proba(X_train_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # # Add prediction and probability columns to the testing dataset
    # X_test_with_preds = X_test.copy()
    # X_test_with_preds['Prediction'] = y_test_pred
    # X_test_with_preds['Target'] = y_test
    # test_probabilities = best_model.predict_proba(X_test_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    # training_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar_rfe_training_with_predictions.csv'
    # testing_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_min_max_scalar_rfe_testing_with_predictions.csv'

    # # Save the datasets with predictions and probabilities to CSV files
    # X_train_with_preds.to_csv(training_file, index=False)
    # X_test_with_preds.to_csv(testing_file, index=False)

    # # Create a dictionary with the required fields and their corresponding values
    # ml_data = {
    #     'Stock': stock,
    #     'Strategy': model_names[strat],
    #     'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
    #     'Model name': "random_forest_rfe",
    #     'Scalar File': min_max_scalar_file,
    #     'Model name': model_file,
    #     'Training Data Length': len(X_train),
    #     'Training Data File name': training_file,
    #     'Testing Data Length': len(X_test),
    #     'Test Data File name': testing_file,
    #     'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
    #     'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
    #     'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
    #     'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
    #     'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
    #     'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
    #     'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
    #     'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
    #     'comments - the target and stop loss kept': "Set 1:1",
    #     'Training Accuracy': train_accuracy,
    #     'Testing Accuracy': test_accuracy,
    #     'Training Precision': train_precision,
    #     'Testing Precision': test_precision,
    #     'Training Recall': train_recall,
    #     'Testing Recall': test_recall,
    #     'Training AUC': train_auc,
    #     'Testing AUC': test_auc,
    #     'Top Features': json.dumps(feature_importance_dict)
    # }

    # # Concatenating the new row with ml_df
    # ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    # print(ml_df)

    logging.info("------- Running Random Forest Max Abs Scalar-------")


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform min-max scaling on the training and testing data
    max_abs_scaler = MaxAbsScaler()
    X_train_scaled = max_abs_scaler.fit_transform(X_train)
    X_test_scaled = max_abs_scaler.transform(X_test)


    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

    # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)


    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "random_forest",
        'Scalar File': max_abs_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)


    # logging.info("------- Running Random Forest Max Abs Scalar with RFE -------")

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
    #                                                     ml_latest_merged_df[target],
    #                                                     test_size=0.2,
    #                                                     random_state=42)

    # # Perform max-abs scaling on the training and testing data
    # max_abs_scaler = MaxAbsScaler()
    # X_train_scaled = max_abs_scaler.fit_transform(X_train)
    # X_test_scaled = max_abs_scaler.transform(X_test)

    # # Define the parameter grid for hyperparameter tuning
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }

    # # Create a Random Forest classifier
    # rf_classifier = RandomForestClassifier()

    # # Perform grid search with stratified K-fold cross-validation
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)

    # # Get the best parameters and best model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print("Best Parameters:", best_params)
    # logging.info("Best Parameters: %s", best_params)
    # model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar.pkl'
    # print(model_file)
    # joblib.dump(best_model, model_file)

    # # Perform Recursive Feature Elimination with cross-validation
    # rfe = RFECV(estimator=best_model, cv=cv)
    # X_train_scaled_rfe = rfe.fit_transform(X_train_scaled, y_train)
    # X_test_scaled_rfe = rfe.transform(X_test_scaled)

    # # Get the selected feature indices
    # selected_feature_indices = rfe.get_support(indices=True)
    # selected_features = np.array(features)[selected_feature_indices]

    # # Fit the model on the selected features
    # best_model.fit(X_train_scaled_rfe, y_train)

    # # Predict on the training data using the best model
    # y_train_pred = best_model.predict(X_train_scaled_rfe)

    # # Predict on the testing data using the best model
    # y_test_pred = best_model.predict(X_test_scaled_rfe)

    # # Print the classification report for training data
    # train_report = classification_report(y_train, y_train_pred)
    # print("Training Classification Report:\n", train_report)

    # # Print the classification report
    # report = classification_report(y_test, y_test_pred)
    # print("Test Classification Report:\n", report)

    # logging.info("Classification Report:\n")
    # logging.info(report)

    # # Evaluate the accuracy of the best model
    # train_accuracy = accuracy_score(y_train, y_train_pred)
    # print("Train Accuracy:", train_accuracy)

    # # Evaluate the accuracy of the best model
    # test_accuracy = accuracy_score(y_test, y_test_pred)
    # print("Test Accuracy:", test_accuracy)

    # # logging.info("Accuracy:\n")
    # # logging.info(accuracy)  

    # # Calculate training precision
    # train_precision = precision_score(y_train, y_train_pred)
    # print("Training Precision:", train_precision)

    # # Calculate testing precision
    # test_precision = precision_score(y_test, y_test_pred)
    # print("Testing Precision:", test_precision)

    # # Calculate training recall
    # train_recall = recall_score(y_train, y_train_pred)
    # print("Training Recall:", train_recall)

    # # Calculate testing recall
    # test_recall = recall_score(y_test, y_test_pred)
    # print("Testing Recall:", test_recall)

    # # Calculate training AUC
    # train_auc = roc_auc_score(y_train, y_train_pred)
    # print("Training AUC:", train_auc)

    # # Calculate testing AUC
    # test_auc = roc_auc_score(y_test, y_test_pred)
    # print("Testing AUC:", test_auc)

    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # selected_feature_importances = feature_importances[selected_feature_indices]
    # sorted_indices = np.argsort(selected_feature_importances)[::-1]
    # sorted_features = selected_features[sorted_indices]

    # print("Feature Importances:")
    # for feature, importance in zip(sorted_features, selected_feature_importances[sorted_indices]):
    #     print(feature, ":", importance)
    #     log_message = f"{feature}: {importance}\n"
    #     logging.info(log_message)

    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # selected_feature_importances = [feature_importances[i] for i in selected_feature_indices if i < len(feature_importances)]
    # sorted_indices = np.argsort(selected_feature_importances)[::-1]
    # sorted_features = selected_features[sorted_indices]

    # # Create a dictionary to store the feature importances
    # feature_importance_dict = {
    #     "Feature": sorted_features.tolist(),
    #     "Importance": selected_feature_importances[sorted_indices].tolist()
    # }

    # # Add prediction and probability columns to the training dataset
    # X_train_with_preds = X_train.copy()
    # X_train_with_preds['Prediction'] = y_train_pred
    # X_train_with_preds['Target'] = y_train
    # train_probabilities = best_model.predict_proba(X_train_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # # Add prediction and probability columns to the testing dataset
    # X_test_with_preds = X_test.copy()
    # X_test_with_preds['Prediction'] = y_test_pred
    # X_test_with_preds['Target'] = y_test
    # test_probabilities = best_model.predict_proba(X_test_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    # training_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar_rfe_training_with_predictions.csv'
    # testing_file = f'US_Stock_Models/{stock}/{model_names[strat]}_rf_max_abs_scalar_rfe_testing_with_predictions.csv'

    # # Save the datasets with predictions and probabilities to CSV files
    # X_train_with_preds.to_csv(training_file, index=False)
    # X_test_with_preds.to_csv(testing_file, index=False)

    # # Create a dictionary with the required fields and their corresponding values
    # ml_data = {
    #     'Stock': stock,
    #     'Strategy': model_names[strat],
    #     'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
    #     'Model name': "random_forest_rfe",
    #     'Scalar File': max_abs_scalar_file,
    #     'Model name': model_file,
    #     'Training Data Length': len(X_train),
    #     'Training Data File name': training_file,
    #     'Testing Data Length': len(X_test),
    #     'Test Data File name': testing_file,
    #     'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
    #     'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
    #     'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
    #     'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
    #     'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
    #     'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
    #     'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
    #     'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
    #     'comments - the target and stop loss kept': "Set 1:1",
    #     'Training Accuracy': train_accuracy,
    #     'Testing Accuracy': test_accuracy,
    #     'Training Precision': train_precision,
    #     'Testing Precision': test_precision,
    #     'Training Recall': train_recall,
    #     'Testing Recall': test_recall,
    #     'Training AUC': train_auc,
    #     'Testing AUC': test_auc,
    #     'Top Features': json.dumps(feature_importance_dict)
    # }

    # # Concatenating the new row with ml_df
    # ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    # print(ml_df)

    logging.info("------- Running XGBoost Model Min Max Scalar -------")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform min-max scaling on the training and testing data
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create an XGBoost classifier
    xgb_classifier = XGBClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

     # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "xg_boost",
        'Scalar File': min_max_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)


    # logging.info("------- Running XGBoost Model Min Max Scalar with RFE -------")

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
    #                                                     ml_latest_merged_df[target],
    #                                                     test_size=0.2,
    #                                                     random_state=42)

    # # Perform min-max scaling on the training and testing data
    # min_max_scaler = MinMaxScaler()
    # X_train_scaled = min_max_scaler.fit_transform(X_train)
    # X_test_scaled = min_max_scaler.transform(X_test)

    # # Define the parameter grid for hyperparameter tuning
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [3, 5, 7],
    #     'learning_rate': [0.1, 0.01, 0.001],
    #     'subsample': [0.8, 0.9, 1.0],
    #     'colsample_bytree': [0.8, 0.9, 1.0]
    # }

    # # Create an XGBoost classifier
    # xgb_classifier = XGBClassifier()

    # # Perform grid search with stratified K-fold cross-validation
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    # grid_search.fit(X_train_scaled, y_train)

    # # Get the best parameters and best model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    # print("Best Parameters:", best_params)
    # logging.info("Best Parameters: %s", best_params)
    # model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar.pkl'
    # print(model_file)
    # joblib.dump(best_model, model_file)

    # # Perform Recursive Feature Elimination with cross-validation
    # rfe = RFECV(estimator=best_model, cv=cv)
    # X_train_scaled_rfe = rfe.fit_transform(X_train_scaled, y_train)
    # X_test_scaled_rfe = rfe.transform(X_test_scaled)

    # # Get the selected feature indices
    # selected_feature_indices = rfe.get_support(indices=True)
    # selected_features = np.array(features)[selected_feature_indices]

    # # Fit the model on the selected features
    # best_model.fit(X_train_scaled_rfe, y_train)

    # # Predict on the training data using the best model
    # y_train_pred = best_model.predict(X_train_scaled_rfe)

    # # Predict on the testing data using the best model
    # y_test_pred = best_model.predict(X_test_scaled_rfe)

    # # Print the classification report for training data
    # train_report = classification_report(y_train, y_train_pred)
    # print("Training Classification Report:\n", train_report)

    # # Print the classification report
    # report = classification_report(y_test, y_test_pred)
    # print("Test Classification Report:\n", report)

    # logging.info("Classification Report:\n")
    # logging.info(report)

    # # Evaluate the accuracy of the best model
    # accuracy = accuracy_score(y_test, y_test_pred)
    # print("Accuracy:", accuracy)

    # logging.info("Test Accuracy:\n")
    # logging.info(accuracy)

    # # Get feature importances
    # feature_importances = best_model.feature_importances_
    # selected_feature_importances = feature_importances[selected_feature_indices]
    # sorted_indices = np.argsort(selected_feature_importances)[::-1]
    # sorted_features = selected_features[sorted_indices]

    # print("Feature Importances:")
    # for feature, importance in zip(sorted_features, selected_feature_importances[sorted_indices]):
    #     print(feature, ":", importance)
    #     log_message = f"{str(feature)}: {str(importance)}\n"
    #     logging.info(log_message)

    # # Add prediction and probability columns to the training dataset
    # X_train_with_preds = X_train.copy()
    # X_train_with_preds['Prediction'] = y_train_pred
    # X_train_with_preds['Target'] = y_train
    # train_probabilities = best_model.predict_proba(X_train_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # # Add prediction and probability columns to the testing dataset
    # X_test_with_preds = X_test.copy()
    # X_test_with_preds['Prediction'] = y_test_pred
    # X_test_with_preds['Target'] = y_test
    # test_probabilities = best_model.predict_proba(X_test_scaled_rfe)
    # for i, class_name in enumerate(best_model.classes_):
    #     X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    # training_file = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar_rfe_training_with_predictions.csv'
    # testing_file = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_min_max_scalar_rfe_testing_with_predictions.csv'

    # # Save the datasets with predictions and probabilities to CSV files
    # X_train_with_preds.to_csv(training_file, index=False)
    # X_test_with_preds.to_csv(testing_file, index=False)

    logging.info("------- Running XGBoost Model Max Abs Scalar -------")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform min-max scaling on the training and testing data
    max_abs_scaler = MaxAbsScaler()
    X_train_scaled = max_abs_scaler.fit_transform(X_train)
    X_test_scaled = max_abs_scaler.transform(X_test)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create an XGBoost classifier
    xgb_classifier = XGBClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_max_abs_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

     # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_max_abs_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_xg_boost_max_abs_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "xg_boost",
        'Scalar File': max_abs_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)

    logging.info("------- Running LightGBM Model Min Max Scalar-------")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform min-max scaling on the training and testing data
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create a LightGBM classifier
    lgb_classifier = LGBMClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_min_max_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

     # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_min_max_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_min_max_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "light_gbm",
        'Scalar File': min_max_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)

    logging.info("------- Running LightGBM Model Max Abs Scalar-------")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(ml_latest_merged_df[features],
                                                        ml_latest_merged_df[target],
                                                        test_size=0.2,
                                                        random_state=42)

    # Perform MaxAbsScaler scaling on the training and testing data
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Create a LightGBM classifier
    lgb_classifier = LGBMClassifier()

    # Perform grid search with stratified K-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=lgb_classifier, param_grid=param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print("Best Parameters:", best_params)
    logging.info("Best Parameters: %s", best_params)
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_max_abs_scalar.pkl'
    print(model_file)
    joblib.dump(best_model, model_file)

    # Predict on the training data using the best model
    y_train_pred = best_model.predict(X_train_scaled)

    # Predict on the testing data using the best model
    y_test_pred = best_model.predict(X_test_scaled)

    # Print the classification report for training data
    train_report = classification_report(y_train, y_train_pred)
    print("Training Classification Report:\n", train_report)

    # Print the classification report
    report = classification_report(y_test, y_test_pred)
    print("Test Classification Report:\n", report)

    logging.info("Classification Report:\n")
    logging.info(report)  

    # Evaluate the accuracy of the best model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Train Accuracy:", train_accuracy)

    # Evaluate the accuracy of the best model
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test Accuracy:", test_accuracy)

    # logging.info("Accuracy:\n")
    # logging.info(accuracy)  

    # Calculate training precision
    train_precision = precision_score(y_train, y_train_pred)
    print("Training Precision:", train_precision)

    # Calculate testing precision
    test_precision = precision_score(y_test, y_test_pred)
    print("Testing Precision:", test_precision)

    # Calculate training recall
    train_recall = recall_score(y_train, y_train_pred)
    print("Training Recall:", train_recall)

    # Calculate testing recall
    test_recall = recall_score(y_test, y_test_pred)
    print("Testing Recall:", test_recall)

    # Calculate training AUC
    train_auc = roc_auc_score(y_train, y_train_pred)
    print("Training AUC:", train_auc)

    # Calculate testing AUC
    test_auc = roc_auc_score(y_test, y_test_pred)
    print("Testing AUC:", test_auc)

    # Get feature importances
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_features = np.array(features)[sorted_indices]

    # Create a dictionary to store the feature importances
    feature_importance_dict = {
        "Feature": sorted_features.tolist(),
        "Importance": feature_importances[sorted_indices].tolist()
    }

    print("Feature Importances:")
    for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
        print(feature, ":", importance)
        log_message = f"{feature}: {importance}\n"
        logging.info(log_message)

     # Add prediction and probability columns to the training dataset
    X_train_with_preds = X_train.copy()
    X_train_with_preds['Prediction'] = y_train_pred
    X_train_with_preds['Target'] = y_train
    train_probabilities = best_model.predict_proba(X_train_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_train_with_preds[f'Probability_{class_name}'] = train_probabilities[:, i]

    # Add prediction and probability columns to the testing dataset
    X_test_with_preds = X_test.copy()
    X_test_with_preds['Prediction'] = y_test_pred
    X_test_with_preds['Target'] = y_test
    test_probabilities = best_model.predict_proba(X_test_scaled)
    for i, class_name in enumerate(best_model.classes_):
        X_test_with_preds[f'Probability_{class_name}'] = test_probabilities[:, i]

    training_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_max_abs_scalar_training_with_predictions.csv'
    testing_file  = f'US_Stock_Models/{stock}/{model_names[strat]}_light_gbm_max_abs_scalar_testing_with_predictions.csv'

    # Save the datasets with predictions and probabilities to CSV files
    X_train_with_preds.to_csv(training_file, index=False)
    X_test_with_preds.to_csv(testing_file, index=False)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "light_gbm",
        'Scalar File': max_abs_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': y_train.value_counts().get(1, 0),
        'Training Distribution of 0 Stoploss hit': y_train.value_counts().get(0, 0),
        'Testing Distribution of 1 Target hit': y_test.value_counts().get(1, 0),
        'Testing Distribution of 0 Stoploss Hit': y_test.value_counts().get(0, 0),
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': json.dumps(feature_importance_dict)
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)




    logging.info("------- Running CNN + GRU Model -------")

    # # Define the features and target
    # features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day_High_Till_Time', 'Day_Low_Till_Time',
    #             'Previous_1_Open', 'Previous_1_High', 'Previous_1_Low', 'Previous_1_Close', 'Previous_1_Volume',
    #             'Previous_2_Open', 'Previous_2_High', 'Previous_2_Low', 'Previous_2_Close', 'Previous_2_Volume',
    #             'Call', 'Value', 'points', 'SMA_Call', 'RSI_Call', 'MACD_Call', 'Pivot_Call',
    #             'BB_Call', 'VWAP_Call', 'SuperTrend_Call', 'buy_probability',
    #             'sell_probability', 'Day', 'Month', 'Year', 'Hour', 'Minute','DayOfWeek','Quarter', 'STOCHk_14_3_3', 'STOCHd_14_3_3',
    #             'ATRr_14', 'OBV']
    # target = 'Target_SL'

    # Split the data into features and target variables
    X = ml_latest_merged_df[features].values
    y = ml_latest_merged_df[target].values

    # Perform min-max scaling on the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    min_max_scalar_file = f'US_Stock_Models/{stock}/{scalar_names[strat]}_cnn.pkl'
    print(min_max_scalar_file)
    joblib.dump(X_scaled, min_max_scalar_file)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    X_train_unscaled = scaler.inverse_transform(X_train)
    X_test_unscaled = scaler.inverse_transform(X_test)

    X_train_unscaled = pd.DataFrame(X_train_unscaled, columns=features)
    X_test_unscaled = pd.DataFrame(X_test_unscaled, columns=features)

    # print(X_train_unscaled.head())
    # print(X_train_unscaled.columns)
    # # y_train_unscaled = scaler.inverse_transform(y_train)
    # # y_test_unscaled = scaler.inverse_transform(y_test)

    # Calculate sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Define the CNN+GRU model architecture
    def create_model(optimizer=Adam()):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(64))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])
        return model

    # Create KerasClassifier wrapper for scikit-learn compatibility
    model = KerasClassifier(build_fn=create_model)

    # Define hyperparameters for tuning
    param_grid = {
        'epochs': [50, 100],
        'batch_size': [32, 64],
    }

    # Create GridSearchCV instance
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc')

    # Train the model with hyperparameter tuning
    grid_search.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, sample_weight=sample_weights)

    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_.model

    logging.info("Best Parameters: %s", best_params)

    # Save the model
    model_file = f'US_Stock_Models/{stock}/{model_names[strat]}_cnn.h5'
    best_model.save(model_file)

    # Predict on the testing data
    y_pred_prob = best_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Print the classification report
    report = classification_report(y_test, y_pred)
    print("Test Classification Report:\n", report)

    # Calculate training and testing accuracy
    y_train_pred = best_model.predict(X_train.reshape(X_train.shape[0], X_train.shape[1], 1))
    train_accuracy = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Calculate training and testing precision, recall, and AUC
    train_precision = precision_score(y_train, (y_train_pred > 0.5).astype(int))
    test_precision = precision_score(y_test, y_pred)
    train_recall = recall_score(y_train, (y_train_pred > 0.5).astype(int))
    test_recall = recall_score(y_test, y_pred)
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_pred)

    print("Training Precision:", train_precision)
    print("Testing Precision:", test_precision)
    print("Training Recall:", train_recall)
    print("Testing Recall:", test_recall)
    print("Training AUC:", train_auc)
    print("Testing AUC:", test_auc)

    # # Reshape X_test to be 2D
    # X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # # Calculate permutation feature importances
    # perm_importance = permutation_importance(best_model, X_test_2d, y_test, scoring=None)

    # # Get feature importances
    # feature_importances = perm_importance.importances_mean
    # sorted_indices = np.argsort(feature_importances)[::-1]
    # sorted_features = np.array(features)[sorted_indices]

    # # Create a dictionary to store the feature importances
    # feature_importance_dict = {
    #     "Feature": sorted_features.tolist(),
    #     "Importance": feature_importances[sorted_indices].tolist()
    # }

    # print("Permutation Feature Importances:")
    # for feature, importance in zip(sorted_features, feature_importances[sorted_indices]):
    #     print(feature, ":", importance)

    # Create a dictionary with the required fields and their corresponding values
    ml_data = {
        'Stock': stock,
        'Strategy': model_names[strat],
        'Latest Run time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # current time
        'Model name': "cnn_gru",
        'Scalar File': min_max_scalar_file,
        'Model name': model_file,
        'Training Data Length': len(X_train),
        'Training Data File name': training_file,
        'Testing Data Length': len(X_test),
        'Test Data File name': testing_file,
        'Training distribution of 1 Signals': X_train_unscaled['Call'].value_counts().get(1, 0),
        'Training distribution of -1 Signals': X_train_unscaled['Call'].value_counts().get(-1, 0),
        'Testing distribution of 1 Signals': X_test_unscaled['Call'].value_counts().get(1, 0),
        'Testing distribution of -1 Signals': X_test_unscaled['Call'].value_counts().get(-1, 0),
        'Training Distribution of 1 Target hit': '',
        'Training Distribution of 0 Stoploss hit': '',
        'Testing Distribution of 1 Target hit': '',
        'Testing Distribution of 0 Stoploss Hit': '',
        'comments - the target and stop loss kept': "Set 1:1",
        'Training Accuracy': train_accuracy,
        'Testing Accuracy': test_accuracy,
        'Training Precision': train_precision,
        'Testing Precision': test_precision,
        'Training Recall': train_recall,
        'Testing Recall': test_recall,
        'Training AUC': train_auc,
        'Testing AUC': test_auc,
        'Top Features': ''
    }

    # Concatenating the new row with ml_df
    ml_df = pd.concat([ml_df, pd.DataFrame([ml_data])], ignore_index=True)

    print(ml_df)

    # MongoDB connection details
    username = 'Titania'
    password = 'Mahadev'
    cluster_url = 'cluster0.igqnlsy.mongodb.net'
    database_name = 'Backtesting'
    collection_name = 'machine_learning_backtest'

    # MongoDB connection URI
    uri = f"mongodb+srv://{username}:{password}@{cluster_url}/{database_name}?retryWrites=true&w=majority"

    # Create a MongoClient instance
    client = MongoClient(uri)

    # Access the desired database and collection
    db = client[database_name]
    collection = db[collection_name]

    # Convert DataFrame to a list of dictionaries (each row will be a dictionary)
    records = ml_df.to_dict(orient='records')

    # Insert the list of dictionaries (rows) into the collection
    result = collection.insert_many(records)

    # Output the result to check if the insert was successful
    print(f"Inserted {len(result.inserted_ids)} documents into the '{collection_name}' collection.")