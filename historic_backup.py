import logging
from datetime import datetime
from pytz import timezone
import pandas as pd
from pandas.io import gbq
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pandas_gbq
from google.oauth2 import service_account


server_api = ServerApi('1')
client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)
credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/client_secret.json',
)
db = client.titania_trading
db = client["titania_trading"]
list_of_collections = db.list_collection_names()

start_time = datetime.now(timezone("Asia/Kolkata"))


def upload_data_gbq(data,destination_table,replace_type):
    try:
        data.to_gbq(destination_table=destination_table,
            project_id='reddy000-c898c',
           if_exists=str(replace_type),
           credentials=credentials)
        logging.info("data appended to table : %s",destination_table)
        print("data appended successfully : "+str(destination_table))
    except Exception as e:
        logging.error("Exception occured in appending %s", destination_table)
        print("Error while appending : "+str(destination_table))


# def filter_todays_stock_data(table_name):
#     ## The logic should fetch the maximum Execution_date from big query and filter for gte of that Datetime from Mongodb
#     collection = db[table_name]
#     data = collection.find({})
#     data =  pd.DataFrame(list(data))
#     data = data[['Stock','Datetime','Open','High','Low','Close','Volume','instrumenttype','Execution_Date']]
#     print(data.head())
#     destination_table = 'Titania_Dataset.' + str(table_name)
#     print(destination_table)
#     # upload_data_gbq(data,destination_table)

def filter_todays_stock_data(table_name):
    ## The logic should fetch the maximum Execution_date from big query and filter for gte of that Datetime from Mongodb
    collection = db[table_name]
    data = collection.find({})
    data =  pd.DataFrame(list(data))
    data = data[['Stock','Datetime','Open','High','Low','Close','Volume','instrumenttype','Execution_Date']]
    # print(data.head())
    destination_table = 'Titania_Dataset.' + str(table_name)
    
    print(destination_table)



    sql = 'SELECT max(Execution_Date) FROM '+str('reddy000-c898c.') + destination_table

    print(sql)
    
    # sql = """
    # SELECT max(Execution_Date) FROM `reddy000-c898c.Titania_Dataset.Stocks_data_5_minutes`
    # """
    df = pandas_gbq.read_gbq(sql, project_id='reddy000-c898c')

    latest_date = df.iloc[0,0]
    print(latest_date)
    
#     db = client["titania_trading"]

    today_orders_raw_data = db[str(table_name)].find({'Execution_Date': {'$gte': "2022-11-30"}})

    today_orders_raw_data =  pd.DataFrame(list(today_orders_raw_data))
    
    # print(today_orders_raw_data)
    
    today_orders_raw_data = today_orders_raw_data[['Stock','Datetime','Open','High','Low','Close','Volume','instrumenttype','Execution_Date']]
    
    print(today_orders_raw_data)
    
    print(today_orders_raw_data.dtypes)

    upload_data_gbq(today_orders_raw_data,destination_table,"append")
    
    
    # today_orders_raw_data.to_gbq(destination_table=destination_table,
    #         project_id='reddy000-c898c',
    #        if_exists='replace')

# def upload_candle_data(table_name):
#     collection = db[table_name]
#     data = collection.find({})
#     data =  pd.DataFrame(list(data))
    
#     if table_name == 'Candle_stick_pattern_5_minutes':
#         data = data[['Stock', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
#        'instrumenttype', 'Execution_Date', 'candlestick_pattern',
#        'candlestick_match_count']]
#     else:
#         data = data[['Datetime', 'Strike_Price', 'future_volume',
#        'call_traded_volume', 'call_pchange', 'call_changeinopeninterest',
#        'put_traded_volume', 'put_pchange', 'put_changeinopeninterest',
#        'pcr_ratio', 'fut_volume_rank', 'call_volume_rank', 'put_volume_rank',
#        'Call_Majority', 'Put_Majority', 'call_value', 'put_value', 'Stock',
#        'pcr_call', 'Execution_Date']]
    
# #     print(data.columns)
    
#     destination_table = 'Titania_Dataset.' + str(table_name)
#     print(destination_table)

#     # upload_data_gbq(data,destination_table)


def upload_candle_data(table_name):
    collection = db[table_name]
    data = collection.find({})
    data =  pd.DataFrame(list(data))

    destination_table = 'Titania_Dataset.' + str(table_name)
    print(destination_table)
    
    if table_name == 'Candle_stick_pattern_5_minutes':
        data = data[['Stock', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume','instrumenttype', 'Execution_Date', 'candlestick_pattern','candlestick_match_count']]
        print(data)
        upload_data_gbq(data,destination_table,"replace")

    else:
        data = data[['Datetime', 'Strike_Price', 'future_volume','call_traded_volume', 'call_pchange', 'call_changeinopeninterest','put_traded_volume', 'put_pchange', 'put_changeinopeninterest','pcr_ratio', 'fut_volume_rank', 'call_volume_rank', 'put_volume_rank','Call_Majority', 'Put_Majority', 'call_value', 'put_value', 'Stock','pcr_call', 'Execution_Date']]
        print(data)
        upload_data_gbq(data,destination_table,"append")



# def upload_technical_data_to_big_query(table_name):
#     collection = db[table_name]
#     data = collection.find({})
#     data =  pd.DataFrame(list(data))
#     data = data[['Stock', 'Execution_Date', 'Datetime', 'Open', 'High', 'Low',
#        'Close', 'Volume', 'instrumenttype', 'SMA_Call', 'RSI_Call',
#        'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call',
#        'SuperTrend_Call', 'buy_probability', 'sell_probability']]
    
#     destination_table = 'Titania_Dataset.' + str(table_name)
    
#     print(destination_table)
    
#     # upload_data_gbq(data,destination_table)

def upload_technical_data_to_big_query(table_name):
    collection = db[table_name]
    data = collection.find({})
    data =  pd.DataFrame(list(data))
    data = data[['Stock', 'Execution_Date', 'Datetime', 'Open', 'High', 'Low',
       'Close', 'Volume', 'instrumenttype', 'SMA_Call', 'RSI_Call',
       'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call', 'VWAP_Call',
       'SuperTrend_Call', 'buy_probability', 'sell_probability']]
    
    destination_table = 'Titania_Dataset.' + str(table_name)
    
    print(destination_table)
    
    today_orders_raw_data = db[str(table_name)].find({})

    today_orders_raw_data =  pd.DataFrame(list(today_orders_raw_data))
    
    print(today_orders_raw_data)
    
    today_orders_raw_data = today_orders_raw_data[['Stock','Datetime','Open','High','Low','Close','Volume','instrumenttype','Execution_Date']]

    print(today_orders_raw_data)

    upload_data_gbq(today_orders_raw_data,destination_table,"append")
    
    # today_orders_raw_data.to_gbq(destination_table=destination_table,
    #         project_id='reddy000-c898c',
    #        if_exists='append')

# def upload_options_and_futures_data(table_name):
#     collection = db[table_name]
#     data = collection.find({})
#     data =  pd.DataFrame(list(data))
#     if table_name == 'options_signals':
#         data = data[['Datetime', 'Call_Interpretation', 'Put_Interpretation',
#        'pcr_ratio', 'current_call_volume', 'current_put_volume',
#        'Call_Majority', 'Put_Majority', 'call_volume_rank', 'put_volume_rank',
#        'signal', 'Stock']]
#     else:
#         data = data[['Datetime', 'Strike_Price', 'future_volume',
#        'call_traded_volume', 'call_pchange', 'call_changeinopeninterest',
#        'put_traded_volume', 'put_pchange', 'put_changeinopeninterest',
#        'pcr_ratio', 'fut_volume_rank', 'call_volume_rank', 'put_volume_rank',
#        'Call_Majority', 'Put_Majority', 'call_value', 'put_value', 'Stock']]
    
#     destination_table = 'Titania_Dataset.' + str(table_name)
    
#     print(destination_table)
    
#     # upload_data_gbq(data,destination_table)

def upload_options_and_futures_data(table_name):
    collection = db[table_name]
    data = collection.find({})
    data =  pd.DataFrame(list(data))
    
    
    if table_name == 'options_signals':
        data = data[['Datetime', 'Call_Interpretation', 'Put_Interpretation',
       'pcr_ratio', 'current_call_volume', 'current_put_volume',
       'Call_Majority', 'Put_Majority', 'call_volume_rank', 'put_volume_rank',
       'signal', 'Stock']]
        
        
    
    else:
        data = data[['Datetime', 'Strike_Price', 'future_volume',
       'call_traded_volume', 'call_pchange', 'call_changeinopeninterest',
       'put_traded_volume', 'put_pchange', 'put_changeinopeninterest',
       'pcr_ratio', 'fut_volume_rank', 'call_volume_rank', 'put_volume_rank',
       'Call_Majority', 'Put_Majority', 'call_value', 'put_value', 'Stock']]
    
    print(data.columns)
    
    print(data)
    
    destination_table = 'Titania_Dataset.' + str(table_name)
    
    print(destination_table)

    upload_data_gbq(data,destination_table,"append")
    
    # data.to_gbq(destination_table=destination_table,
    #         project_id='reddy000-c898c',
    #        if_exists='append')

# def upload_algo_data(table_name):
#     collection = db[table_name]
#     data = collection.find({})
#     data =  pd.DataFrame(list(data))
    
#     if table_name == 'final_orders_raw_data':
#         data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call',
#        'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call',
#        'VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability',
#        'StopLoss', 'Target', 'Qty', 'Spot_Price', 'expiry', 'Strike_Buy_Price',
#        'premium_StopLoss', 'premium_Target', 'lotsize', 'premium_Qty',
#        'historic_profit', 'current_script', 'token', 'exec_rnk', 'conclusion',
#        'execution_date']]
#     elif table_name == 'algo_orders_place_data':
#         data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call',
#        'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call',
#        'VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability',
#        'StopLoss', 'Target', 'Qty', 'Spot_Price', 'expiry', 'Strike_Buy_Price',
#        'premium_StopLoss', 'premium_Target', 'lotsize', 'premium_Qty',
#        'historic_profit', 'current_script', 'token', 'exec_rnk', 'conclusion',
#        'execution_date', 'rnk', 'order_id', 'order_place', 'client_id']]
#     else:
#         data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call',
#        'RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call',
#        'VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability',
#        'StopLoss', 'Target', 'Qty', 'expiry', 'exec_rnk',
#        'execution_date']]
    
    
# #     print(data.columns)
#     data = data.astype(str)
#     destination_table = 'Titania_Dataset.' + str(table_name)

#     print(destination_table)

#     # upload_data_gbq(data,destination_table)


def upload_algo_data(table_name):
    collection = db[table_name]
    data = collection.find({})
    data =  pd.DataFrame(list(data))
    
    if table_name == 'final_orders_raw_data':
        data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call','VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability','StopLoss', 'Target', 'Qty', 'Spot_Price', 'expiry', 'Strike_Buy_Price','premium_StopLoss', 'premium_Target', 'lotsize', 'premium_Qty','historic_profit', 'current_script', 'token', 'exec_rnk', 'conclusion','execution_date']]
        
        
    elif table_name == 'algo_orders_place_data':
        data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call','VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability','StopLoss', 'Target', 'Qty', 'Spot_Price', 'expiry', 'Strike_Buy_Price','premium_StopLoss', 'premium_Target', 'lotsize', 'premium_Qty','historic_profit', 'current_script', 'token', 'exec_rnk', 'conclusion','execution_date', 'rnk', 'order_id', 'order_place', 'client_id']]
    else:
        data = data[['Strategy', 'Stock', 'Signal', 'Datetime', 'Value', 'SMA_Call','RSI_Call', 'MACD_Call', 'Pivot_Call', 'PCR_Call', 'BB_Call','VWAP_Call', 'SuperTrend_Call', 'buy_probability', 'sell_probability','StopLoss', 'Target', 'Qty', 'expiry', 'exec_rnk','execution_date']]
#     print(data.columns)
    data = data.astype(str)
    destination_table = 'Titania_Dataset.' + str(table_name)
#     print(destination_table)
    print(data)
    upload_data_gbq(data,destination_table,"replace")
    # data.to_gbq(destination_table=destination_table,
    #         project_id='reddy000-c898c',
    #        if_exists='replace')



    


def main():
    # Configure the logging system
    logging.basicConfig(filename ='/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Historic_Backup/historic_backup_'+ str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")) +'.log',
                        level = logging.DEBUG)

    logging.info("Script execution started")
    logging.info(start_time)

    for tbl in range(0,len(list_of_collections)):
        print(list_of_collections[tbl])
        if list_of_collections[tbl] in ['technical_indicator_pcr','Candle_stick_pattern_5_minutes']:
            # upload_candle_data(str(list_of_collections[tbl]))
            print("Need to complete the candle stick upload script")
        elif list_of_collections[tbl] in ['technical_indicator_1_minutes',
                                        'technical_indicator_5_minutes',
                                        'technical_indicator_15_minutes',
                                        'technical_indicator_30_minutes',
                                        'technical_indicator_60_minutes',
                                        'technical_indicator_1_day']:
            upload_technical_data_to_big_query(str(list_of_collections[tbl]))
        elif list_of_collections[tbl] in ['Stocks_data_1_minutes',
                                        'Stocks_data_5_minutes',
                                        'Stocks_data_15_minutes',
                                        'Stocks_data_30_minutes',
                                        'Stocks_data_60_minutes',
                                        'Stocks_data_1_day']:
            filter_todays_stock_data(str(list_of_collections[tbl]))
        elif list_of_collections[tbl] in ['final_orders_raw_data','algo_orders_place_data','orders_raw_data']:
            upload_algo_data(str(list_of_collections[tbl]))
        elif list_of_collections[tbl] in ['options_signals','futures_options_signals']:
            upload_options_and_futures_data(str(list_of_collections[tbl]))
        else:
            print("Skipping")



if __name__ == '__main__':
	main()
