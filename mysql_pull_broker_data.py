import pandas as pd
import os
from datetime import datetime,timedelta
from smartapi import SmartConnect
from pandasql import sqldf
import math
from pandas.io.json import json_normalize
import json
import numpy as np
import pandas_ta as ta
import time
import datetime as dt
from pytz import timezone
import pyotp

from pymongo import MongoClient
from pymongo.server_api import ServerApi

pd.set_option('display.max_columns', None)

start_time = datetime.now(timezone("Asia/Kolkata")) 
print("Script execution started")
print(start_time)

server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

# db = client.titania_trading

db = client["titania_trading"]

# client_details = db["client_details"].find({'client_id':{'$in' :['J95213','S1604557','G304915','K256027','M591295','M181705']}})
client_details = db["client_details"].find({'client_id':{'$in' :['J95213','M591295','M181705']}})
client_details =  pd.DataFrame(list(client_details))

def upload_data_gbq(data,destination_table,replace_type):
    try:
        data.to_gbq(destination_table=destination_table,
            project_id='reddy000-c898c',
           if_exists=str(replace_type))
        print("data appended successfully : "+str(destination_table))
    except Exception as e:
        print("Error while appending : "+str(destination_table))
        
        
final_order_data = pd.DataFrame()
final_position_data = pd.DataFrame()

for idx in range(0,len(client_details)):
    try:
        user_id = str(client_details.loc[idx,'client_id'])
        obj=SmartConnect(api_key=str(client_details.loc[idx,'client_api_key']))

        totp = pyotp.TOTP(str(client_details.loc[idx,'totp_code']))
        print("pyotp",totp.now())
        attempts = 5
        while attempts > 0:
            attempts = attempts-1
    #         data = obj.generateSession(user_id,str(client_details.loc[idx,'client_password']), totp.now())
            data = obj.generateSession(user_id,str(client_details.loc[idx,'m_pin']), totp.now())
            if data['status']:
                break
        print(data)

        my_positons_data = obj.position()
        current_position_data = pd.DataFrame(my_positons_data['data'])

        my_orders = obj.orderBook()
        order_data = pd.DataFrame(my_orders['data'])

        print(current_position_data)
        print(order_data)
        print(order_data.columns)

        if len(current_position_data) > 0:
            current_position_data['Client_id'] = user_id
            final_position_data = final_position_data.append(current_position_data)

        if len(order_data) > 0:
            order_data['Client_id'] = user_id
            final_order_data = final_order_data.append(order_data)
    except Exception as e:
        print(str(e))
        
        
if len(final_position_data) > 0:
    
    final_position_data.reset_index(inplace=True, drop=True)
        
    destination_table = 'Titania_Dataset.' + str('Position_Data')

    final_position_data["execution_date"] = (datetime.now(timezone("Asia/Kolkata"))).strftime("%Y-%m-%d")

    Raw_Position_Data = db["Position_Data"]

    x = Raw_Position_Data.delete_many({"execution_date": (datetime.now(timezone("Asia/Kolkata")) ).strftime("%Y-%m-%d")})

    print(x.deleted_count, " documents deleted.")

    final_position_data = final_position_data.astype(str)
    
    Raw_Position_Data.insert_many(final_position_data.to_dict('records'))

#     upload_data_gbq(final_position_data,destination_table,"append")

    print("Position Data updated")
        
if len(final_order_data) > 0:
    
    final_order_data.reset_index(inplace=True, drop=True)
        
    destination_table = 'Titania_Dataset.' + str('Order_Data')

    final_order_data["execution_date"] = (datetime.now(timezone("Asia/Kolkata"))).strftime("%Y-%m-%d")
    
    Raw_Order_Data = db["Order_Data"]

    x = Raw_Order_Data.delete_many({"execution_date": (datetime.now(timezone("Asia/Kolkata")) ).strftime("%Y-%m-%d")})

    print(x.deleted_count, " documents deleted.")

    final_order_data = final_order_data.astype(str)
    
    Raw_Order_Data.insert_many(final_order_data.to_dict('records'))

#     upload_data_gbq(final_order_data,destination_table,"append")
    
    print("Order Data updated")
    
end_time = datetime.now(timezone("Asia/Kolkata")) 

print(end_time)

print('Duration: {}'.format(end_time - start_time))