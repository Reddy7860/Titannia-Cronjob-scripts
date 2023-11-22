from google.oauth2 import service_account
import pandas_gbq
import requests
import json
import pandas as pd
from datetime import timedelta, date
import datetime
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from pytz import timezone 

import requests
from requests.adapters import HTTPAdapter

import warnings
from pandas.io import gbq
import numpy as np
warnings.filterwarnings('ignore')

start_time = datetime.now(timezone("Asia/Kolkata"))
execution_time = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:00")
print(start_time)

nse_url = "https://www.nseindia.com/"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()


session.mount(nse_url, HTTPAdapter(max_retries=5))
request = session.get(nse_url, headers=headers, timeout=10)

cookies = dict(request.cookies)

credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/client_secret.json',
)

symbols = ['NIFTY','BANKNIFTY']

final_df = pd.DataFrame()

for sym in symbols:
    print(sym)
    try:
        # fut_url = "https://www1.nseindia.com/live_market/dynaContent/live_watch/fomwatchsymbol.jsp?key="+str(sym)+"&Fut_Opt=Futures"
        fut_url = ''
        if sym == "NIFTY":
            fut_url = 'https://www.nseindia.com/api/equity-stock?index=fu_nifty50'
        elif sym == "BANKNIFTY":
            fut_url = 'https://www.nseindia.com/api/equity-stock?index=fu_niftybank'
            
        print(fut_url)
        session.mount(fut_url, HTTPAdapter(max_retries=5))
        request = session.get(fut_url, headers=headers, timeout=10,cookies=cookies)

        json_data = request.json()
        values = json_data['value']
        rows = []
        for value in values:
            print(value)
            row = {
                'Instrument': value['instrument'],
                'Underlying': value['underlying'],
                'Expiry': value['expiryDate'],
                'Option_type': value['optionType'],
                'Strike_Price': value['strikePrice'],
                'Open_Price': value['openPrice'],
                'High_Price': value['highPrice'],
                'Low_Price': value['lowPrice'],
                'Prev_Close': None, # not present in json_data
                'Last_Price': value['lastPrice'],
                'Volume': value['numberOfContractsTraded'],
                'Turnover': value['totalTurnover'],
                'Underlying_Value': value['underlyingValue']
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df['execution_time'] = json_data['val_timestamp']
        final_df = final_df.append(df)
    except Exception as e:
        print("Exception occured : ")
        print(date)
    
destination_table = 'Titania_Dataset.' + str("futures_nse_data")

final_df.reset_index(drop=True,inplace=True)

final_df = final_df.astype(str)

final_df.to_gbq(destination_table=destination_table,
            project_id='reddy000-c898c',
           if_exists='append',credentials=credentials)

end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))