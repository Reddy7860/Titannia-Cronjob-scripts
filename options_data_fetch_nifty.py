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

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()

new_nseurl = "https://www1.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?symbolCode=-9999&symbol=NIFTY&symbol=NIFTY&instrument=OPTIDX&date=-&segmentLink=17&segmentLink=17"

request = session.get(new_nseurl, headers=headers, timeout=10)

soup = BeautifulSoup(request.content, "html.parser")

required_dates = []

for dates in soup.find_all('option'):
    if '20' in dates.text:
        required_dates.append(dates.text)
        
credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/client_secret.json',
)

symbols = ['NIFTY','BANNKNIFTY']

final_data = pd.DataFrame()

for sym in symbols:
    print(sym)
    for date in required_dates:
        main_url = "https://www1.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?segmentLink=17&instrument=OPTIDX&symbol="+str(sym)+"&date="
        main_url = main_url + str(date)
        print(main_url)
        session.mount(main_url, HTTPAdapter(max_retries=5))
        request = session.get(new_nseurl, headers=headers, timeout=10)
        soup = BeautifulSoup(request.content, "html.parser")
        my_data_table = soup.find_all('table')
        df = pd.read_html(str(my_data_table[2]))[0]
        df.columns = ['_'.join(col) for col in df.columns.values]
        df.columns = ['CALLS_Chart', 'CALLS_OI', 'CALLS_Chng_in_OI', 'CALLS_Volume',
           'CALLS_IV', 'CALLS_LTP', 'CALLS_Net_Chng', 'CALLS_Bid_Qty',
           'CALLS_Bid_Price', 'CALLS_Ask_Price', 'CALLS_Ask_Qty',
           'Strike_Price', 'PUTS_Bid_Qty', 'PUTS_Bid_Price',
           'PUTS_Ask_Price', 'PUTS_Ask_Qty', 'PUTS_Net_Chng', 'PUTS_LTP',
           'PUTS_IV', 'PUTS_Volume', 'PUTS_Chng_in_OI', 'PUTS_OI', 'PUTS_Chart']

        df = df[['CALLS_OI', 'CALLS_Chng_in_OI', 'CALLS_Volume',
           'CALLS_IV', 'CALLS_LTP', 'CALLS_Net_Chng', 'CALLS_Bid_Qty',
           'CALLS_Bid_Price', 'CALLS_Ask_Price', 'CALLS_Ask_Qty',
           'Strike_Price', 'PUTS_Bid_Qty', 'PUTS_Bid_Price',
           'PUTS_Ask_Price', 'PUTS_Ask_Qty', 'PUTS_Net_Chng', 'PUTS_LTP',
           'PUTS_IV', 'PUTS_Volume', 'PUTS_Chng_in_OI', 'PUTS_OI']]
        df['Stock'] = sym
        df['Expiry'] = date
        df['execution_time'] = execution_time
        
        final_data = final_data.append(df)
        
        
destination_table = 'Titania_Dataset.' + str("new_nifty_options_chain_nse_data")

final_data.reset_index(drop=True,inplace=True)

final_data = final_data.astype(str)

final_data.to_gbq(destination_table=destination_table,
            project_id='reddy000-c898c',
           if_exists='append',credentials=credentials)

end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))
