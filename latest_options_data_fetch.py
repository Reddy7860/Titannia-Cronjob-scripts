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
from pandasql import sqldf
warnings.filterwarnings('ignore')

start_time = datetime.now(timezone("Asia/Kolkata"))
ct = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S") 
execution_time = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:00")
ct1 = '09:15:00'
ct2 = '15:30:00'

credentials = service_account.Credentials.from_service_account_file(
    '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/client_secret.json',
)

nse_url = "https://www.nseindia.com/"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()


session.mount(nse_url, HTTPAdapter(max_retries=5))
request = session.get(nse_url, headers=headers, timeout=10)

cookies = dict(request.cookies)


headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()

new_nseurl = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

request = session.get(new_nseurl, headers=headers, timeout=10,cookies=cookies)

if request.status_code == 200:
    # Success! The request was completed successfully.
    print("Request successful")
    required_dates = []
    for dates in request.json()['records']['expiryDates']:
        if '2023' in dates:
            required_dates.append(dates)
elif request.status_code == 404:
    # The requested resource was not found on the server.
    print("Error 404: Resource not found")
elif request.status_code == 500:
    # An error occurred on the server while processing the request.
    print("Error 500: Internal server error")
else:
    # Any other status code
    print(f"Error: Status code {request.status_code}")

print(required_dates)

symbols = ['NIFTY','BANKNIFTY']

nifty_final_data = pd.DataFrame()
banknifty_final_data = pd.DataFrame()

nifty_futures_final_data = pd.DataFrame()
banknifty_futures_final_data = pd.DataFrame()


nifty_opt_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Options_data/Nifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))
banknifty_opt_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Options_data/BankNifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))
nifty_fut_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Futures_data/Nifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))
banknifty_fut_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Futures_data/BankNifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))



current_data = request.json()['records']['data']

pysqldf = lambda q: sqldf(q, globals())

# combined_data = pd.DataFrame()
for sym in symbols:
    print(sym)
    for dt in required_dates:
        print(dt)
        filtered_records = [record for record in current_data if record['expiryDate'] == dt]
        ce_data = pd.DataFrame()
        pe_data = pd.DataFrame()
        for rec in filtered_records:
            print(rec)
            if 'CE' in rec.keys():
                ce_data = ce_data.append(pd.DataFrame(rec['CE'], index=[0]))
            if 'PE' in rec.keys():
                pe_data = pe_data.append(pd.DataFrame(rec['PE'], index=[0]))

        print(pe_data.head())
        
        combined_data = pysqldf(''' 
                select t1.openInterest as CALLS_OI,
                    t1.changeinOpenInterest as CALLS_Chng_in_OI,
                    t1.totalTradedVolume as CALLS_Volume,
                    t1.impliedVolatility as CALLS_IV,
                    t1.lastPrice as CALLS_LTP,
                    t1.pChange as CALLS_Net_Chng,
                    t1.bidQty as CALLS_Bid_Qty,
                    t1.bidprice as CALLS_Bid_Price,
                    t1.askPrice as CALLS_Ask_Price,
                    t1.askQty as CALLS_Ask_Qty,
                    t1.strikePrice as Strike_Price,
                    t2.bidQty as PUTS_Bid_Qty,
                    t2.bidprice as PUTS_Bid_Price,
                    t2.askPrice as PUTS_Ask_Price,
                    t2.askQty as PUTS_Ask_Qty,
                    t2.pChange as PUTS_Net_Chng,
                    t2.lastPrice as PUTS_LTP,
                    t2.impliedVolatility as PUTS_IV,
                    t2.totalTradedVolume as PUTS_Volume,
                    t2.changeinOpenInterest as PUTS_Chng_in_OI,
                    t2.openInterest as PUTS_OI,
                    t1.underlying as Stock,
                    t1.expiryDate as Expiry

                from ce_data t1 
                left join pe_data t2 on t1.strikePrice = t2.strikePrice and t1.expiryDate = t2.expiryDate''')
        
        combined_data['Expiry_date'] = datetime.strptime(dt, "%d-%b-%Y").strftime("%d-%m-%Y")
        combined_data['execution_time'] = datetime.now(timezone('Asia/Kolkata'))
        
        print(combined_data.head())

        if sym == "NIFTY":
            nifty_final_data = nifty_final_data.append(combined_data)
        elif sym == "BANKNIFTY":
            banknifty_final_data = banknifty_final_data.append(combined_data)


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

        # create a list of dictionaries for each row in the DataFrame
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
        # create a DataFrame from the rows list
        df = pd.DataFrame(rows)
        # add execution_time column to the DataFrame
        df['execution_time'] = json_data['val_timestamp']
        
        # soup = BeautifulSoup(request.content, "html.parser")
        # my_data_table = soup.find_all('table')
        # df = pd.read_html(str(my_data_table[1]))[0]
        # df.columns = ['Instrument','Underlying','Expiry','Option_type','Strike_Price','Open_Price','High_Price','Low_Price','Prev_Close','Last_Price','Volume','Turnover','Underlying_Value']
        # df['execution_time'] = execution_time
        if sym == "NIFTY":
            nifty_futures_final_data = nifty_futures_final_data.append(df)
        elif sym == "BANKNIFTY":
            banknifty_futures_final_data = banknifty_futures_final_data.append(df)       
    except Exception as e:
        print("Exception occured : ")
        print(date)

if len(nifty_final_data) > 0:
    nifty_final_data.Strike_Price = nifty_final_data.Strike_Price.round()
    nifty_final_data['Strike_Price'] = nifty_final_data['Strike_Price'].fillna(0)
    nifty_final_data["Strike_Price"] = nifty_final_data["Strike_Price"].astype(int)
    banknifty_final_data.Strike_Price = banknifty_final_data.Strike_Price.round()
    banknifty_final_data['Strike_Price'] = banknifty_final_data['Strike_Price'].fillna(0)
    banknifty_final_data["Strike_Price"] = banknifty_final_data["Strike_Price"].astype(int)

if len(nifty_futures_final_data) > 0:
    if not os.path.exists(nifty_fut_dir_name):
        os.makedirs(nifty_fut_dir_name)

    if os.path.exists(nifty_fut_dir_name+"/"+str('fut_data.csv')):
        nifty_futures_final_data.to_csv(nifty_fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=False)
    else:
        nifty_futures_final_data.to_csv(nifty_fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=True)


if len(banknifty_futures_final_data) > 0:
    if not os.path.exists(banknifty_fut_dir_name):
        os.makedirs(banknifty_fut_dir_name)

    if os.path.exists(banknifty_fut_dir_name+"/"+str('fut_data.csv')):
        banknifty_futures_final_data.to_csv(banknifty_fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=False)
    else:
        banknifty_futures_final_data.to_csv(banknifty_fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=True)

if len(nifty_final_data) > 0:
    current_strike_prices = nifty_final_data['Strike_Price'].unique()

    data = yf.download(tickers='%5ENSEI', period="5d", interval="5m")

    data = pd.DataFrame(data)

    latest_data = data.tail(1)

    latest_data.reset_index(inplace = True, drop = True)

    filter_strike = ((current_strike_prices > (latest_data.iloc[0]['Close']) - 1000) & (current_strike_prices < (latest_data.iloc[0]['Close']) + 1000))

    current_strike_prices = current_strike_prices[filter_strike]

    for i in range(0,len(current_strike_prices)):
    # for i in range(0,1):
        temp_data = nifty_final_data.loc[nifty_final_data['Strike_Price'] == current_strike_prices[i]]
        temp_data.reset_index(level=0, inplace=True)
        temp_data = pd.DataFrame(temp_data)
    #     print(temp_data)
        for idx in range(0,len(temp_data)):
            print(str(temp_data.loc[idx,"Expiry"]))
            dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Options_data/Nifty/'+str(temp_data.loc[idx,"Expiry_date"])
            file_name = dir_name+'/' + str(temp_data.loc[idx,"Strike_Price"])+'.csv'
            final_data  = temp_data.loc[temp_data['Expiry'] == str(temp_data.loc[idx,"Expiry"])]
            final_data.reset_index(drop=True, inplace=True)

            final_data = final_data.drop_duplicates()
    #         print(final_data)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if os.path.exists(file_name):
                final_data.to_csv(file_name, mode='a', index=False, header=False) 
            else:
                final_data.to_csv(file_name, mode='a', index=False, header=True)


if len(banknifty_final_data) > 0:
    current_strike_prices = banknifty_final_data['Strike_Price'].unique()


    data = yf.download(tickers='%5ENSEBANK', period="5d", interval="5m")

    data = pd.DataFrame(data)

    latest_data = data.tail(1)

    latest_data.reset_index(inplace = True, drop = True)

    filter_strike = ((current_strike_prices > (latest_data.iloc[0]['Close']) - 2000) & (current_strike_prices < (latest_data.iloc[0]['Close']) + 2000))

    current_strike_prices = current_strike_prices[filter_strike]

    print(current_strike_prices)

    for i in range(0,len(current_strike_prices)):
    # for i in range(0,1):
        temp_data = banknifty_final_data.loc[banknifty_final_data['Strike_Price'] == current_strike_prices[i]]
        temp_data.reset_index(level=0, inplace=True)
        temp_data = pd.DataFrame(temp_data)
        print(temp_data)
        for idx in range(0,len(temp_data)):
            print(str(temp_data.loc[idx,"Expiry"]))
            dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Latest_Options_data/BankNifty/'+str(temp_data.loc[idx,"Expiry_date"])
            file_name = dir_name+'/' + str(temp_data.loc[idx,"Strike_Price"])+'.csv'
            final_data  = temp_data.loc[temp_data['Expiry'] == str(temp_data.loc[idx,"Expiry"])]
            final_data.reset_index(drop=True, inplace=True)

            final_data = final_data.drop_duplicates()
            print(final_data)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            if os.path.exists(file_name):
                final_data.to_csv(file_name, mode='a', index=False, header=False) 
            else:
                final_data.to_csv(file_name, mode='a', index=False, header=True)


end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))


