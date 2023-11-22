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
import time

import requests
from requests.adapters import HTTPAdapter


ct = datetime.now(timezone("Asia/Kolkata")).strftime("%H:%M:%S") 
ct1 = '09:15:00'
ct2 = '15:30:00'

ts = str(datetime.now(timezone("Asia/Kolkata")))
start_time = datetime.now(timezone("Asia/Kolkata"))
print(start_time)


# if (ct >= ct1) and (ct <= ct2):
nse_url = "https://www.nseindia.com/"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()


session.mount(nse_url, HTTPAdapter(max_retries=5))
request = session.get(nse_url, headers=headers, timeout=10)

# print(request.text)

cookies = dict(request.cookies)
# response = session.get(url, headers=headers, timeout=5, cookies=cookies)


new_url = 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
fut_url = "https://www.nseindia.com/api/liveEquity-derivatives?index=nse50_fut"
money_control_url = "https://www.moneycontrol.com/india/indexfutures/nifty/9/2023-06-29/FUTIDX/XX/0.00/true"
# future_url = "https://www.nseindia.com/api/historical/fo/derivatives?&symbol=NIFTY&identifier=FUTIDXNIFTY24-02-2022XX0.00"

# headers = {'User-Agent': 'Mozilla/5.0'}
# session.mount(new_url, HTTPAdapter(max_retries=5))
page = requests.get(new_url,headers=headers,timeout=10,cookies=cookies)
# print(page.text)


# headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
#                      'like Gecko) '
#                      'Chrome/80.0.3987.149 Safari/537.36',
#        'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}

# base_url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
# page_resp = requests.get(base_url,headers=headers,timeout=10)
# page_cookies = page_resp.cookies

# print(page_cookies)


session.mount(fut_url, HTTPAdapter(max_retries=5))
fut_page = requests.get(fut_url,headers=headers, timeout=10,cookies = cookies)
# print(fut_page.text)

try:
    fut_dajs = json.loads(fut_page.text)

    fut_data = pd.DataFrame(fut_dajs['data'])
#     print(fut_data)
    fut_data['execution_time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

    fut_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Futures_data/Nifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))

    if not os.path.exists(fut_dir_name):
        os.makedirs(fut_dir_name)

    if os.path.exists(fut_dir_name+"/"+str('fut_data.csv')):
        fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=False)
    else:
        fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=True)
except:
    print("Exception inn Futures Nifty Data")
    time.sleep(2)
    session.mount(fut_url, HTTPAdapter(max_retries=5))
    fut_page = requests.get(fut_url,headers=headers, timeout=10,cookies = cookies)
    
    fut_dajs = json.loads(fut_page.text)

    fut_data = pd.DataFrame(fut_dajs['data'])
    fut_data['execution_time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

    fut_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Futures_data/Nifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))

    if not os.path.exists(fut_dir_name):
        os.makedirs(fut_dir_name)

    if os.path.exists(fut_dir_name+"/"+str('fut_data.csv')):
        fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=False)
    else:
        fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=True)
        
# future_page = requests.get(future_url,headers=headers, timeout=10,cookies = page_cookies)
# future_dajs = json.loads(future_page.text)
# future_dajs = pd.DataFrame(future_dajs['data'])
# current_fut_data = future_dajs.sort_values('FH_TIMESTAMP',ascending = False).head(1)
# current_fut_data = current_fut_data[["FH_INSTRUMENT","FH_SYMBOL","FH_EXPIRY_DT","FH_OPENING_PRICE","FH_TRADE_HIGH_PRICE","FH_TRADE_LOW_PRICE","FH_CLOSING_PRICE","FH_LAST_TRADED_PRICE","FH_PREV_CLS","FH_SETTLE_PRICE","FH_TOT_TRADED_QTY","FH_TOT_TRADED_VAL","FH_OPEN_INT","FH_CHANGE_IN_OI","FH_MARKET_LOT","FH_TIMESTAMP","CALCULATED_PREMIUM_VAL"]]

session.mount(money_control_url, HTTPAdapter(max_retries=5))
money_control_page = requests.get(money_control_url)

soup = BeautifulSoup(money_control_page.content, "html.parser")

job_elements = soup.find_all("div", class_="FL PR15")
job_elements_OI = soup.find_all("div", class_="FR PA10")

mytable = job_elements[12]

my_data_table = mytable.find_all('table',class_ = "tbldata")
my_data_table_OI = job_elements_OI[0].find_all('table',class_ = "tbldata")

my_data_table[0]

current_fut_data = pd.DataFrame()

for el in my_data_table[0].findAll('tr'):
  for th in el.findAll('th'):
      if th.text == "Open Price":
          current_fut_data.loc[0,"OpenPrice"] = el.findAll('td')[0].text
      elif th.text == "Low Price":
          current_fut_data.loc[0,"LowPrice"] = el.findAll('td')[0].text
      elif th.text == "Prev. Close":
          current_fut_data.loc[0,"PrevClose"] = el.findAll('td')[0].text
      elif th.text == "Spot Price":
          current_fut_data.loc[0,"SpotPrice"] = el.findAll('td')[0].text
      elif th.text == "Open Int PCR":
          current_fut_data.loc[0,"OpenIntPCR"] = el.findAll('td')[0].text
      elif th.text == "Prev OI PCR":
          current_fut_data.loc[0,"PrevOIPCR"] = el.findAll('td')[0].text
      elif th.text == "Bid Price":
          current_fut_data.loc[0,"BidPrice"] = el.findAll('td')[0].text
      elif th.text == "Rollover %":
          current_fut_data.loc[0,"RolloverPerc"] = el.findAll('td')[0].text
          
for el in my_data_table_OI[0].findAll('tr'):
    for th in el.findAll('th'):
        if th.text == "Average Price":
            current_fut_data.loc[0,"AvgPrice"] = el.findAll('td')[0].text
        elif th.text == "No. of Contracts Traded":
            current_fut_data.loc[0,"ContractsTraded"] = el.findAll('td')[0].text
        elif th.text == "Turnover (Rs. in lakhs)":
            current_fut_data.loc[0,"Turnover"] = el.findAll('td')[0].text
        elif th.text == "Market Lot":
            current_fut_data.loc[0,"MarketLot"] = el.findAll('td')[0].text
        elif th.text == "Open Interest":
            current_fut_data.loc[0,"OpenInterest"] = el.findAll('td')[0].text
        elif th.text == "Open Int. Chg":
            current_fut_data.loc[0,"OpenIntChg"] = el.findAll('td')[0].text
        elif th.text == "Open Int. Chg %":
            current_fut_data.loc[0,"OpenIntChgPerc"] = el.findAll('td')[0].text
        elif th.text == "Offer Price":
            current_fut_data.loc[0,"OfferPrice"] = el.findAll('td')[0].text
        elif th.text == "Offer Qty":
            current_fut_data.loc[0,"OfferQty"] = el.findAll('td')[0].text

current_fut_data['execution_time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

print("current_fut_data")
print(current_fut_data)

futures_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Futures_Updated_data/Nifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))

if not os.path.exists(futures_dir_name):
    os.makedirs(futures_dir_name)

if os.path.exists(futures_dir_name+"/"+str('fut_updated_data.csv')):
    current_fut_data.to_csv(futures_dir_name+"/"+str('fut_updated_data.csv'), mode='a', index=False, header=False)
else:
    current_fut_data.to_csv(futures_dir_name+"/"+str('fut_updated_data.csv'), mode='a', index=False, header=True)


today = datetime.now(timezone("Asia/Kolkata")).today()
start = today - timedelta(days=today.weekday())
end = start + timedelta(days=3)
#    expiry_dt = end.strftime('%d-%b-%Y')
# expiry_dt = '16-Dec-2021'
    
# ce_values = [data['CE'] for data in dajs['records']['data'] if "CE" in data]
# pe_values = [data['PE'] for data in dajs['records']['data'] if "PE" in data]
# print(dajs)

try:
    dajs = json.loads(page.text)
    try:
      ce_values = [data['CE'] for data in dajs['records']['data'] if "CE" in data]
      pe_values = [data['PE'] for data in dajs['records']['data'] if "PE" in data]

    except:
      ce_values = [data['CE'] for data in dajs['filtered']['data'] if "CE" in data]
      pe_values = [data['PE'] for data in dajs['filtered']['data'] if "PE" in data]
        
        
    ce_dt = pd.DataFrame(ce_values).sort_values(['strikePrice'])
    pe_dt = pd.DataFrame(pe_values).sort_values(['strikePrice'])



    ce = ce_dt[['askPrice', 'strikePrice','bidprice', 'bidQty', 'change', 'changeinOpenInterest', 'expiryDate', 
       'identifier', 'impliedVolatility', 'lastPrice',  'openInterest', 'pChange', 'pchangeinOpenInterest',
       'strikePrice', 'totalBuyQuantity', 'totalSellQuantity', 'totalTradedVolume', 'underlying', 'underlyingValue']]

    ce = ce.set_axis(['askprice', 'strikeprice','bidprice', 'bidqty', 'change', 'changeinopeninterest', 'expirydate', 
       'identifier', 'impliedvolatility', 'lastprice',  'openinterest', 'pchange', 'pchangeinopeninterest',
       'strikeprice', 'totalbuyquantity', 'totalsellquantity', 'totaltradedvolume', 'underlying', 'underlyingvalue'], axis=1, inplace=False)
    ce['type'] = "CALL"
    ce['time'] = ts

    pe = pe_dt[['askPrice', 'strikePrice','bidprice', 'bidQty', 'change', 'changeinOpenInterest', 'expiryDate', 
       'identifier', 'impliedVolatility', 'lastPrice',  'openInterest', 'pChange', 'pchangeinOpenInterest',
       'strikePrice', 'totalBuyQuantity', 'totalSellQuantity', 'totalTradedVolume', 'underlying', 'underlyingValue']]

    pe = pe.set_axis(['askprice', 'strikeprice','bidprice', 'bidqty', 'change', 'changeinopeninterest', 'expirydate', 
       'identifier', 'impliedvolatility', 'lastprice',  'openinterest', 'pchange', 'pchangeinopeninterest',
       'strikeprice', 'totalbuyquantity', 'totalsellquantity', 'totaltradedvolume', 'underlying', 'underlyingvalue'], axis=1, inplace=False)
    pe['type'] = "PUT"
    pe['time'] = ts


    option_chain = pd.concat([ce, pe], ignore_index=True)
    option_chain = option_chain.loc[:,~option_chain.columns.duplicated()]

    # option_chain.to_csv('/Users/apple/Desktop/Python_Stocks_Automation/option_chain_nifty.csv', mode='a', index=False, header=False) 
    print(str(ts)+" Nifty Success")
        
except:
    print("Exception Occured in Nifty")




# else:
#     print(str(ts)+" Nifty Market Closed")

current_strike_prices = option_chain['strikeprice'].unique()

data = yf.download(tickers='%5ENSEI', period="5d", interval="5m")

data = pd.DataFrame(data)

latest_data = data.tail(1)

latest_data.reset_index(inplace = True, drop = True)

filter_strike = ((current_strike_prices > (latest_data.iloc[0]['Close']) - 1000) & (current_strike_prices < (latest_data.iloc[0]['Close']) + 1000))

current_strike_prices = current_strike_prices[filter_strike]

for i in range(0,len(current_strike_prices)):
  # print(current_strike_prices[i])
  # Get all the strike peices
  temp_data = option_chain.loc[option_chain['strikeprice'] == current_strike_prices[i]]

  temp_data.reset_index(level=0, inplace=True)
  temp_data = pd.DataFrame(temp_data)
  for idx in range(0,len(temp_data)):
    dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/Nifty/'+str(temp_data.loc[idx,"expirydate"])
    file_name = dir_name+'/' + str(temp_data.loc[idx,"strikeprice"])+'.csv'
    final_data  = temp_data.loc[temp_data['expirydate'] == temp_data.loc[idx,"expirydate"]]

    final_data = final_data.drop_duplicates(keep=False, inplace=False)

    final_data.reset_index(level=0, inplace=True)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    if os.path.exists(file_name):
        final_data.to_csv(file_name, mode='a', index=False, header=False) 
    else:
        final_data.to_csv(file_name, mode='a', index=False, header=True)

# if (ct >= ct1) and (ct <= ct2):

new_url = 'https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY'
fut_url = "https://www.nseindia.com/api/liveEquity-derivatives?index=nifty_bank_fut"
money_control_url = "https://www.moneycontrol.com/india/indexfutures/banknifty/23/2023-06-29/FUTIDX/XX/0.00/true"
# future_url = "https://www.nseindia.com/api/historical/fo/derivatives?&symbol=BANKNIFTY&identifier=FUTIDXBANKNIFTY24-02-2022XX0.00"

# headers = {'User-Agent': 'Mozilla/5.0'}
session.mount(new_url, HTTPAdapter(max_retries=5))
page = requests.get(new_url,headers=headers, timeout=10,cookies=cookies)



# headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
#                      'like Gecko) '
#                      'Chrome/80.0.3987.149 Safari/537.36',
#        'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}

# # base_url = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
# # page_resp = requests.get(base_url,headers=headers,timeout=10)
# # page_cookies = page_resp.cookies

# print(page_cookies)


session.mount(fut_url, HTTPAdapter(max_retries=5))
fut_page = requests.get(fut_url,headers=headers, timeout=10,cookies = cookies)
# print(fut_page.text)
fut_dajs = json.loads(fut_page.text)

fut_data = pd.DataFrame(fut_dajs['data'])
fut_data['execution_time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

fut_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Futures_data/BankNifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))

if not os.path.exists(fut_dir_name):
    os.makedirs(fut_dir_name)

if os.path.exists(fut_dir_name+"/"+str('fut_data.csv')):
    fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=False)
else:
    fut_data.to_csv(fut_dir_name+"/"+str('fut_data.csv'), mode='a', index=False, header=True)

# future_page = requests.get(future_url,headers=headers, timeout=10,cookies = page_cookies)
# future_dajs = json.loads(future_page.text)
# future_dajs = pd.DataFrame(future_dajs['data'])
# current_fut_data = future_dajs.sort_values('FH_TIMESTAMP',ascending = False).head(1)
# current_fut_data = current_fut_data[["FH_INSTRUMENT","FH_SYMBOL","FH_EXPIRY_DT","FH_OPENING_PRICE","FH_TRADE_HIGH_PRICE","FH_TRADE_LOW_PRICE","FH_CLOSING_PRICE","FH_LAST_TRADED_PRICE","FH_PREV_CLS","FH_SETTLE_PRICE","FH_TOT_TRADED_QTY","FH_TOT_TRADED_VAL","FH_OPEN_INT","FH_CHANGE_IN_OI","FH_MARKET_LOT","FH_TIMESTAMP","CALCULATED_PREMIUM_VAL"]]
session.mount(money_control_url, HTTPAdapter(max_retries=5))
money_control_page = requests.get(money_control_url)

soup = BeautifulSoup(money_control_page.content, "html.parser")

job_elements = soup.find_all("div", class_="FL PR15")
job_elements_OI = soup.find_all("div", class_="FR PA10")

mytable = job_elements[12]

my_data_table = mytable.find_all('table',class_ = "tbldata")
my_data_table_OI = job_elements_OI[0].find_all('table',class_ = "tbldata")

my_data_table[0]

current_fut_data = pd.DataFrame()

for el in my_data_table[0].findAll('tr'):
  for th in el.findAll('th'):
      if th.text == "Open Price":
          current_fut_data.loc[0,"OpenPrice"] = el.findAll('td')[0].text
      elif th.text == "Low Price":
          current_fut_data.loc[0,"LowPrice"] = el.findAll('td')[0].text
      elif th.text == "Prev. Close":
          current_fut_data.loc[0,"PrevClose"] = el.findAll('td')[0].text
      elif th.text == "Spot Price":
          current_fut_data.loc[0,"SpotPrice"] = el.findAll('td')[0].text
      elif th.text == "Open Int PCR":
          current_fut_data.loc[0,"OpenIntPCR"] = el.findAll('td')[0].text
      elif th.text == "Prev OI PCR":
          current_fut_data.loc[0,"PrevOIPCR"] = el.findAll('td')[0].text
      elif th.text == "Bid Price":
          current_fut_data.loc[0,"BidPrice"] = el.findAll('td')[0].text
      elif th.text == "Rollover %":
          current_fut_data.loc[0,"RolloverPerc"] = el.findAll('td')[0].text
          
for el in my_data_table_OI[0].findAll('tr'):
    for th in el.findAll('th'):
        if th.text == "Average Price":
            current_fut_data.loc[0,"AvgPrice"] = el.findAll('td')[0].text
        elif th.text == "No. of Contracts Traded":
            current_fut_data.loc[0,"ContractsTraded"] = el.findAll('td')[0].text
        elif th.text == "Turnover (Rs. in lakhs)":
            current_fut_data.loc[0,"Turnover"] = el.findAll('td')[0].text
        elif th.text == "Market Lot":
            current_fut_data.loc[0,"MarketLot"] = el.findAll('td')[0].text
        elif th.text == "Open Interest":
            current_fut_data.loc[0,"OpenInterest"] = el.findAll('td')[0].text
        elif th.text == "Open Int. Chg":
            current_fut_data.loc[0,"OpenIntChg"] = el.findAll('td')[0].text
        elif th.text == "Open Int. Chg %":
            current_fut_data.loc[0,"OpenIntChgPerc"] = el.findAll('td')[0].text
        elif th.text == "Offer Price":
            current_fut_data.loc[0,"OfferPrice"] = el.findAll('td')[0].text
        elif th.text == "Offer Qty":
            current_fut_data.loc[0,"OfferQty"] = el.findAll('td')[0].text



current_fut_data['execution_time'] = datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y %H:%M:%S")

futures_dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Futures_Updated_data/BankNifty/' + str(datetime.now(timezone("Asia/Kolkata")).strftime("%d-%m-%Y"))

if not os.path.exists(futures_dir_name):
    os.makedirs(futures_dir_name)

if os.path.exists(futures_dir_name+"/"+str('fut_updated_data.csv')):
    current_fut_data.to_csv(futures_dir_name+"/"+str('fut_updated_data.csv'), mode='a', index=False, header=False)
else:
    current_fut_data.to_csv(futures_dir_name+"/"+str('fut_updated_data.csv'), mode='a', index=False, header=True)



# print(dajs)

today = datetime.now(timezone("Asia/Kolkata")).today()
start = today - timedelta(days=today.weekday())
end = start + timedelta(days=3)
#   expiry_dt = end.strftime('%d-%b-%Y')
# expiry_dt = '16-Dec-2021'
   
try:  
    dajs = json.loads(page.text)

    try:
      ce_values = [data['CE'] for data in dajs['records']['data'] if "CE" in data]
      pe_values = [data['PE'] for data in dajs['records']['data'] if "PE" in data]

    except:
      ce_values = [data['CE'] for data in dajs['filtered']['data'] if "CE" in data]
      pe_values = [data['PE'] for data in dajs['filtered']['data'] if "PE" in data]
        
    ce_dt = pd.DataFrame(ce_values).sort_values(['strikePrice'])
    pe_dt = pd.DataFrame(pe_values).sort_values(['strikePrice'])



    ce = ce_dt[['askPrice', 'strikePrice','bidprice', 'bidQty', 'change', 'changeinOpenInterest', 'expiryDate', 
       'identifier', 'impliedVolatility', 'lastPrice',  'openInterest', 'pChange', 'pchangeinOpenInterest',
       'strikePrice', 'totalBuyQuantity', 'totalSellQuantity', 'totalTradedVolume', 'underlying', 'underlyingValue']]

    ce = ce.set_axis(['askprice', 'strikeprice','bidprice', 'bidqty', 'change', 'changeinopeninterest', 'expirydate', 
       'identifier', 'impliedvolatility', 'lastprice',  'openinterest', 'pchange', 'pchangeinopeninterest',
       'strikeprice', 'totalbuyquantity', 'totalsellquantity', 'totaltradedvolume', 'underlying', 'underlyingvalue'], axis=1, inplace=False)
    ce['type'] = "CALL"
    ce['time'] = ts

    pe = pe_dt[['askPrice', 'strikePrice','bidprice', 'bidQty', 'change', 'changeinOpenInterest', 'expiryDate', 
       'identifier', 'impliedVolatility', 'lastPrice',  'openInterest', 'pChange', 'pchangeinOpenInterest',
       'strikePrice', 'totalBuyQuantity', 'totalSellQuantity', 'totalTradedVolume', 'underlying', 'underlyingValue']]

    pe = pe.set_axis(['askprice', 'strikeprice','bidprice', 'bidqty', 'change', 'changeinopeninterest', 'expirydate', 
       'identifier', 'impliedvolatility', 'lastprice',  'openinterest', 'pchange', 'pchangeinopeninterest',
       'strikeprice', 'totalbuyquantity', 'totalsellquantity', 'totaltradedvolume', 'underlying', 'underlyingvalue'], axis=1, inplace=False)
    pe['type'] = "PUT"
    pe['time'] = ts


    option_chain = pd.concat([ce, pe], ignore_index=True)
    option_chain = option_chain.loc[:,~option_chain.columns.duplicated()]
    # option_chain.to_csv('/Users/apple/Desktop/Python_Stocks_Automation/option_chain_bank_nifty.csv', mode='a', index=False, header=False) 
    print(str(ts)+" Bank Nifty Success")
except Exception as e:
    print(str(e))
    print("Exception Occured in Bank Nifty")




# else:
#     print(str(ts)+" Bank Nifty Market Closed")



current_strike_prices = option_chain['strikeprice'].unique()

data = yf.download(tickers='%5ENSEBANK', period="5d", interval="5m")

data = pd.DataFrame(data)

latest_data = data.tail(1)

latest_data.reset_index(inplace = True, drop = True)

filter_strike = ((current_strike_prices > (latest_data.iloc[0]['Close']) - 2000) & (current_strike_prices < (latest_data.iloc[0]['Close']) + 2000))

current_strike_prices = current_strike_prices[filter_strike]

for i in range(0,len(current_strike_prices)):
  # Get all the strike peices
  temp_data = option_chain.loc[option_chain['strikeprice'] == current_strike_prices[i]]

  temp_data.reset_index(level=0, inplace=True)
  temp_data = pd.DataFrame(temp_data)
  for idx in range(0,len(temp_data)):
    dir_name = '/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/BankNifty/'+str(temp_data.loc[idx,"expirydate"])
    file_name = dir_name+'/' + str(temp_data.loc[idx,"strikeprice"])+'.csv'
    final_data  = temp_data.loc[temp_data['expirydate'] == temp_data.loc[idx,"expirydate"]]

    final_data.reset_index(level=0, inplace=True)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    if os.path.exists(file_name):
        final_data.to_csv(file_name, mode='a', index=False, header=False) 
    else:
        final_data.to_csv(file_name, mode='a', index=False, header=True)

end_time = datetime.now(timezone("Asia/Kolkata"))

print(end_time)

print('Duration: {}'.format(end_time - start_time))
#    