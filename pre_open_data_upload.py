import requests
import pandas as pd
import time
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE")
db = client.titania_trading
collection = db.today_pre_open_market_data

nse_url = "https://www.nseindia.com/"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()

data_list = []
final_data = pd.DataFrame()

while True:
    session.mount(nse_url, requests.adapters.HTTPAdapter(max_retries=5))
    request = session.get(nse_url, headers=headers, timeout=10)

    cookies = dict(request.cookies)

    new_nseurl = "https://www.nseindia.com/api/market-data-pre-open?key=NIFTY"

    request = session.get(new_nseurl, headers=headers, timeout=10, cookies=cookies)

    if request.status_code == 200:
        # Success! The request was completed successfully.
        print("Request successful")
        json_data = pd.DataFrame(request.json()['data'])
        json_data['lastUpdateTime'] = json_data.loc[0,'detail']['preOpenMarket']['lastUpdateTime']
        
        print(json_data)
        data_list.append(json_data)
        
        # Check if it is past 9:07:59
        now = pd.Timestamp.now(tz='Asia/Kolkata')
        print(now)
        market_close = now.replace(hour=9, minute=7, second=59, microsecond=0)
        if now >= market_close:
            # Concatenate all the dataframes in the list
            final_data = pd.concat(data_list, ignore_index=True)

            final_data['symbol'] = final_data['metadata'].apply(lambda x: x['symbol'])
            
            # Delete all records from the collection
            collection.delete_many({})

            print("Records deleted")
            
            # Upload the final data to MongoDB
            collection.insert_many(final_data.to_dict('records'))
            
            print("Data uploaded to MongoDB")
            
            # Empty the data list
            data_list = []
            
            # Exit the loop
            break
    else:
        # An error occurred while processing the request.
        print(f"Error: Status code {request.status_code}")
    
    # Wait for 5 seconds before making the next request
    time.sleep(5)