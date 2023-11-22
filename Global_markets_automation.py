import yfinance as yf
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo
from pytz import timezone

# Define the indices you want to fetch data for
indices = ['^DJI','^IXIC', '^GSPC', '000001.SS', '399001.SZ', '^FTSE', '^FCHI','^N225','^HSI']
index_names = ['Dow Jones','Nasdaq', 'S&P 500', 'Shanghai Composite', 'Shenzhen Composite', 'FTSE 100', 'CAC 40','Nikkei 225','Hang Seng']
index_market = ['USA','USA', 'USA', 'China', 'China', 'Europe', 'Europe','Japan','Hong Kong']


server_api = ServerApi('1')
client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)
db = client["titania_trading"]

today = datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d")

main_data = pd.DataFrame()

for ind in range(0,len(indices)):
    # Fetch the data using the yfinance library
    data = yf.download(indices[ind], start='1950-01-01', end=today)
    data = pd.DataFrame(data)
    data['Index'] = str(index_names[ind])
    data['Market'] = str(index_market[ind])

    main_data = main_data.append(data)

main_data['Date'] = main_data.index
main_data.reset_index(inplace=True,drop=True)
print(main_data)


collection = db["global_markets"]

x = collection.delete_many({})

print(x.deleted_count," documents deleted")

collection.insert_many(main_data.to_dict('records'))

print("Data Replaced successfully !!")