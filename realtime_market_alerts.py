import requests
import pandas as pd
import requests
from datetime import datetime
from pytz import timezone 
import yfinance as yf

bot_token = '1931575614:AAFhtU1xieFDqC9WAAzw15G4KdB8rdzrif4'
chat_id = ["535162272"]


nse_url = "https://www.nseindia.com/"
headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                         'like Gecko) '
                         'Chrome/80.0.3987.149 Safari/537.36',
           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}
session = requests.Session()

data_list = []
final_data = pd.DataFrame()

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

# Create an empty DataFrame
df = pd.DataFrame(columns=['symbol', 'previousClose', 'iep', 'change', 'pChange', 'lastPrice',
                           'yearHigh', 'yearLow', 'finalQuantity', 'totalTurnover', 'marketCap'])

# Iterate over the data_list and extract metadata
for data in range(0,len(data_list[0])):
    metadata = data_list[0].loc[data,'metadata']
    row = {
        'symbol': metadata['symbol'],
        'previousClose': metadata['previousClose'],
        'iep': metadata['iep'],
        'change': metadata['change'],
        'pChange': round(metadata['pChange'],2),
        'lastPrice': metadata['lastPrice'],
        'yearHigh': metadata['yearHigh'],
        'yearLow': metadata['yearLow'],
        'finalQuantity': metadata['finalQuantity'],
        'totalTurnover': metadata['totalTurnover'],
        'marketCap': metadata['marketCap']
    }
    df = df.append(row, ignore_index=True)

# Print the resulting DataFrame
print(df)

top_gainers = df.head(5)

top_losers = df.tail(5)

top_gainers = top_gainers[['symbol', 'change', 'pChange','iep']]
top_losers = top_losers[['symbol', 'change', 'pChange','iep']]

current_time = datetime.now(timezone("Asia/Kolkata"))

current_hour = current_time.hour
current_minute = current_time.minute

# if current_hour == 9 and current_minute <= 15:

# Fetch the data for each symbol
data_frames = pd.DataFrame()

for stk in range(0,len(df)):
    symbol = df.loc[stk,'symbol'] + '.NS'
    data = yf.download(symbol, period='1d', interval='5m')
    data = pd.DataFrame(data)
    data.reset_index(level=0, inplace=True)
    data['Index'] =  df.loc[stk,'symbol']
    data = data.tail(1)
    data_frames = data_frames.append(data)

merged_df = df.merge(data_frames, left_on='symbol', right_on='Index', how='inner')
final_df = merged_df[['symbol', 'iep', 'Close']]

final_df['Close'] = final_df['Close'].astype(float)
final_df['iep'] = final_df['iep'].astype(float)

final_df['pChange'] = round((final_df['Close'] - final_df['iep']) / final_df['iep'] * 100,2)
final_df['Close'] = round(final_df['Close'],2)

print(final_df)

top_5 = final_df.nlargest(5, 'pChange')
bottom_5 = final_df.nsmallest(5, 'pChange')

print("Top 5 Perc Change:")
print(top_5)

print("Bottom 5 Perc Change:")
print(bottom_5)

# Convert the DataFrames to formatted strings
top_gainers_string = top_5.to_string(index=False)
top_losers_string = bottom_5.to_string(index=False)


# Prepare the message with the table strings
chat_message = "Current Market Report at "+str(current_hour)+" : "+str(current_minute)+" from Open Price - \n\n Top 5 Gainers:\n```\n" + top_gainers_string + "\n```"

print("Sending the message")
for cht in chat_id:
    message = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={cht}&parse_mode=Markdown&text={chat_message}'
    send = requests.post(message)

chat_message = "\nTop 5 Losers:\n```\n" + top_losers_string + "\n```"
for cht in chat_id:
    message = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={cht}&parse_mode=Markdown&text={chat_message}'
    send = requests.post(message)