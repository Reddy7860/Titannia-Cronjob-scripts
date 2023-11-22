import requests
import pandas as pd
import requests
from datetime import datetime
from pytz import timezone 
import yfinance as yf

bot_token = '5897580623:AAELFlrjmilMq256GbDhibdmo47SSbHbnzo'
chat_id = ["535162272"]

us_stocks = pd.read_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/US - 30 Stocks.csv')

df = pd.DataFrame(columns=['Symbol', 'Perc_Chng'])

for _, row in us_stocks.iterrows():
    current_symbol = row['Symbol']
    data = yf.download(current_symbol, period='2d', interval='5m')
    data.reset_index(inplace=True)
    data['Symbol'] = current_symbol
    data['Date'] = data['Datetime'].dt.date

    latest_data = data[data['Date'] == data['Date'].max()]
    previous_data = data[data['Date'] == data['Date'].min()]
    previous_data = previous_data[previous_data['Datetime'] == previous_data['Datetime'].max()]
    latest_data = latest_data[latest_data['Datetime'] == latest_data['Datetime'].max()]

    percent_change = round((latest_data['Close'].iloc[0] - previous_data['Close'].iloc[0]) / previous_data['Close'].iloc[0] * 100, 2)
    df = df.append({'Symbol': current_symbol, 'Perc_Chng': percent_change}, ignore_index=True)

top_5 = df.nlargest(5, 'Perc_Chng')
bottom_5 = df.nsmallest(5, 'Perc_Chng')

print(top_5)
print(bottom_5)

# Convert the DataFrames to formatted strings
top_gainers_string = top_5.to_string(index=False)
top_losers_string = bottom_5.to_string(index=False)

current_time = datetime.now(timezone("America/New_York"))

current_hour = current_time.hour
current_minute = current_time.minute


# Prepare the message with the table strings
chat_message = "Current Market Report at "+str(current_hour)+" : "+str(current_minute)+" from Open Price - \n\n Top 5 Gainers:\n```\n" + top_gainers_string + "\n```"
chat_message += "\nTop 5 Losers:\n```\n" + top_losers_string + "\n```"

print("Sending the message")
for cht in chat_id:
    message = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={cht}&parse_mode=Markdown&text={chat_message}'
    send = requests.post(message)