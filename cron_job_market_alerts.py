
import pymongo
import yfinance as yf
import requests
import ssl
import pandas as pd

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

MONGO_URI = "mongodb+srv://titaniatraders7:Mahadev_143@cluster0.kafx1p4.mongodb.net/"
client = pymongo.MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client['stock_alerts']
alerts_collection = db['alerts']

def check_and_send_alerts():
    alerts_to_check = alerts_collection.find({"alert_sent": "no", "alert_status": "on"})
    
    for alert in alerts_to_check:

        symbol = alert['stock']
        print(symbol)
        
        # Get the data
        data = yf.download(tickers=symbol, period="1d", interval="1m")
        data = pd.DataFrame(data)
        data.reset_index(level=0, inplace=True)

        print(data)

        # Getting the closing price of the last day from the 5 days data
        current_price = data['Close'].iloc[-1]
        
        should_alert = False
        if alert['action'] == 'above' and current_price > alert['price']:
            should_alert = True
        elif alert['action'] == 'below' and current_price < alert['price']:
            should_alert = True
        
        if should_alert:
            send_alert_to_telegram("Stock {} has {} the price of {}".format(
                alert['stock'], alert['action'], alert['price']
            ))
            alerts_collection.update_one({"_id": alert['_id']}, {"$set": {"alert_sent": "yes"}})

def send_alert_to_telegram(message):
    bot_token = "1931575614:AAFhtU1xieFDqC9WAAzw15G4KdB8rdzrif4"
    chat_ids = ["535162272"]
    
    for chat_id in chat_ids:
        send_message_url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}&parse_mode=Markdown&text={}'.format(
            bot_token, chat_id, message
        )
        requests.post(send_message_url)

if __name__ == '__main__':
    check_and_send_alerts()
