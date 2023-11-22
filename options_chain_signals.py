from datetime import datetime,timedelta
import pandas as pd
from pandasql import sqldf
import pandasql as pdsql
import os
from smartapi import SmartConnect
import time
import pyotp
from pytz import timezone

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import pymongo

# import mysql.connector as mysql
# import pymysql
from sqlalchemy.engine import result
import sqlalchemy
from sqlalchemy import create_engine, MetaData,\
Table, Column, Numeric, Integer, VARCHAR, update, delete

from sqlalchemy import create_engine

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

# engine = create_engine("mysql+pymysql://root:Mahadev_143@localhost/titania_trading")

# con = mysql.connect(user='root', password='Mahadev_143', database='titania_trading')
# cursor = con.cursor()


server_api = ServerApi('1')

client = MongoClient("mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE", server_api=server_api)

db = client.titania_trading

db = client["titania_trading"]

list_of_collections = db.list_collection_names()

print(list_of_collections)

pysqldf = lambda q: sqldf(q, globals())

today_now = datetime.now(timezone("Asia/Kolkata"))

print(today_now)

expiry_date = today_now
print(today_now.strftime("%w"))

if today_now.strftime("%w") == '1':
    expiry_date = today_now + timedelta(days=3)
elif today_now.strftime("%w") == '2':
    expiry_date = today_now + timedelta(days=2)
elif today_now.strftime("%w") == '3':
    expiry_date = today_now + timedelta(days=1)
elif today_now.strftime("%w") == '4':
    expiry_date = today_now + timedelta(days=0)
elif today_now.strftime("%w") == '5':
    expiry_date = today_now + timedelta(days=6)
elif today_now.strftime("%w") == '6':
    expiry_date = today_now + timedelta(days=5)
elif today_now.strftime("%w") == '7':
    expiry_date = today_now + timedelta(days=4)

print("Expiry date")
print(expiry_date.strftime("%d-%b-%Y"))

# expiry_date = '27-Oct-2022'
expiry_date = expiry_date.strftime("%d-%b-%Y")

nse_data = pd.DataFrame([["BANKNIFTY","%5ENSEBANK","BANKNIFTY-EQ"],["Nifty50","%5ENSEI","Nifty50-EQ"]],columns=["Symbol","Yahoo_Symbol","TradingSymbol"])

### Tells to run the command or not
talk_Command = 'No'


starting_strike_price = 0
ending_strike_price = 0
seq = 0
symbol = ""


collection = db.Stocks_data_1_minutes

for nse_cnt in range(0,len(nse_data)):
	todays_date = (datetime.now(timezone("Asia/Kolkata"))).strftime("%Y-%m-%d")
	print(todays_date)

	# todays_date = '2022-01-07'

	print(nse_data.loc[nse_cnt,"Symbol"])

# 	sql = ""
	collection = db.Stocks_data_1_minutes
	live_data = pd.DataFrame()

	if nse_data.loc[nse_cnt,"Symbol"] == "BANKNIFTY":
		seq = 100
		symbol = "BankNifty"
		live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "BankNifty"})
		live_data =  pd.DataFrame(list(live_data))


	else:
		seq = 50
		symbol = "Nifty"
		live_data = collection.find({"instrumenttype": "FUTIDX", "Stock": "Nifty"})
		live_data =  pd.DataFrame(list(live_data))
	live_data = live_data.loc[live_data['Execution_Date'] == max(live_data.Execution_Date),]
    
	live_data["Datetime"] = live_data["Datetime"] + timedelta(hours=5, minutes=30)

	hist_df = live_data[['Datetime','Open', 'High','Low', 'Close','Volume']]
    
	hist_df = hist_df.drop_duplicates(keep='first')

	hist_df.reset_index(inplace=True,drop=True) 
    

	if symbol == "BankNifty":
		future_data = hist_df[['Datetime','Close','Volume']]
		futures_latest_close = int(future_data.tail(1)['Close'])

		print("futures_latest_close")
		print(futures_latest_close)
		# future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], infer_datetime_format=True, utc=True )
		# future_data['Datetime'] = future_data['Datetime'].dt.tz_convert('Asia/Kolkata')
		future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], format='%Y-%m-%d %H:%M:%00')
		# future_data['Datetime'] = future_data['Datetime'].dt.tz_localize(None)

		strike_price = (futures_latest_close + (100 - futures_latest_close % 100)) if futures_latest_close % 100 > 50 else (futures_latest_close - futures_latest_close % 100)
		print("latest bnf strike : ",strike_price)

		starting_strike_price = strike_price - 500
		ending_strike_price = strike_price + 500

		def label_bnf_strike(row):
			# print(row)
			if row['Close'] % 100 > 50:
				strike = row['Close'] + (100 - row['Close'] % 100)
				strike = str(strike)
				# strike = str(strike)[:len(str(strike)) - 2]
				return (strike)
			else:
				strike = row['Close'] - row['Close'] % 100
				strike = str(strike)
				# strike = str(strike)[:len(str(strike)) - 2]
				return (strike)

		

		future_data['Strike_Price'] = future_data.apply(lambda row: label_bnf_strike(row), axis=1)

	else:
		future_data = hist_df[['Datetime','Close','Volume']]
		futures_latest_close = int(future_data.tail(1)['Close'])
		print("futures_latest_close")
		print(futures_latest_close)
		# future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], infer_datetime_format=True, utc=True )
		# future_data['Datetime'] = future_data['Datetime'].dt.tz_convert('Asia/Kolkata')
		future_data['Datetime'] = pd.to_datetime(future_data['Datetime'], format='%Y-%m-%d %H:%M:%00')
		# future_data['Datetime'] = future_data['Datetime'].dt.tz_localize(None)

		strike_price = (futures_latest_close + (50 - futures_latest_close % 50)) if futures_latest_close % 50 > 25 else (futures_latest_close - futures_latest_close % 50)
		print("latest nifty strike : ",strike_price)

		starting_strike_price = strike_price - 500
		ending_strike_price = strike_price + 500

		def label_nf_strike(row):
			if row['Close'] % 50 > 25:
				strike = row['Close'] + (50 - row['Close'] % 50)
				strike = str(strike)
				# strike = str(strike)[:len(str(strike)) - 2]
				return (strike)
			else:
				strike = row['Close'] - row['Close'] % 50
				strike = str(strike)
				# strike = str(strike)[:len(str(strike)) - 2]
				return (strike)
		# print("latest nifty strike : ",label_nf_strike(futures_latest_close))
		future_data['Strike_Price'] = future_data.apply(lambda row: label_nf_strike(row), axis=1)
	# print(future_data.columns)

	print(future_data.head(5))

	main_data = pd.DataFrame()

	for i in range(starting_strike_price,ending_strike_price,seq):
		# print(i)
		current_data = pd.read_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+str(expiry_date)+"/"+str(i)+'.csv',index_col=False)
		current_data.drop(['level_0', 'index'], axis=1, inplace=True)
		# current_data = current_data.loc[current_data['time'].dt.strftime('%Y-%m-%d') == todays_date]
		# current_data.reset_index(inplace = True, drop = True)
		main_data = pd.concat([main_data,current_data])
		main_data.reset_index(inplace = True, drop = True)
# 		print(main_data.tail())
	# print("Main Data")
	# print(main_data)
	# print(main_data.columns)
	# print(type(main_data['time']))
	from datetime import datetime

	# temp = [x for x in main_data['time'] if len(x) > K]
	# print(main_data.loc[0,'time'])
	# print(main_data.loc[195845,'time'])
	# print(len(main_data.loc[0,'time']))
	# print(len(main_data.loc[195845,'time']))

	main_data['time'] = main_data['time'].apply(lambda x: x if len(x) == 26 else x[:26])

	# print(main_data)
	

	main_data['time'] = pd.to_datetime(main_data['time']).dt.strftime('%Y-%m-%d %H:%M:00')

	main_data['time'] = pd.to_datetime(main_data['time'])
    
	print(sorted(main_data['time'].dt.strftime('%Y-%m-%d').unique()))
	print(todays_date)



	# main_data = main_data.loc[main_data['time'] <= '2021-12-13 15:00:00']

	main_data = main_data.loc[main_data['time'].dt.strftime('%Y-%m-%d') == todays_date]

	print("Actual Main")
	print(main_data.tail())

	main_data.reset_index(inplace = True, drop = True)

	main_data = main_data.drop_duplicates(keep='first')

	# print("****************main_data****************")
	# print(min(main_data['time']))

	# main_data['time'] = pd.DatetimeIndex(main_data['time']) + timedelta(hours=9,minutes=30)
	# print(main_data.head(5))
	# print(max(main_data['time']))



	call_data = main_data.loc[main_data['type'] == "CALL"]
	# print(call_data)
	call_data = call_data.loc[call_data['time'].dt.strftime('%Y-%m-%d') == todays_date]
	call_data.reset_index(inplace = True, drop = True)

	# print("call_data")
	# print(call_data)

	put_data = main_data.loc[main_data['type'] == "PUT"]
	put_data = put_data.loc[put_data['time'].dt.strftime('%Y-%m-%d') == todays_date]
	put_data.reset_index(inplace = True, drop = True)

	call_buy_and_sell_data = call_data.groupby(['time'])[['totaltradedvolume','totalbuyquantity','totalsellquantity']].sum().reset_index()
	put_buy_and_sell_data = put_data.groupby(['time'])[['totaltradedvolume','totalbuyquantity','totalsellquantity']].sum().reset_index()

# 	print("call_buy_and_sell_data")
# 	print(call_buy_and_sell_data)

	# print(call_data.columns)
	# print(call_data[['pchange','changeinopeninterest']])


	call_buy_and_sell_data['call_side'] = ['Buyers' if x>y else "Sellers" for x,y in zip(call_buy_and_sell_data['totalbuyquantity'],call_buy_and_sell_data['totalsellquantity'])]
	put_buy_and_sell_data['call_side'] = ['Buyers' if x>y else "Sellers" for x,y in zip(put_buy_and_sell_data['totalbuyquantity'],put_buy_and_sell_data['totalsellquantity'])]

	# call_data['value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(call_data['pchange'],call_data['changeinopeninterest'])]
	# put_data['value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(put_data['pchange'],put_data['changeinopeninterest'])]

	call_data['value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(call_data['pchange'],call_data['changeinopeninterest'])]
	put_data['value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(put_data['pchange'],put_data['changeinopeninterest'])]



	summarised_data = call_data.groupby(['time'])['value'].value_counts(normalize=True).reset_index(name='count')

	# print(summarised_data.tail())

	summarised_data = pd.DataFrame(summarised_data)

	summarised_data.reset_index(drop=True)
	print(summarised_data.tail(20))


	summarised_put_data = put_data.groupby(['time'])['value'].value_counts(normalize=True).reset_index(name='count')

	summarised_put_data = pd.DataFrame(summarised_put_data)

	summarised_put_data.reset_index(drop=True)
	print(summarised_put_data.tail(20))

	# print(call_buy_and_sell_data.tail(5))
	# print(put_buy_and_sell_data.tail(5))


	call_volume_data = call_data.groupby('time')['totaltradedvolume'].sum().reset_index(name='call_volume_total')
	put_volume_data = put_data.groupby('time')['totaltradedvolume'].sum().reset_index(name='put_volume_total')


	result = pd.merge(call_volume_data, put_volume_data, on="time")

	# print(result)

	result['pcr_ratio'] = (result['put_volume_total']/result['call_volume_total']).round(decimals = 2)

	result['call_previous_volume'] = result['call_volume_total'].shift(1)
	result['put_previous_volume'] = result['put_volume_total'].shift(1)
	result['current_call_volume'] = result['call_volume_total'] - result['call_previous_volume']
	result['current_put_volume'] = result['put_volume_total'] - result['put_previous_volume']
	# result['current_call_volume'] = result['current_call_volume'].astype(int)
	# result['current_put_volume'] = result['current_put_volume'].astype(int)
	# result = result[result['current_call_volume'].notna()]

	pd.set_option('display.max_rows', 500)
	pd.set_option('display.max_columns', 500)
	# print(summarised_data)
	summarised_data = summarised_data.loc[summarised_data['count'] >= 0.9]
	summarised_data.reset_index()

	# print("Latest summary : ")
	# print(summarised_data)

	summarised_put_data = summarised_put_data.loc[summarised_put_data['count'] >= 0.9]
	summarised_put_data.reset_index()
	# print(summarised_put_data)


	final_data = pd.DataFrame(pd.date_range("09:15", "15:30", freq="1min",tz='Asia/Kolkata'))
	# final_data = pd.DataFrame(pd.date_range("2022-01-07 09:15", "2022-01-07 15:30", freq="1min"))

	final_data.columns = ['Datetime']

# 	print(final_data)


	temp_df = pysqldf("select t1.Datetime,t2.value as Call_Interpretation,t3.value as Put_Interpretation,t4.pcr_ratio,t4.current_call_volume,t4.current_put_volume,t5.call_side as Call_Majority,t6.call_side as Put_Majority from final_data t1  left join summarised_data t2 on t1.Datetime = t2.time left join summarised_put_data t3 on t1.Datetime = t3.time left join result t4 on t1.Datetime = t4.time left join call_buy_and_sell_data t5 on t1.Datetime = t5.time left join put_buy_and_sell_data t6 on t1.Datetime = t6.time")
	# print("temp_df tail")

	

	# print(temp_df)

	temp_df = pysqldf("select *, row_number() over(order by current_call_volume desc) as call_volume_rank,row_number() over(order by current_put_volume desc) as put_volume_rank from temp_df")

	# print(temp_df.head(5))

	# pd.set_option('display.max_rows', 500)

	Signals_df = pysqldf("select *,case when (Call_Interpretation = 'Shorts Buildups' and Put_Interpretation = 'Long Buildups' and call_volume_rank <= 25 and put_volume_rank <= 25) then 'Sell' when (Call_Interpretation = 'Long Buildups' and Put_Interpretation = 'Shorts Buildups' and call_volume_rank <= 25 and put_volume_rank <= 25) then 'Buy' else '-' end as signal from temp_df")

	# print(Signals_df.head(5))

	call_data['time'] = pd.to_datetime(call_data['time'])
	call_data = call_data.drop_duplicates(keep='first')

	call_data.to_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+'call_data.csv',sep=',', header=True)

	put_data['time'] = pd.to_datetime(put_data['time'])
	put_data = put_data.drop_duplicates(keep='first')
	future_data['Datetime'] = pd.to_datetime(future_data['Datetime'])

	# print("future_data : ")
	# print(future_data)

	# print("call_data : ")
	# print(call_data)

	# print("put_data : ")
	# print(put_data)

	# print("call_buy_and_sell_data : ")
	# print(call_buy_and_sell_data)

	# print("call_data")
	# print(call_data)


	futures_df = pysqldf("select t1.Datetime,t1.Strike_Price,t1.Volume as future_volume,t2.totaltradedvolume as call_traded_volume,t2.pchange as call_pchange,t2.changeinopeninterest as call_changeinopeninterest,t3.totaltradedvolume as put_traded_volume,t3.pchange as put_pchange,t3.changeinopeninterest as put_changeinopeninterest,round(t3.totaltradedvolume*1.00/ t2.totaltradedvolume,2) as pcr_ratio, row_number() over(order by t1.Volume desc) as fut_volume_rank, row_number() over(order by t2.totaltradedvolume desc) as call_volume_rank,row_number() over(order by t3.totaltradedvolume desc) as put_volume_rank,t5.call_side as Call_Majority,t6.call_side as Put_Majority from future_data t1 left join call_data t2 on t1.Datetime = t2.time and t1.Strike_Price = t2.strikeprice left join put_data t3 on t1.Datetime = t3.time and t1.Strike_Price = t3.strikeprice left join call_buy_and_sell_data t5 on t1.Datetime = t5.time left join put_buy_and_sell_data t6 on t1.Datetime = t6.time")

	# print("futures_df : ")
	# print(futures_df)

	futures_df.to_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+'futures_df.csv',sep=',', header=True)

	temp_fut_data = pysqldf("select t1.Datetime,t1.Strike_Price,t1.Volume as future_volume,t2.totaltradedvolume as call_traded_volume,t2.pchange as call_pchange from future_data t1 left join call_data t2 on t1.Datetime = t2.time and t1.Strike_Price = t2.strikeprice")

	temp_fut_data.to_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+'Futures_testing.csv',sep=',', header=True)

	# print(futures_df['put_pchange'])
	# print(futures_df['put_changeinopeninterest'])
	futures_df['put_pchange'] = futures_df['put_pchange'].fillna(0)
	futures_df['put_changeinopeninterest'] = futures_df['put_changeinopeninterest'].fillna(0)
	futures_df['put_pchange'] = futures_df['put_pchange'].astype('int')
	futures_df['put_changeinopeninterest'] = futures_df['put_changeinopeninterest'].astype('int')
	# print(futures_df.tail(5))
	# print(futures_df.columns)
	futures_df['call_pchange'] = futures_df['call_pchange'].fillna(0)
	futures_df['call_changeinopeninterest'] = futures_df['call_changeinopeninterest'].fillna(0)
	print(futures_df[['call_pchange','call_changeinopeninterest']])
	futures_df['call_value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(futures_df['call_pchange'],futures_df['call_changeinopeninterest'])]
	futures_df['put_value'] = ['Long Buildups' if x>0 and y>0 else 'Long Unwinding' if x<0 and y <0 else 'Shorts Buildups' if x < 0 and y > 0 else 'Shorts Coverings' if x > 0 and y < 0 else '-' for x,y in zip(futures_df['put_pchange'],futures_df['put_changeinopeninterest'])]

	# print("Futures Data")

	# print(futures_df.head(5))

	Signals_df = Signals_df.sort_values(by='Datetime')
	futures_df = futures_df.sort_values(by='Datetime')

	Signals_df['Stock'] = symbol
	futures_df['Stock'] = symbol

	print(futures_df.tail(5))

	try:
	    if 'options_signals' in list_of_collections:
	        collection = db["options_signals"]
	        print("Collection exists")
	        
	        x = collection.delete_many({"Stock":symbol})
	        # x = collection.delete_many({})
	        print(x.deleted_count, " documents deleted.")
	except pymongo.errors.OperationFailure:  # If the collection doesn't exist
	    print("This collection doesn't exist")

	collection = db["options_signals"]
    
	print("Before inserting")
	print(Signals_df)

	collection.insert_many(Signals_df.to_dict('records'))

	try:
	    if 'futures_options_signals' in list_of_collections:
	        collection = db["futures_options_signals"]
	        print("Collection exists")

	        x = collection.delete_many({"Stock":symbol})
	        # x = collection.delete_many({})
	        print(x.deleted_count, " documents deleted.")
	except pymongo.errors.OperationFailure:  # If the collection doesn't exist
	    print("This collection doesn't exist")

	collection = db["futures_options_signals"]
    
# 	print(futures_df)

	collection.insert_many(futures_df.to_dict('records'))

	Signals_df.to_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+str(todays_date)+'_Options_Signals.csv',sep=',', header=True)
	futures_df.to_csv('/home/sjonnal3/Hate_Speech_Detection/Trading_Application/Options_data/'+str(symbol)+'/'+str(todays_date)+'_Futures_Options_Signals.csv',sep=',', header=True)


	final_signals_df = Signals_df.loc[Signals_df['signal'] != '-']

	# print(final_signals_df)

	final_signals_df['Datetime'] = pd.to_datetime(final_signals_df['Datetime']).dt.strftime('%Y-%m-%d %H:%M:00')



# 	if not final_signals_df.empty:
# 		if talk_Command == "Yes":
# 			os.system("say 'Check the Options Chain Analysis'")
# 			time = final_signals_df.iloc[0,0]
# 			direction = final_signals_df.iloc[0,10]
# 			command = "Signal to "+str(symbol)+" at "+ str(time) + " for "+ str(direction)
# 			print(command)
# 			os.system("say "+str(command))