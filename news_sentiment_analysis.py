import requests
import json
import pandas as pd
from datetime import timedelta, date
import datetime
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import string
import re
import nltk
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
# from sklearn.feature_extraction.text import TafidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer # notice the spelling with the f before Vectorizer
from sklearn.naive_bayes import MultinomialNB # notice the Caps on the M
from sklearn.pipeline import make_pipeline
from nltk.stem import WordNetLemmatizer
import boto3
import json
from pytz import timezone

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import en_core_web_sm
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.lang.en import English

from pymongo import MongoClient
from pymongo.server_api import ServerApi


start_time = datetime.now()
print(start_time)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')


server_api = ServerApi("1")

client = MongoClient(
    "mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE",
    server_api=server_api,
)
db = client["titania_trading"]

collection = db["New_Sentiment_Analysis"]

def fetch_news_data(date_filter):

	news_dataframe = pd.DataFrame()
	row_append = 0
	counter = 0

	if date_filter == 'today':

		economic_times = "https://economictimes.indiatimes.com/headlines.cms"

		headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
		                         'like Gecko) '
		                         'Chrome/80.0.3987.149 Safari/537.36',
		           'accept-language': 'en,gu;q=0.9,hi;q=0.8', 'accept-encoding': 'gzip, deflate, br'}

		news_page = requests.get(economic_times)

		soup = BeautifulSoup(news_page.content, "html.parser")

		content = soup.body.find(id="pageContent")

		print("Reading the raw website")

		for ul in content.findAll('ul'):
			for li in ul.findAll('li'):
				news_link = li.find('a').get('href')
				try:
					news_content = li.find('a').contents[1]
				except:
					news_content = li.find('a').contents[0]
					# print(news_content)
				news_dataframe.loc[row_append,"Content"] = news_content
				news_dataframe.loc[row_append,"Link"] = "https://economictimes.indiatimes.com" + str(news_link)

				row_append = row_append + 1
			counter = counter + 1

		print("Reading website complete")

		print("Opening the news links")

		for line in range(0,len(news_dataframe)):
			news_link = news_dataframe.loc[line,'Link']
			try:
				news_link_page = requests.get(news_link)
				#         print(news_link)
				news_soup = BeautifulSoup(news_link_page.content, "html.parser")

				print(len(news_soup.body.findAll('script')))

				for script_len in range(0,len(news_soup.body.findAll('script'))):

					script = news_soup.body.findAll('script')[script_len]
# 					print(script)
					if 'articleBody' in script.text:
						try:
							news_text = json.loads(news_soup.body.findAll('script')[script_len].text)
							news_dataframe.loc[line,"Keywords"] = str(news_text['keywords'])
							news_dataframe.loc[line,"description"] = news_text['description']
							news_dataframe.loc[line,"articleSection"] = news_text['articleSection']
							news_dataframe.loc[line,"articleBody"] = news_text['articleBody']
							news_dataframe.loc[line,"datePublished"] = news_text['dateModified']
							news_dataframe.loc[line,"author"] = str(news_text['author'])
						except:
							print("Exception occured")
			except:
				print("unable to read html")

		print("News Content read successfully")
		return news_dataframe


news_dataframe = fetch_news_data('today')

news_dataframe['datePublished'] = pd.to_datetime(news_dataframe['datePublished'], errors='coerce')
news_dataframe['Date'] = news_dataframe['datePublished'].dt.strftime("%Y-%m-%d")


final_news_df = news_dataframe[['Date','Content','Link','Keywords','description','articleSection','articleBody','datePublished','author']]

final_news_df['Date'].replace('', np.nan, inplace=True)

final_news_df.dropna(subset=['Date'], inplace=True)

final_news_df = final_news_df.sort_values(by='datePublished',ascending=False)

final_news_df.to_csv('/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/New_Sentiment_Analysis/'+str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d"))+".csv")

final_news_df = pd.read_csv('/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/New_Sentiment_Analysis/'+str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d"))+".csv")

string.punctuation

#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


#defining function for tokenization
def tokenization(text):
    tokens = re.split('W+',text)
    return tokens

#defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#defining a function for stemming
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

#storing the puntuation free text
final_news_df['clean_msg']= final_news_df['articleBody'].astype(str).apply(lambda x:remove_punctuation(x))
final_news_df.head()

final_news_df['msg_lower']= final_news_df['clean_msg'].apply(lambda x: x.lower())


#applying function to the column
final_news_df['msg_tokenied']= final_news_df['msg_lower'].apply(lambda x: tokenization(x))

#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]


#applying the function
final_news_df['no_stopwords']= final_news_df['msg_tokenied'].apply(lambda x:remove_stopwords(x))


#defining the object for stemming
porter_stemmer = PorterStemmer()



final_news_df['msg_stemmed']=final_news_df['no_stopwords'].apply(lambda x: stemming(x))



#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()




final_news_df['msg_lemmatized']=final_news_df['no_stopwords'].apply(lambda x:lemmatizer(x))






punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
desc = final_news_df['msg_lemmatized'].astype(str).values
vectorizer = TfidfVectorizer(stop_words = stop_words)
X = vectorizer.fit_transform(desc)


word_features = vectorizer.get_feature_names()

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')


vectorizer2 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize)
X2 = vectorizer2.fit_transform(desc)
word_features2 = vectorizer2.get_feature_names()


vectorizer3 = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, max_features = 1000)
X3 = vectorizer3.fit_transform(desc)
words = vectorizer3.get_feature_names()


cluster_data = final_news_df[['Date','articleBody', 'msg_lemmatized', 'description','datePublished']]

# cluster_data.columns
cluster_data["Sentiment"] = ""
cluster_data["Mixed_Sentiment"] = ""
cluster_data["Negative_Sentiment"] = ""
cluster_data["Neutral_Sentiment"] = ""
cluster_data["Positive_Sentiment"] = ""



s3 = boto3.resource('s3',
         aws_access_key_id="AKIATKSEH532EWWG4BM7",
         aws_secret_access_key= "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")

comprehend = boto3.client(service_name='comprehend', region_name='ap-south-1',aws_access_key_id="AKIATKSEH532EWWG4BM7",
         aws_secret_access_key= "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")



for cli in range(0,len(cluster_data)):
    print(cli)
#     text = cluster_data.loc[cli,"articleBody"]
    text = cluster_data.loc[cli,"description"]
#     print(str(text))
    try:
        #Entity Extraction
        entities = comprehend.detect_entities(Text = text, LanguageCode = 'en') #API call for entity extraction
        entities = entities['Entities'] #all entities
#         print(text)
#         print(entities)
        textEntities = [dict_item['Text'] for dict_item in entities] #the text that has been identified as entities
        typeEntities = [dict_item['Type'] for dict_item in entities] #the type of entity the text is
#         print(textEntities)
#         print(typeEntities)
        json_result = json.dumps(comprehend.detect_sentiment(Text=text, LanguageCode='en'), sort_keys=True, indent=4)
#         print(json_result)
        cluster_data.loc[cli,"Sentiment"] = json.loads(json_result)['Sentiment']
        cluster_data.loc[cli,"Mixed_Sentiment"] = json.loads(json_result)['SentimentScore']['Mixed']
        cluster_data.loc[cli,"Negative_Sentiment"] = json.loads(json_result)['SentimentScore']['Negative']
        cluster_data.loc[cli,"Neutral_Sentiment"] = json.loads(json_result)['SentimentScore']['Neutral']
        cluster_data.loc[cli,"Positive_Sentiment"] = json.loads(json_result)['SentimentScore']['Positive']
    except:
        print("Exception")




analyzer = SentimentIntensityAnalyzer()

for row in range(0,len(cluster_data)):
    news = cluster_data.loc[row,'articleBody']
#     news_list.append(news)
#     print(news)
#     print(type(news))
    analyzer = SentimentIntensityAnalyzer().polarity_scores(str(news))
    cluster_data.loc[row,'neg'] = analyzer['neg']
    cluster_data.loc[row,'neu'] = analyzer['neu']
    cluster_data.loc[row,'pos'] = analyzer['pos']
    cluster_data.loc[row,'compound'] = analyzer['compound']



nltk.download('punkt')


nlp=English()
# import spacy
# nlp=spacy.load("en_core_web_sm")

nlp.add_pipe('spacytextblob')

for row in range(0,len(cluster_data)):
#     print(row)
    news = str(cluster_data.loc[row,'articleBody'])
    doc = nlp(news)
    sentiment = doc._.blob.polarity
    sentiment = round(sentiment,2)

    if sentiment > 0:
        cluster_data.loc[row,'scapy_sentiment'] = "POSITIVE"
    else:
        cluster_data.loc[row,'scapy_sentiment'] =  "NEGATIVE"
        
    positive_words = []
    negative_words = []
    
    for x in doc._.blob.sentiment_assessments.assessments:
        if x[1] > 0:
            positive_words.append(x[0][0])
        elif x[1] < 0:
            negative_words.append(x[0][0])
        else:
            pass
    cluster_data.loc[row,'scapy_positive_words'] = str(positive_words)
    cluster_data.loc[row,'scapy_negative_words'] = str(negative_words)



cluster_data['articleBody'] = cluster_data['articleBody'].astype(str)
cluster_data['description'] = cluster_data['description'].astype(str)


df = cluster_data[['articleBody','description','Date']]

df_array = np.array(df)
df_list = list(df_array[:,1]) 

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
#tokenize text to be sent to model
inputs = tokenizer(df_list, padding = True, truncation = True, return_tensors='pt')
# print(inputs)
outputs = model(**inputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

model.config.id2label

#Tweet #Positive #Negative #Neutral
positive = predictions[:, 0].tolist()
negative = predictions[:, 1].tolist()
neutral = predictions[:, 2].tolist()

table = {'Headline':df_list,
         'articleBody':cluster_data['articleBody'],
         'Date':df['Date'],
         'Sentiment':cluster_data['Sentiment'],
         'Mixed_Sentiment':cluster_data['Mixed_Sentiment'],
         'Negative_Sentiment':cluster_data['Negative_Sentiment'],
         'Neutral_Sentiment':cluster_data['Neutral_Sentiment'],
         'Positive_Sentiment':cluster_data['Positive_Sentiment'],
         'neg':cluster_data['neg'],
         'neu':cluster_data['neu'],
         'pos':cluster_data['pos'],
         'compound':cluster_data['compound'],
         'scapy_sentiment':cluster_data['scapy_sentiment'],
         'scapy_positive_words':cluster_data['scapy_positive_words'],
         'scapy_negative_words':cluster_data['scapy_negative_words'],
         'Description':cluster_data['description'],
         'datePublished':cluster_data['datePublished'],
         "Positive":positive,
         "Negative":negative, 
         "Neutral":neutral}
      
df2 = pd.DataFrame(table, columns = ["Headline",'articleBody',"Date",'Sentiment',
       'Mixed_Sentiment', 'Negative_Sentiment', 'Neutral_Sentiment',
       'Positive_Sentiment', 'neg', 'neu', 'pos', 'compound',
       'scapy_sentiment', 'scapy_positive_words', 'scapy_negative_words','datePublished', "Positive", "Negative", "Neutral"])


# df2['Sentiment'].value_counts()
df2['Vader_Sentiment'] = np.where(df2['compound'] >=  0.05, 'POSITIVE', np.where(df2['compound'] <= -0.05,'NEGATIVE','NEUTRAL'))

# df2['Finbert_Sentiment'] = np.where(df2['Positive'] >=  df2['Negative'] and df2['Positive'] >=  df2['Neutral'],'Positive',np.where(df2['Negative'] >=  df2['Positive'] and df2['Negative'] >=  df2['Neutral'],'Negative','Neutral'))

for sent in range(0,len(df2)):
    if (df2.loc[sent,'Positive'] >= df2.loc[sent,'Negative'] and df2.loc[sent,'Positive'] >= df2.loc[sent,'Neutral']):
        df2.loc[sent, 'Finbert_Sentiment'] = 'POSITIVE'
    elif (df2.loc[sent,'Negative'] >= df2.loc[sent,'Positive'] and df2.loc[sent,'Negative'] >= df2.loc[sent,'Neutral']):
        df2.loc[sent, 'Finbert_Sentiment'] = 'NEGATIVE'
    else:
        df2.loc[sent, 'Finbert_Sentiment'] = 'NEUTRAL'


server_api = ServerApi("1")

client = MongoClient(
    "mongodb+srv://Titania:Mahadev@cluster0.zq3w2cn.mongodb.net/titania_trading?ssl=true&ssl_cert_reqs=CERT_NONE",
    server_api=server_api,
)
db = client["titania_trading"]

collection = db["New_Sentiment_Analysis"]

# x = collection.delete_many({"Stock":"Nifty","instrumenttype":"FUTIDX"})



final_sentiment_df = df2[['Date','Headline','articleBody','datePublished','Sentiment','Vader_Sentiment','scapy_sentiment','Finbert_Sentiment']]

final_sentiment_df['Execution_Date'] = str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d"))

collection.insert_many(final_sentiment_df.to_dict('records'))

final_sentiment_df.to_csv('/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/New_Sentiment_Analysis/'+str(datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d"))+"_Final_Analysis.csv")


end_time = datetime.now()

print(end_time)

print('Duration: {}'.format(end_time - start_time))
