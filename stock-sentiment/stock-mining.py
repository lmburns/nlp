#!/usr/bin/env python
# coding: utf-8

# In[78]:


import requests
from bs4 import BeautifulSoup as bs
import json
from string import punctuation
from get_all_tickers import get_tickers as gt
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cufflinks as cf
import chart_studio.plotly as py
import plotly.express as px
import plotly.io as pio

from plotly.offline import download_plotlyjs, init_notebook_mode
from plotly.offline import plot, iplot

pio.templates.default = 'plotly_dark'
init_notebook_mode(connected=True)
plt.style.use(['dark_background'])


# ## Scraping Yahoo Finance

# In[76]:


# Top 50 Stock Tickers
tickers = gt.get_biggest_n_tickers(50)

def get_data(ticker):
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:82.0) Gecko/20100101 Firefox/82.0'}
    
    url = f'https://finance.yahoo.com/quote/{ticker}/'
    r = requests.get(url, headers=headers)
    soup = bs(r.text, 'html.parser')

    stock_stats = soup.find_all('td', {'class': 'Ta(end) Fw(600) Lh(14px)'})
    stocks = {
    'ticker' : soup.find('h1', {'class': 'D(ib) Fz(18px)'}).text.split('(')[-1].strip(punctuation),
    'closing' : stock_stats[0].text,
    'opening' : stock_stats[1].text,
    'days_range' : stock_stats[4].text,
    'volume' : stock_stats[6].text,
    'fair_value' : soup.find('div', {'class': 'Fw(b) Fl(end)--m Fz(s) C($primaryColor'}).text
    }
    return stocks

stock_data = []
for idx, ticker in enumerate(tickers):
    stock_data.append(get_data(ticker))
    time.sleep(2)
    print(f'Finished with number {idx} -- {ticker}')


# In[77]:


with open('11-15-yahoo-stock.json', 'w') as f:
    json.dump(stock_data, f)


# In[89]:


yahoo = pd.DataFrame(stock_data)
# yahoo.to_csv('11-15-yahoo-stock.csv', index=False, header=df.columns.values)


# ## Scraping Tingo

# In[83]:


# I'm using Docker to access this website
local = 'http://localhost:8050/render.html'

news_list = []

for idx, ticker in enumerate(tickers):
    url = f'https://www.tiingo.com/{ticker}/overview'
    r = requests.get(local, params={'url': url, 'wait': 2})
    soup = bs(r.content, 'html.parser')
    
    # News article headlines
    headlines = soup.find_all('div', class_='headline')
    headline_list = [headline.find('a').text for headline in headlines]

    # News article content that is previewed
    articles = soup.find_all('div', class_='lede')
    article_list = [article.text for article in articles]

    dates = soup.find_all('div', class_='date-source')
    date_list = [date.text.split('|')[0] for date in dates]

    sources = soup.find_all('div', class_='date-source')
    source_list = [source.text.split('|')[1] for source in sources]

    news = {
        'ticker': ticker,
        'headline': headline_list,
        'article': article_list,
        'date': date_list,
        'source': source_list,
    }
    
    temp_df = pd.DataFrame(news)
    news_list.append(temp_df)
    
    time.sleep(2)
    print(f'Finished number {idx} -- {ticker}')


# In[195]:


# Combine the list of dataframes
news_df = pd.concat(news_list)
# Index did not line up, so reset it
news_df.reset_index(inplace=True, drop=True)
# Rename column because I'm going to use date for something else
news_df.rename(columns={'date': 'date_time'}, inplace=True)
# news_df.to_csv('11-15-tingo-dirty.csv', index=False, header=news_df.columns.values)


# In[196]:


# Create a datetime object to access the day of the week
news_df['datetime'] = ['2020-' + item.strip() for item in news_df['date_time']] 
news_df['datetime'] = pd.to_datetime(news_df['datetime'], format='%Y-%b-%d %H:%M:%p')
news_df['dayofweek'] = [item.dayofweek for item in news_df['datetime']]
news_df['date'] = pd.to_datetime(news_df['datetime'].dt.date)
news_df['time'] = news_df['datetime'].dt.time
news_df.drop('date_time', axis=1, inplace=True)

# news_df.to_csv('11-15-tingo-stock.csv', index=False, header=news_df.columns.values)
news_df.head()


# In[209]:


combined_df = pd.merge(news_df, yahoo, on='ticker')
# Replace commas to convert to an integer
combined_df['volume'] = combined_df['volume'].replace(r',', '', regex=True).astype(int)
combined_df['fair_value'] = combined_df['fair_value'].astype('category')

# combined_df.to_csv('11-15-combined-stock.csv', index=False, header=combined_df.columns.values)
combined_df.head()


# ## Sentiment Analysis using NLTK

# In[210]:


# Initialization
analyzer = SentimentIntensityAnalyzer()
# Apply the algorithm to the headline column
analyzer_scores = [analyzer.polarity_scores(item)['compound'] for item in combined_df['headline']]
# Create a dataframe from the compounded polarity scores
analyzer_df = pd.DataFrame(analyzer_scores, columns=['compound'])
# combine the two dataframes
analyzer_news = pd.concat([combined_df, analyzer_df], axis=1)
# Remove entries where the algorithm either did not find anything or was not able to calculate anything
analyzer_news = analyzer_news[analyzer_news['compound'] != 0].reset_index(drop=True)
analyzer_news.head()


# ## NLTK - `dayofweek` using Plotly

# In[212]:


# Group by the ticker and the day of the week, creating a multi-index
mean_day = analyzer_news.groupby(['ticker', 'dayofweek']).mean()
# Unstack, the innermost index level 'unstacks' across the columns
mean_day = mean_day.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_day = mean_day.xs('compound', axis=1).T
# Plotly colors
sunset = px.colors.sequential.Agsunset

px.bar(mean_day, barmode='group', color_discrete_sequence=sunset,
      title='Average Week Day Sentiment of Stock Headlines',
      labels={'value': 'Sentiment', 'dayofweek': 'Day of Week'})


# ## NLTK - `date` using Plotly

# In[232]:


# Filter the dates
analyzer_filtered = analyzer_news[analyzer_news['date'] > '2020-10-31']
# Group by the ticker and the date, creating a multi-index
mean_date = analyzer_filtered.groupby(['ticker', 'date']).mean()
# Drop the dayofweek because we don't need it
mean_date = mean_date.drop('dayofweek', axis=1)
# Unstack, the innermost index level 'unstacks' across the columns
mean_date = mean_date.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_date = mean_date.xs('compound', axis=1).T

sunset = px.colors.sequential.Sunsetdark

px.bar(mean_date, barmode='relative', color_discrete_sequence=sunset,
      title='Average Sentiment of Stock Headlines',
      labels={'value': 'Sentiment', 'date': 'Date'})


# ## Sentiment Analysis using TextBlob

# In[221]:


# Creating a TextBlob item for each article's content
testimonial = [TextBlob(item) for item in combined_df['article']]
# Get the sentiment for each textblob object
blob = [item.sentiment for item in testimonial]
# Convert to a dataframe
blob_df = pd.DataFrame(blob)
# Combine the blob_df and full news_df
blob_news = pd.concat([combined_df, blob_df], axis=1)
# Remove entries where the algorithm either did not find anything or was not able to calculate anything
blob_news = blob_news[blob_news['polarity'] != 0].reset_index(drop=True)
blob_news.head()


# ### Polarity based on `dayofweek` using Plotly

# In[222]:


# blob_news.to_csv('11-15-tingo-blob.csv', header=blob_news.columns.values, index=False)


# In[256]:


# Group by the ticker and the day of the week, creating a multi-index
mean_day = blob_news.groupby(['ticker', 'dayofweek']).mean()
# Unstack, the innermost index level 'unstacks' across the columns
mean_day = mean_day.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_day = mean_day.xs('polarity', axis=1).T
# Plotly colors
agg = px.colors.sequential.Aggrnyl

px.bar(mean_day, barmode='group', color_discrete_sequence=agg,
      title='Average Week Day Polarity of Stock Headlines',
      labels={'value': 'Polarity', 'dayofweek': 'Day of Week'})


# ### Polarity based on `date` using Plotly

# In[255]:


# Filter the dates
blob_filtered = blob_news[(blob_news['date'] > '2020-10-31') & (blob_news['date'] < '2020-11-16')]
# Group by the ticker and the date, creating a multi-index
mean_date = blob_filtered.groupby(['ticker', 'date']).mean()
# Drop the dayofweek because we don't need it
mean_date = mean_date.drop('dayofweek', axis=1)
# Unstack, the innermost index level 'unstacks' across the columns
mean_date = mean_date.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_date = mean_date.xs('polarity', axis=1).T

rainbow = px.colors.sequential.Rainbow

px.bar(mean_date, barmode='relative', color_discrete_sequence=rainbow,
      title='Average Polarity of Stock Headlines',
      labels={'value': 'Polarity', 'date': 'Date'})


# ### Subjectivity based on `dayofweek` using Plotly

# In[254]:


# Group by the ticker and the day of the week, creating a multi-index
mean_day = blob_news.groupby(['ticker', 'dayofweek']).mean()
# Unstack, the innermost index level 'unstacks' across the columns
mean_day = mean_day.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_day = mean_day.xs('subjectivity', axis=1).T
# Plotly colors
purp = px.colors.sequential.Purp

px.bar(mean_day, barmode='group', color_discrete_sequence=purp,
      title='Average Week Day Polarity of Stock Headlines',
      labels={'value': 'Subjectivity', 'dayofweek': 'Day of Week'})


# ### Subjectivity based on `date` using Plotly

# In[253]:


# Filter the dates
blob_filtered = blob_news[(blob_news['date'] > '2020-10-31') & (blob_news['date'] < '2020-11-16')]
# Group by the ticker and the date, creating a multi-index
mean_date = blob_filtered.groupby(['ticker', 'date']).mean()
# Drop the dayofweek because we don't need it
mean_date = mean_date.drop('dayofweek', axis=1)
# Unstack, the innermost index level 'unstacks' across the columns
mean_date = mean_date.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_date = mean_date.xs('subjectivity', axis=1).T

sunset = px.colors.sequential.Sunset

px.bar(mean_date, barmode='relative', color_discrete_sequence=sunset,
      title='Average Polarity of Stock Headlines',
      labels={'value': 'Subjectivity', 'date': 'Date'})


# In[252]:


# Group by the ticker and the day of the week, creating a multi-index
mean_day = blob_news.groupby(['ticker', 'date']).mean()
# Unstack, the innermost index level 'unstacks' across the columns
mean_day = mean_day.unstack()
# Get a cross-section of the above data, and transpose it to make the columns the ticker names
mean_day = mean_day.xs('polarity', axis=1).T

px.box(mean_day, title='Distribution of the Polarity of Stock Headlines',
      labels={'value': 'Polarity'})


# In[ ]:




