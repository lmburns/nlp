# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Sentiment Analysis of Tweets

# +
import tweepy
from tweepy import Cursor
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 300
plt.style.use(['dark_background'])
# %matplotlib inline

# +
# Authentication requirements
app = pd.read_csv('../keys.csv')

consumerKey = app['Key'][0]
consumerSecret = app['SecretKey'][0]
accessToken = app['Token'][0]
accessTokenSecret = app['SecetToken'][0] # Misspelled secret on accident

# +
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

authenticate.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth_handler=authenticate, wait_on_rate_limit=True)
# -

# ### Scraping Twitter

# +
text_query = 'great reset'
max_tweets = 2000
 
# Creation of query method using parameters
tweets = tweepy.Cursor(api.search, q=text_query).items(max_tweets)
 
tweet_list = [[tweet.text, tweet.created_at, tweet.id_str, tweet.user.name, 
               tweet.user.screen_name, tweet.user.location, tweet.user.description, 
               tweet.user.verified, tweet.user.followers_count, tweet.user.friends_count, 
               tweet.user.favourites_count, tweet.user.statuses_count, 
               tweet.user.listed_count, tweet.user.created_at] for tweet in tweets]
 
tweet_df = pd.DataFrame(tweet_list)

tweet_df.shape

# +
cols = ['text', 'tweet_created', 'tweet_id', 'username', 'screen_name', 'location', 'user_descr', 'verified', 'follower_num', 'friends_num', 'favs_num', 'status_num', 'listed_num', 'user_created']

tweets_df.columns = cols

# tweet_df.to_csv('great-reset-1000.csv', index=False, header=tweet_df.columns.values)
tweet_df.head()
# -

# - `tweet.text`: text content of tweet
# - `tweet.created_at`: date tweet was created_at
# - `tweet.id_str`: id of tweet
# - `tweet.user.screen_name`: username of tweets author
# - `tweet.coordinates`: geographic location
# - `tweet.place`; where user signed up when created 
# - `tweet.retweet_count`: count of retweets
# - `tweet.favorite_count`: count of favorites
# - `tweet.source`: source
# - `tweet.in_reply_to_user_id_str`: if a tweet is a reply 
# - `tweet.is_quote_status`: if a tweet is a quote tweet 

# ### Import Saved CSV

df = pd.read_csv('../great-reset-2000.csv')
df.head()


# +
def cleaned(df, col):
    df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')) # Remove emojis
    df[col] = df[col].replace(r'@[\w]+', '', regex=True).replace(r'#', '', regex=True) # Remove mentions
    df[col] = df[col].replace(r'RT[\s]+', '', regex=True) # Remove retweets
    df[col] = df[col].replace(r'https?:\/\/(www\.)?\S', '', regex=True) # Remove https links
    df[col] = df[col].replace(r'.co\/(\w)+', '', regex=True).replace(r':', '', regex=True) # Remove co links
    df[col] = df[col].replace(r'https…', '', regex=True) # edge case
    df[col] = df[col].replace(r'\n', ' ', regex=True) # Remove newline

    df = df.applymap(lambda x: x.strip() if type(x) == str else x).astype(str)
    # Had to use this for the 'practice.csv', so left it in just in case
    df_cleaned = df[~df['text'].str.contains('nan')]  
    df = df_cleaned.copy()

    return df[col]

df['text'] = cleaned(df, 'text')

df.head()


# -

# ## TextBlob

# +
def text_blob(df, col):
    testimonial = [TextBlob(str(item)) for item in df[col]]
    blobs = [item.sentiment for item in testimonial]
    blob_df = pd.DataFrame(blobs)
    df_combined = pd.concat([df, blob_df], axis=1)
    return df_combined

df = text_blob(df, 'text')
df.head()
# -

# ## WordCloud

# +
# String of all tweets joined together
all_words = ' '.join([str(tweet) for tweet in df['text']])
# Remove 'great reset' (the search query on twitter)
words = re.sub(r'great reset', '', all_words, flags=re.IGNORECASE)

word_cloud = WordCloud(height=400, width=600, max_font_size=50,
                      colormap='Dark2').generate(words)

plt.figure(figsize=(20, 10), dpi=300)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off');


# -

# ### Classifying Polarity

# +
def classify(polarity):
    if polarity < 0:
        return 'neg'
    elif polarity == 0:
        return 'neu'
    else:
        return 'pos'

df['classification'] = df['polarity'].apply(classify)
df.head()
# -

# ### Exploratory Data Analysis

df['classification'].value_counts()

plt.figure(figsize=(10, 6), dpi=300)
sns.countplot(x='classification', data=df, palette='Dark2_r').set_title('Tweet Sentiment Classification Value Counts');

# ### Negative Tweet Previews

df.loc[df['polarity'] < 0, ['text', 'polarity', 'classification']].drop_duplicates(keep='first').sort_values(by='polarity')[:10]

# ### Positive Tweet Previews

df.loc[df['polarity'] > 0, ['text', 'polarity', 'classification']].drop_duplicates(keep='first').sort_values(by='polarity', ascending=False)[:10]

# ### Polarity vs Subjectivity

plt.figure(figsize=(10, 6), dpi=300)
sns.scatterplot(x=df['polarity'], y=df['subjectivity'], color='BlueViolet').set_title('Polarity vs Subjectivity of Tweets');

# ### Distribution of Polarity

plt.figure(figsize=(10, 6), dpi=300)
sns.boxplot(x='polarity', data=df, palette='Dark2').set_title('Polarity Distribution');

# ### Verified Twitter Positive vs Negative

# +
df_pn = df[df['classification'] != 'neu']

plt.figure(figsize=(10, 6), dpi=300)
sns.boxplot(x='verified', y='polarity', hue='classification', 
            data=df_pn, palette='Dark2').set_title('Positive & Negative Sentiment of Verified Twitter Users');
# -


