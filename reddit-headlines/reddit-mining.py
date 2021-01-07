#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

import praw
import tqdm as tqdm
from IPython import display
from pprint import pprint
from collections import defaultdict
import pickle
from string import punctuation
from collections import Counter

plt.style.use(['dark_background'])

# ----
# ----
# ## Scraping data from Reddit
# ----
# ----

reddit = praw.Reddit(client_id='##########',
                     client_secret='##########',
                     username='##########',
                     password='##########',
                     user_agent='##########')

headlines = {}
subreddit = reddit.subreddit('worldnews').top(limit=None)

for submission in subreddit:
    if not submission.stickied:

        submission.comment_sort = 'top'
        submission.comment_limit = None

        comments = {}

        for i, top_comment in enumerate(submission.comments):
            if isinstance(top_comment, praw.models.MoreComments):
                continue
            comments[f'comment {i+1}'] = top_comment.body
            #display.clear_output()
            print(f'Comments: {len(comments)}')

    headlines[submission.title] = comments

    display.clear_output()
    print(f'Headlines: {len(headlines)}')

# ----
# ### Save and Reload Headlines Dictionary
# ----

def save_obj(obj, name ):
    with open('data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# pickle.dump(headlines, open('headlines.pkl', 'wb'))
data = load_obj('headlines')
type(data)

next(iter(data.keys()))

df = pd.DataFrame(data).T
df.columns = df.columns.str.replace(' ', '_')
df = df.reset_index().rename(columns={'index': 'headline'})
df.head(1)

# ----
# ----
# ## Cleaning the Data
# ----
# ----

df = df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')) # Remove emojis
df = df.replace(r'\/r\/', '', regex=True)
df = df.replace(r'\/u\/', '', regex=True)
df = df.replace(r'https?:\/\/(www\.)?\S', '', regex=True) # Remove https links
df = df.replace(r'tatus\/\w+\?s=\d+', '', regex=True)
df = df.replace(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', regex=True)
df = df.replace(r'([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', regex=True)
df = df.replace(r'(\w)+\.com\/\w+\/\w?', '', regex=True)
df = df.replace(r'.co\/(\w)+', '', regex=True).replace(r':', '', regex=True) # Remove co links
df = df.replace(r'https…', '', regex=True) # edge case
df = df.replace(r'\n', ' ', regex=True) # Remove newline
# df = df.applymap(lambda x: x.strip() if type(x) == str else x).astype(str)

df = df.replace(r'[deleted]', 'N/A').replace(r'[removed]', 'N/A')
df = df.replace(r'>', '', regex=True)
df = df.replace(r'\*\*', '', regex=True).replace(r'\*', '', regex=True)
df = df.fillna('N/A').replace(r'nan', 'N/A', regex=True)
df.head(1)

# ----
# ----
# ## NLTK for Headline
# ----
# ----

sia = SentimentIntensityAnalyzer()

scores = [sia.polarity_scores(item) for item in df['headline']]
scores_df = pd.DataFrame(scores)
sia_df = pd.concat([df, scores_df], axis=1)

cols = sia_df.columns.to_list()
sia_df = sia_df[[cols[0]] + cols[-4:] + cols[1:-4]]

# sia_df.to_csv('sia_df-2.csv', index=False, header=sia_df.columns.values)
sia_df.head(2)

# ----
# ----
# ## NLTK for each Comment
# ----
# ----

df = pd.read_csv('data/sia_df-24.csv', lineterminator='\n')
df.head(2)

sia = SentimentIntensityAnalyzer()

comment_compounds = []

for idx, value in enumerate(range(1, df.shape[1])):
    score = [sia.polarity_scores(str(value))['compound'] for value in df.iloc[:, idx]]
    score_df = pd.DataFrame(score, columns=[f'{idx}_compound'])
    comment_compounds.append(score_df)

comment_df = pd.concat(comment_compounds, axis=1)
compound_df = pd.concat([df, comment_df], axis=1)

compound_df = compound_df.rename(columns={'0_compound':'headline_compound'})
cols = compound_df.columns.to_list()

col_new = cols[:25]
col_old = cols[25:]
cols_final = [j for i in zip(col_new, col_old) for j in i]
df = compound_df[cols_final]

# df.to_csv('comment_compound.csv', index=False, header=df.columns.values)

df.head(2)

# ----
# ### Mean Comment Compound
# ----

num_only = df.drop_duplicates().copy()
# df.drop_duplicates(inplace=True)

num_only = num_only.drop('headline_compound', axis=1)
num_only['mean'] = num_only.mean(axis=1)

df['mean'] = num_only['mean'].astype(float)
cols = df.columns.to_list()

df = df[[cols[0]] + [cols[-1]] + cols[1:-1]]
# df.to_csv('mean-compound.csv', index=False, header=df.columns.values)
df.head(2)

# ----
# ### Classifying the Sentiment
# ----

df['headline_label'] = 0
df['crowd_label'] = 0

# Headline Sentiment
df.loc[df['headline_compound'] > 0.15, 'headline_label'] = 1
df.loc[df['headline_compound'] < -0.15, 'headline_label'] = -1

# Sentiment of the Crowd
df.loc[df['mean'] > 0.15, 'crowd_label'] = 1
df.loc[df['mean'] < -0.15, 'crowd_label'] = -1

values = [df['headline_label'].value_counts(normalize=True), 
          df['crowd_label'].value_counts(normalize=True),
          df['headline_label'].value_counts(), 
          df['crowd_label'].value_counts()]

cols = ['headline_%', 'crowd_%', 'headline_count', 'crowd_count']

value_count_df = pd.concat(values, axis=1)
value_count_df.columns = cols
value_count_df

# ----
# ----
# # Exploratory Data Analysis
# ----
# ----

# ----
# ### Headlines CountVectorizor
# ----

from sklearn.feature_extraction.text import CountVectorizer

df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
df.fillna('N/A', inplace=True)

cv = CountVectorizer(input='content', binary=False, ngram_range=(1,1), strip_accents='ascii', stop_words='english')
cv_ft = cv.fit_transform(df['headline'])
df_cv = pd.DataFrame(cv_ft.toarray(), columns=cv.get_feature_names())

df_cv = df_cv[df_cv.columns.drop(list(df_cv.filter(regex='\d+')))]
df_cv = df_cv[df_cv.columns.drop(list(df_cv.filter(regex='_+')))]

df_cv.head()

# ----
# ----
# ## Frequency Distribution of words in headline based on sentiment
# ----
# ----

# dict(df_cv.sum(axis=0))

from nltk import ngrams, FreqDist, word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction import text 

stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
tt = TweetTokenizer() 

df['tokenized_headline'] = df['headline'].apply(lambda x: [x for x in tt.tokenize(x) if not x in stop_words if x.isalnum()])

df['tokenized_headline'] = df['tokenized_headline'].replace(r'\d+', '', regex=True)

df.head(2)

# ----
# ### Positive Headlines
# ----

fdist = FreqDist()

for value in df[df['headline_label'] == 1]['tokenized_headline']:
    for word in value:
        fdist[word] += 1

fdist_df = pd.DataFrame.from_dict(fdist, orient='index').rename(columns={0: 'frequency'}).rename_axis('term')

fdist_head_pos = fdist_df[~fdist_df.index.str.contains(r'[0-9]')]

fdist_head_poss = fdist_head_pos.sort_values(by='frequency', ascending=False).head(30)

fdist_head_poss.head()

# ----
# ### Negative Headlines
# ----

fdist = FreqDist()

for value in df[df['headline_label'] == -1]['tokenized_headline']:
    for word in value:
        fdist[word] += 1

fdist_df = pd.DataFrame.from_dict(fdist, orient='index').rename(columns={0: 'frequency'}).rename_axis('term')

fdist_head_neg = fdist_df[~fdist_df.index.str.contains(r'[0-9]')]

fdist_head_negs = fdist_head_neg.sort_values(by='frequency', ascending=False).head(30)

fdist_head_negs.head()

# ----
# ----
# ## Frequency Disribution of words in comments based on sentiment
# ----
# ----

comments = df.select_dtypes(include='object').copy()

comments.drop(['headline', 'tokenized_headline'], axis=1, inplace=True)

comments['all_comments'] = comments.apply(lambda x: ' '.join(x.astype(str)), axis=1)

comments.head(2)

# ----
# ### Comments CountVectorizor
# ----

vect = CountVectorizer(input='content', binary=False, ngram_range=(1,1), strip_accents='ascii', stop_words='english')
cv_ft2 = vect.fit_transform(comments['all_comments'])
df_cv2 = pd.DataFrame(cv_ft2.toarray(), columns=vect.get_feature_names())

com = df_cv2.copy()

com = com[com.columns.drop(list(com.filter(regex='\d+')))]
com = com[com.columns.drop(list(com.filter(regex='_+')))]

com.head()

stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
tt = TweetTokenizer() 

comments['tokenized_comments'] = comments['all_comments'].apply(lambda x: [x for x in tt.tokenize(x) if not x in stop_words if x.isalnum()])

comments['tokenized_comments'] = comments['tokenized_comments'].replace(r'\d+', '', regex=True)

df['all_comments'] = comments['all_comments']
df['tokenized_comments'] = comments['tokenized_comments']

df.head(2)

# df.to_pickle('final-df.pkl')
# df.to_csv('final-df.csv', index=False, header=df.columns.values)

# ----
# ### Positive Comments
# ----

fdist = FreqDist()

for value in df[df['crowd_label'] == 1]['tokenized_comments']:
    for word in value:
        fdist[word] += 1

fdist_df = pd.DataFrame.from_dict(fdist, orient='index').rename(columns={0: 'frequency'}).rename_axis('term')

fdist_com_pos = fdist_df[~fdist_df.index.str.contains(r'[0-9]')]

fdist_com_poss = fdist_com_pos.sort_values(by='frequency', ascending=False).head(30)

fdist_com_poss.head()

# ----
# ### Negative Comments
# ----

fdist = FreqDist()

for value in df[df['crowd_label'] == -1]['tokenized_comments']:
    for word in value:
        fdist[word] += 1

fdist_df = pd.DataFrame.from_dict(fdist, orient='index').rename(columns={0: 'frequency'}).rename_axis('term')

fdist_com_neg = fdist_df[~fdist_df.index.str.contains(r'[0-9]')]

fdist_com_negs = fdist_com_neg.sort_values(by='frequency', ascending=False).head(30)

fdist_com_negs.head()

# ----
# ----
# ## Combined Frequency Dataframe
# ----
# ----

fdist_dfs = [fdist_head_poss.reset_index(), fdist_head_negs.reset_index(), fdist_com_poss.reset_index(), fdist_com_negs.reset_index()]

tot_tf = pd.concat(fdist_dfs, axis=1)

pretty = tot_tf.T.copy()
iterables = [['head_pos', 'head_neg', 'com_pos', 'com_neg'], ['term', 'frequency']]

pretty.index = pd.MultiIndex.from_product(iterables)
pretty = pretty.T
pretty.head()

# pretty2 = pd.DataFrame(pretty.to_records())
# pretty2.to_csv('data/pretty.csv', index=False)

# ----
# ----
# ## Frequency of words found in most common words
# ----
# ----

freq_dict = {}

freq_dict['head_pos'] = list(zip(pretty['head_pos']['term'], pretty['head_pos']['frequency']))
freq_dict['head_neg'] = list(zip(pretty['head_neg']['term'], pretty['head_neg']['frequency']))
freq_dict['com_pos'] = list(zip(pretty['com_pos']['term'], pretty['com_pos']['frequency']))
freq_dict['com_neg'] = list(zip(pretty['com_neg']['term'], pretty['com_neg']['frequency']))

words = [term for key, value in freq_dict.items() for term, frequency in value]

Counter(words).most_common(5)

# ----
# ----
# ## WordCloud
# ----
# ----

from wordcloud import WordCloud

word_cloud = WordCloud(stopwords=stop_words, height=400, width=600, max_font_size=70, colormap='Dark2')

plt.rcParams['figure.dpi'] = 300

cols = ['head_pos', 'head_neg', 'com_pos', 'com_neg']

for idx, (col, tf) in enumerate(pretty.columns[::2]):
    word_cloud.generate(str(pretty.xs(col, axis=1)['term'].values))

    plt.subplot(2, 2, idx+1)
    plt.imshow(word_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(col)

plt.show()

# ----
# ----
# ## Overall Word Frequency of Comments
# ----
# ----

df2 = df.copy()
df2 = df2.set_index('headline')

com.index = df2.index
comm = com.T.copy()

# Top 30 words for the comments of each headline
top = {c: list(zip(comm[c].sort_values(ascending=False).head(30).index, 
                        comm[c].sort_values(ascending=False).head(30).values)) 
                        for c in comm.columns}

# List of all the words without their count
words = [t for c in comm.columns for t in [word for (word, count) in top[c]]]

most_common = Counter(words).most_common()

common_df = pd.DataFrame(most_common).set_index(0).rename(columns={1: 'frequency'}).rename_axis('word')

top30 = common_df.head(30)

plt.rcParams['figure.dpi'] = 300

sns.barplot(x=top30['frequency'], y=top30.index, palette='Dark2_r');

# ----
# ----
# ## Unique Words found in comments for each Headline
# ----
# ----

unique_count = [np.array(comm[c]).nonzero()[0].shape[0] for c in comm.columns]

unique_df = pd.DataFrame(list(zip(comm.columns.values, unique_count)), columns=['headline', 'unique_count'])

unique_df.sort_values(by='unique_count', ascending=False).head()

# ----
# ----
# ## Top Bigrams in Comments
# ----
# ----

vect = CountVectorizer(input='content', binary=False, ngram_range=(2,2), strip_accents='ascii', stop_words='english')
cv_ft2 = vect.fit_transform(comments['all_comments'])
df_cv2 = pd.DataFrame(cv_ft2.toarray(), columns=vect.get_feature_names())

com = df_cv2.copy()

com = com[com.columns.drop(list(com.filter(regex='\d+')))]
com = com[com.columns.drop(list(com.filter(regex='_+')))]

df2 = df.copy()
df2 = df2.set_index('headline')

com.index = df2.index
comm = com.T.copy()

# Top 30 words for the comments of each headline
top = {c: list(zip(comm[c].sort_values(ascending=False).head(30).index, 
                        comm[c].sort_values(ascending=False).head(30).values)) 
                        for c in comm.columns}

# List of all the words without their count
words = [t for c in comm.columns for t in [word for (word, count) in top[c]]]

most_common = Counter(words).most_common()

common_df = pd.DataFrame(most_common).set_index(0).rename(columns={1: 'frequency'}).rename_axis('word')

top20 = common_df.head(20)

plt.rcParams['figure.dpi'] = 300

sns.barplot(x=top20['frequency'], y=top20.index, palette='Paired_r')
plt.title('Comment Bigrams');

# ----
# ----
# ## Number of Words in Headline
# ----
# ----

headline = pd.DataFrame(df['tokenized_headline'].astype(str).str.replace(r"'", '', regex=True).replace(r'\,', '', regex=True).replace(r'\[', '', regex=True).replace(r'\]', '', regex=True))

headline['word_count'] = headline['tokenized_headline'].str.split().str.len()

sns.histplot(headline['word_count'], color='BlueViolet')
plt.title('Number of Words in Headline')
plt.ylabel('headline_count')
plt.xlabel('word_count');

# ----
# ----
# ## Average Length of Words in Headlines
# ----
# ----

headline['avg_word_length'] = headline['tokenized_headline'].str.split().apply(lambda x: sum(map(len, x))/len(x))

sns.histplot(headline['avg_word_length'], color='Gold')
plt.title('Average Length of Words in Headlines')
plt.ylabel('headline_count');

sns.boxplot(headline['word_count'], headline['avg_word_length'])
plt.title('Word Count vs Average Word Length in Headlines')
plt.xticks(rotation=-90);

# ----
# ----
# ## Average Comment Sentiment Score
# ----
# ----

sns.boxplot(data=df, x=df.loc[df['crowd_label'] != 0, 'crowd_label'], 
                        y=df.loc[df['crowd_label'] != 0, 'mean'], 
                        palette='Dark2_r', hue='headline_label')

plt.xlabel('crowd_sentiment_label')
plt.ylabel('mean_crowd_sentiment_score')
plt.title('Mean Sentiment Score of Positive and Negative Comments');

# ----
# ----
# ## Named Entity Recognition
# ----
# ----

import spacy

nlp = spacy.load('en_core_web_md')

headline = headline.rename(columns={'tokenized_headline': 'headline'})

headline['ner'] = headline['headline'].apply(lambda x: [X.label_ for X in nlp(x).ents])

counter = dict(Counter([y for x in headline['ner'] for y in x]).most_common())

sns.barplot(list(counter.values()), list(counter.keys()), palette='hls');

# ----
# #### Meaning of Spacy Named Entity Recognition Labels
# ----

pd.read_html('https://spacy.io/api/annotation')[6]

# ----
# ----
# ## Profanity and Average Crowd Sentiment
# ----
# ----

df['profanity'] = df['all_comments'].str.contains('(fuck|shit)').astype(int)

sns.violinplot(data=df, x='profanity', y='mean', palette='CMRmap')
plt.ylabel('mean_crowd_sentiment_score')
plt.title('Profanity vs Mean Crowd Sentiment');

# ----
# ----
# ## Average Number of Words in Comments
# ----
# ----

obj_df = df.select_dtypes('object').drop(['headline', 'tokenized_headline', 'all_comments', 'tokenized_comments'], axis=1)

stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
tt = TweetTokenizer() 

tilted = obj_df.T.copy()

length = []
temp = []

for y in range(tilted.shape[1]):
    for item in tilted.loc[:, y]:
        for x in tt.tokenize(str(item)):
            if not x in stop_words and x.isalnum():
                temp.append(len(x))
    length.append(sum(temp)/len(temp))
    temp.clear()

obj_df['avg_word_len'] = length

df['avg_word_len'] = obj_df['avg_word_len']

# ----
# ### Average Number of Words, Crowd Label, and Profanity
# ----

sns.boxplot(data=df, x='crowd_label', y='avg_word_len', hue='profanity', palette='CMRmap_r')

plt.title('Crowd Label & Profanity Based on Avg. Number of Words in Comments');

# df.to_csv('data/final-df.csv', index=False, header=df.columns.values)
# df.to_pickle('data/final-df.pkl')

