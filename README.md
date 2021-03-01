Natural Language Processing
===========================

## [Stock Sentiment Analysis](stock-sentiment)
------------------------
Analysis of news article headlines for various stocks from [tingo](https://www.tiingo.com). `nltk`'s `SentimentIntensityAnalyzer` and `TextBlob`'s `.sentiment` tool were used to perform a sentiment analysis.

- **NLTK Sentiment Analysis**:
	- [How to - Sentiment](https://www.nltk.org/howto/sentiment.html)
	- [NLTK Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
- **TextBlob**:
	- [TextBlob - Docs](https://textblob.readthedocs.io/en/dev/)
	- [TextBlob - Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html)

The graphs are made using `plotly`, which don't load on github. Pasting the link into https://nbviewer.jupyterorg/ will allow one to view the graphs.

## [Twitter Sentiment Analysis](twitter-sentiment)

A sentiment analysis done on tweets containing the phrase 'great reset' using `TextBlob`.  The topic was chosen because it was trending on Twitter a couple of days ago and I thought that it would be controversial enough to get a wide variety of opinions.

#### CSV Files
- [great-reset-2000.csv](twitter-sentiment/csv/great-reset-2000.csv) = CSV of 2000 tweets containing the phrase 'great reset'
- [vote-fraud-1000.csv](twitter-sentiment/csv/vote-fraud-1000.csv) = CSV of 1000 tweets containing the phrase 'voter fraud'
- [practice.csv](twitter-sentiment/csv/practice.csv) = CSV of 100 of Donald Trump's tweets to practice with

## [Reddit News Headlines and the Post's Comments](reddit-headlines)

`NLTK` was used to perform sentiment analysis of the headline title, as well as performing sentiment analysis on the comments about the headlines.

The number of words, length of the words, and frequency of profanity in the headlines and comments were also analyzed.
