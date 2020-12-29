## [Stock Sentiment Analysis](stock-sentiment)
I wanted to try and do something with Natural Language Processing, so I found a website that compiles news articles about stocks [tingo](https://www.tiingo.com) and used `nltk`'s `SentimentIntensityAnalyzer`, as well as `TextBlob`'s `.sentiment` tool.

- **NLTK Sentiment Analysis**: 
	- [How to - Sentiment](https://www.nltk.org/howto/sentiment.html)
	- [NLTK Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
- **TextBlob**: 
	- [TextBlob - Docs](https://textblob.readthedocs.io/en/dev/)
	- [TextBlob - Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html)

The graphs are made using `plotly`, which don't load on github.  So I would paste the link into https://nbviewer.jupyterorg/ to view the graphs.  

I would've also used `seaborn` but I was unable to figure out how to plot the charts side by side, or how to plot a dataframe without specifying an x or y.  I want `x` to = `date` or `dayofweek` and `y` to = the values, and then color (`hue`) the chart based on the `ticker`.

## [Twitter Sentiment Analysis](twitter-sentiment)
This is a sentiment analysis done on tweets containing the phrase 'great reset' using `TextBlob`.  I decided to do it on this topic because I saw that it was trending on Twitter a couple of days ago and thought that it would be controversial enough to get a wide variety of opinions.

#### CSV Files
- [great-reset-2000.csv](twitter-sentiment/csv/great-reset-2000.csv) = CSV of 2000 tweets containing the phrase 'great reset'
- [vote-fraud-1000.csv](twitter-sentiment/csv/vote-fraud-1000.csv) = CSV of 1000 tweets containing the phrase 'voter fraud'
- [practice.csv](twitter-sentiment/csv/practice.csv) = CSV of 100 of Donald Trump's tweets to practice with

## [Reddit News Headlines and the Post's Comments](reddit-headlines)
`NLTK` was used to perform sentiment analysis of the headline title, as well as performing sentiment analysis on the comments about the headlines.

I analyzed the number of words in the headlines and comments, along with the length of the words.  I also analyzed the profanity found in negative vs positive comments and a little bit more.
