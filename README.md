Natural Language Processing
===========================

### [Stock Sentiment Analysis](stock-sentiment)

News article headlines about stocks were scraped from [Tiingo](https://tiingo.com) and a sentiment analysis was performed using both `nltk` and `TextBlob`.

The graphs are made using `plotly`, which doesn't load on GitHub, so pasting the link to the Pasting the link of the [Jupyter Notebook](https://github.com/lmburns/nlp/blob/master/stock-sentiment/stock-mining.ipynb) into https://nbviewer.jupyterorg/ will allow one to view them.

------------------------
### [Twitter Sentiment Analysis](twitter-sentiment)

Twitter's API was used to gather tweets about the 'Great Reset' as it was trending on social media when this project was being worked on and I thought that it would be controversial enough to get a wide range of opinions on the subject.  A sentiment analysis was then done using `TextBlob`.

#### Controversial CSV Files
- [great-reset-2000.csv](twitter-sentiment/csv/great-reset-2000.csv) = 2000 tweets containing the phrase 'great reset'
- [vote-fraud-1000.csv](twitter-sentiment/csv/vote-fraud-1000.csv) = 1000 tweets containing the phrase 'voter fraud'

------------------------
### [Reddit News Headlines and the Post's Comments](reddit-headlines)
Reddit's API was used to gather post titles and comments from `r/worldnews`.  `NLTK` was used to perform a sentiment analysis of the title and comments of the post.  Finally, graphs were created analyzing the cleaned data.  The number of words, length of words, and frequency of profanity in the headlines and comments were also analyzed.

The [csv](reddit-headlines/data/final-df.csv) to look at contains the posts title, followed by its' polarity score, then each comment on the post, followed by its' polarity score.  It may be too large to view on GitHub, so one could download `vd` (`pip install visidata`) to preview the csv quickly from the command line.
