from imblearn.over_sampling import SMOTE
from collections import Counter
from collections import defaultdict
import cufflinks as cf
from numpy.core.numeric import NaN
from scipy import interp
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
import warnings

warnings.filterwarnings('ignore')

stop_words = ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
              'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
              'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
              'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
              'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
              'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from',
              'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
              'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
              'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
              'at', 'after', 'its', 'which', 'there', 'our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
              'over', 'again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']


class SentimentAnalysis_Live:

    def __init__(self, data: pd.DataFrame):
        self.raw_reviews = data
        self.preprocessing_data()
        self.data_visualization()

    def preprocessing_data(self):
        '''
        This function preprocesses the data and stores it in a dataframe
        Steps taken:
        1. Removing the rows with NaN values
        2. Combine the 'reviewText' and 'summary' column into a single column 'reviews'
        3. Converting stars to sentiment values ['Positive', 'Negative', 'Neutral']
        4. Splitting date into year, date and adding them as seperate columns
        5. Remove whitespaces
        6. Calculate values of helpfulness
        7. Remove stopwords, punctuation and stem the words
        '''
        # Creating a copy of the initial file read
        self.process_reviews = self.raw_reviews.copy()

        # Convert all NA values in review column to 'Missing'
        self.process_reviews['content'] = self.process_reviews['content'].fillna(
            'Missing')

        # Creating a new column called 'reviews', combining the reviewText and summary columns
        self.process_reviews['reviews'] = self.process_reviews['content'] + \
            self.process_reviews['title']
        self.process_reviews = self.process_reviews.drop(
            ['content', 'title'], axis=1)

        # Figuring out the distribution of categories

        def sentiment_value(row):
            '''This function returns sentiment value based on the overall ratings from the user'''
            if row['rating'] == 3.0:
                val = 'Neutral'
            elif row['rating'] == 1.0 or row['rating'] == 2.0:
                val = 'Negative'
            elif row['rating'] == 4.0 or row['rating'] == 5.0:
                val = 'Positive'
            else:
                val = -1
            return val

        # Applying the function in our new column
        self.process_reviews['sentiment'] = self.process_reviews.apply(
            sentiment_value, axis=1)

        # print(self.process_reviews.info())
        # print(self.process_reviews['sentiment'].value_counts())

        # print(self.process_reviews['date'][5])
        new = self.process_reviews["date"].str.split(
            " ", n=2, expand=True)
        self.process_reviews["day"] = new[0]
        self.process_reviews["month"] = new[1]
        self.process_reviews["year"] = new[2]
        self.process_reviews = self.process_reviews.drop(
            ['date', 'url', 'author', 'variant', 'images', 'verified', 'product'], axis=1)

        def review_cleaning(text):
            '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
            and remove words containing numbers.'''
            text = str(text).lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            return text

        # print(self.process_reviews['reviews'][0])
        self.process_reviews['reviews'] = self.process_reviews['reviews'].apply(
            lambda x: review_cleaning(x))

        global stop_words
        self.process_reviews['reviews'] = self.process_reviews['reviews'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        # print(self.process_reviews['reviews'][0])

        self.process_reviews['overall'] = self.process_reviews['rating']
        self.process_reviews = self.process_reviews.drop(['rating'], axis=1)
        # print(self.process_reviews.info())

    def data_visualization(self) -> None:
        '''
        This function is used to visualize the data
        Graphs generated:
        1. Sentiment vs helpful rate
        2. Year vs number of reviews, grouped by setiment
        3. Day vs review count
        4. Polarity distribution
        5. Rating distribution
        6. Review length distribution
        7. Review word count distribution
        8. Frequency of words grouped by sentiment (groups of 1,2,3)
        9. Word cloud of each sentiment
        '''
        ###########################################
        # Add ROC curve and AUC score to the plot #
        ###########################################
        # plot layout
        plt.rcParams.update({'font.size': 18})
        rcParams['figure.figsize'] = 16, 9

        # Creating dataframe and removing 0 helpfulrate records
        senti_help = pd.DataFrame(self.process_reviews, columns=[
            'sentiment', 'helpful_rate'])
        senti_help = senti_help[senti_help['helpful_rate'] != 0.00]

        # Plotting phase
        sns.violinplot(x=senti_help["sentiment"], y=senti_help["helpful_rate"])
        plt.title('Sentiment vs Helpfulness')
        plt.xlabel('Sentiment categories')
        plt.ylabel('helpful rate')
        plt.savefig('images/sentiment_helpful_rate.png')
        plt.clf()

        self.process_reviews.groupby(['year', 'sentiment'])[
            'sentiment'].count().unstack().plot(legend=True)
        plt.title('Year and Sentiment count')
        plt.xlabel('Year')
        plt.ylabel('Sentiment count')
        plt.savefig('images/year_sentiment_count.png')
        plt.clf()

        # Creating a dataframe
        day = pd.DataFrame(self.process_reviews.groupby(
            'day')['reviews'].count()).reset_index()
        day['day'] = day['day'].astype('int64')
        day.sort_values(by=['day'])

        # Plotting the graph
        sns.barplot(x="day", y="reviews", data=day)
        plt.title('Day vs Reviews count')
        plt.xlabel('Day')
        plt.ylabel('Reviews count')
        plt.savefig('images/day_reviews_count.png')
        plt.clf()

        self.process_reviews['polarity'] = self.process_reviews['reviews'].map(
            lambda text: TextBlob(text).sentiment.polarity)
        self.process_reviews['review_len'] = self.process_reviews['reviews'].astype(
            str).apply(len)
        self.process_reviews['word_count'] = self.process_reviews['reviews'].apply(
            lambda x: len(str(x).split()))

        cf.go_offline()
        cf.set_config_file(offline=False, world_readable=True)

        self.process_reviews['polarity'].plot(
            kind='hist',
            bins=50,
            title='Sentiment Polarity Distribution')
        plt.savefig('images/polarity_distribution.png')
        plt.clf()

        self.process_reviews['overall'].plot(
            kind='hist',
            title='Review Rating Distribution')
        plt.savefig('images/rating_distribution.png')
        plt.clf()

        self.process_reviews['review_len'].plot(
            kind='hist',
            bins=100,
            title='Review Text Length Distribution')
        plt.savefig('images/review_len_distribution.png')
        plt.clf()

        self.process_reviews['word_count'].plot(
            kind='hist',
            bins=100,
            title='Review Text Word Count Distribution')
        plt.savefig('images/review_word_count_distribution.png')
        plt.clf()

        # Filtering data
        review_pos = self.process_reviews[self.process_reviews["sentiment"]
                                          == 'Positive'].dropna()
        review_neu = self.process_reviews[self.process_reviews["sentiment"]
                                          == 'Neutral'].dropna()
        review_neg = self.process_reviews[self.process_reviews["sentiment"]
                                          == 'Negative'].dropna()

        ## custom function for ngram generation ##

        def generate_ngrams(text, n_gram=1):
            token = [token for token in text.lower().split(
                " ") if token != "" if token not in STOPWORDS]
            ngrams = zip(*[token[i:] for i in range(n_gram)])
            return [" ".join(ngram) for ngram in ngrams]

        ## custom function for horizontal bar chart ##

        def horizontal_bar_chart(df, color):
            trace = go.Bar(
                y=df["word"].values[::-1],
                x=df["wordcount"].values[::-1],
                showlegend=False,
                orientation='h',
                marker=dict(
                    color=color,
                ),
            )
            return trace

        def word_count(number_of_words):
            ## Get the bar chart from positive reviews ##
            freq_dict = defaultdict(int)
            for sent in review_pos["reviews"]:
                for word in generate_ngrams(sent, number_of_words):
                    freq_dict[word] += 1
            fd_sorted = pd.DataFrame(
                sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
            fd_sorted.columns = ["word", "wordcount"]
            trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

            ## Get the bar chart from neutral reviews ##
            freq_dict = defaultdict(int)
            for sent in review_neu["reviews"]:
                for word in generate_ngrams(sent, number_of_words):
                    freq_dict[word] += 1
            fd_sorted = pd.DataFrame(
                sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
            fd_sorted.columns = ["word", "wordcount"]
            trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

            ## Get the bar chart from negative reviews ##
            freq_dict = defaultdict(int)
            for sent in review_neg["reviews"]:
                for word in generate_ngrams(sent, number_of_words):
                    freq_dict[word] += 1
            fd_sorted = pd.DataFrame(
                sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
            fd_sorted.columns = ["word", "wordcount"]
            trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

            # Creating two subplots
            fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                                      subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                                      "Frequent words of negative reviews"])
            fig.append_trace(trace0, 1, 1)
            fig.append_trace(trace1, 2, 1)
            fig.append_trace(trace2, 3, 1)
            fig['layout'].update(height=1200, width=900,
                                 paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

            # plot(fig, filename='word-plots', image='png')
            fig.write_image("images/word_count_plots_" +
                            str(number_of_words)+".png")

        word_count(1)
        word_count(2)
        word_count(3)

        def plot_word_cloud(sentiment, text):
            wordcloud = WordCloud(
                width=3000,
                height=2000,
                background_color='black',
                stopwords=STOPWORDS).generate(str(text))
            fig = plt.figure(
                figsize=(40, 30),
                facecolor='k',
                edgecolor='k')
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig('images/wordcloud_'+sentiment+'.png')

        plot_word_cloud('positive', review_pos['reviews'])
        plot_word_cloud('neutral', review_neu['reviews'])
        plot_word_cloud('negative', review_neg['reviews'])
