import logging
from imblearn.over_sampling import SMOTE
from collections import Counter
from collections import defaultdict
import cufflinks as cf
from itertools import cycle
from numpy.core.numeric import NaN
from scipy import interp
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import nltk
import re
import string
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import plot
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalysis:

    def __init__(self, file_path: str):
        # Initalizing the class with the file
        self.raw_reviews = pd.read_csv(file_path)
        self.preprocessing_data()
        print(self.process_reviews.info())
        self.data_visualization()

    def preprocessing_data(self) -> None:
        # Preprocessing the data

        # Creating a copy of the initial file read
        self.process_reviews = self.raw_reviews.copy()

        # Convert all NA values in review column to 'Missing'
        self.process_reviews['reviewText'] = self.process_reviews['reviewText'].fillna(
            'Missing')

        # Creating a new column called 'reviews', combining the reviewText and summary columns
        self.process_reviews['reviews'] = self.process_reviews['reviewText'] + \
            self.process_reviews['summary']
        self.process_reviews = self.process_reviews.drop(
            ['reviewText', 'summary'], axis=1)

        # Figuring out the distribution of categories

        def sentiment_value(row):
            '''This function returns sentiment value based on the overall ratings from the user'''
            if row['overall'] == 3.0:
                val = 'Neutral'
            elif row['overall'] == 1.0 or row['overall'] == 2.0:
                val = 'Negative'
            elif row['overall'] == 4.0 or row['overall'] == 5.0:
                val = 'Positive'
            else:
                val = -1
            return val

        # Applying the function in our new column
        self.process_reviews['sentiment'] = self.process_reviews.apply(
            sentiment_value, axis=1)

        # new data frame which has date and year
        new = self.process_reviews["reviewTime"].str.split(
            ",", n=1, expand=True)

        # making separate date column from new data frame
        self.process_reviews["date"] = new[0]

        # making separate year column from new data frame
        self.process_reviews["year"] = new[1]

        self.process_reviews = self.process_reviews.drop(
            ['reviewTime'], axis=1)

        # Splitting the date
        new1 = self.process_reviews["date"].str.split(" ", n=1, expand=True)

        # adding month to the main dataset
        self.process_reviews["month"] = new1[0]

        # adding day to the main dataset
        self.process_reviews["day"] = new1[1]

        self.process_reviews = self.process_reviews.drop(['date'], axis=1)

        # Splitting the dataset based on comma and square bracket
        new1 = self.process_reviews["helpful"].str.split(",", n=1, expand=True)
        new2 = new1[0].str.split("[", n=1, expand=True)
        new3 = new1[1].str.split("]", n=1, expand=True)

        # Resetting the index
        new2.reset_index(drop=True, inplace=True)
        new3.reset_index(drop=True, inplace=True)

        # Dropping empty columns due to splitting
        new2 = new2.drop([0], axis=1)
        new3 = new3.drop([1], axis=1)

        # Concatenating the splitted columns
        helpful = pd.concat([new2, new3], axis=1)

        # I found few spaces in new3, so it is better to strip all the values to find the rate

        def trim_all_columns(df):
            """
            Trim whitespace from ends of each value across all series in dataframe
            """
            def trim_strings(x): return x.strip() if isinstance(x, str) else x
            return df.applymap(trim_strings)

        # Applying the function
        helpful = trim_all_columns(helpful)

        # Converting into integer types
        helpful[0] = helpful[0].astype(str).astype(int)
        helpful[1] = helpful[1].astype(str).astype(int)

        # Dividing the two columns, we have 0 in the second columns when dvided gives error, so I'm ignoring those errors
        try:
            helpful['result'] = helpful[1]/helpful[0]
        except ZeroDivisionError:
            helpful['result'] = 0

        # Filling the NaN values(created due to dividing) with 0
        helpful['result'] = helpful['result'].fillna(0)

        # Rounding of the results to two decimal places
        helpful['result'] = helpful['result'].round(2)

        # Attaching the results to a new column of the main dataframe
        self.process_reviews['helpful_rate'] = helpful['result']

        # dropping the helpful column from main dataframe
        self.process_reviews = self.process_reviews.drop(['helpful'], axis=1)

        # Removing unnecessary columns
        self.process_reviews = self.process_reviews.drop(
            ['reviewerName', 'unixReviewTime'], axis=1)

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

        self.process_reviews['reviews'] = self.process_reviews['reviews'].apply(
            lambda x: review_cleaning(x))

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

        self.process_reviews['reviews'] = self.process_reviews['reviews'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    def data_visualization(self) -> None:

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

        ## Get the bar chart from positive reviews ##
        freq_dict = defaultdict(int)
        for sent in review_pos["reviews"]:
            for word in generate_ngrams(sent):
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(
            sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

        ## Get the bar chart from neutral reviews ##
        freq_dict = defaultdict(int)
        for sent in review_neu["reviews"]:
            for word in generate_ngrams(sent):
                freq_dict[word] += 1
        fd_sorted = pd.DataFrame(
            sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
        fd_sorted.columns = ["word", "wordcount"]
        trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

        ## Get the bar chart from negative reviews ##
        freq_dict = defaultdict(int)
        for sent in review_neg["reviews"]:
            for word in generate_ngrams(sent):
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
        print("hello")
        fig['layout'].update(height=1200, width=900,
                             paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
        print(type(fig))

        # plot(fig, filename='word-plots', image='png')
        fig.write_image("images/word_count_plots.png")
        ###################################################
        # Need to create plot with more words (2, 3, etc) #
        ###################################################

    # def feature_extraction(process_reviews):
    #     # calling the label encoder function
    #     label_encoder = preprocessing.LabelEncoder()

    #     # Encode labels in column 'sentiment'.
    #     process_reviews['sentiment'] = label_encoder.fit_transform(
    #         process_reviews['sentiment'])
    #     # Extracting 'reviews' for processing
    #     review_features = process_reviews.copy()
    #     review_features = review_features[['reviews']].reset_index(drop=True)

    #     # Performing stemming on the review dataframe
    #     ps = PorterStemmer()
    #     stop_words = ['yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
    #                   'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
    #                   'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
    #                   'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
    #                   'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
    #                   'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from',
    #                   'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
    #                   'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
    #                   'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
    #                   'at', 'after', 'its', 'which', 'there', 'our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
    #                   'over', 'again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all']
    #     # splitting and adding the stemmed words except stopwords
    #     corpus = []
    #     for i in range(0, len(review_features)):
    #         review = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])
    #         review = review.split()
    #         review = [ps.stem(word)
    #                   for word in review if not word in stop_words]
    #         review = ' '.join(review)
    #         corpus.append(review)

    #     tfidf_vectorizer = TfidfVectorizer(
    #         max_features=5000, ngram_range=(2, 2))
    #     # TF-IDF feature matrix
    #     X = tfidf_vectorizer.fit_transform(review_features['reviews'])

    #     # Getting the target variable(encoded)
    #     y = process_reviews['sentiment']

    #     print(f'Original dataset shape : {Counter(y)}')

    #     smote = SMOTE(random_state=42)
    #     X_res, y_res = smote.fit_resample(X, y)

    #     print(f'Resampled dataset shape {Counter(y_res)}')

    #     # Divide the dataset into Train and Test
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_res, y_res, test_size=0.25, random_state=0)

    #     def plot_confusion_matrix(cm, classes,
    #                               normalize=False,
    #                               title='Confusion matrix',
    #                               cmap=plt.cm.Blues):
    #         """
    #         This function prints and plots the confusion matrix.
    #         Normalization can be applied by setting `normalize=True`.
    #         """

    #         plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #         plt.title(title)
    #         plt.colorbar()
    #         tick_marks = np.arange(len(classes))
    #         plt.xticks(tick_marks, classes, rotation=45)
    #         plt.yticks(tick_marks, classes)

    #         if normalize:
    #             cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #             print("Normalized confusion matrix")
    #         else:
    #             print('Confusion matrix, without normalization')

    #         thresh = cm.max() / 2.
    #         for i in range(cm.shape[0]):
    #             for j in range(cm.shape[1]):
    #                 plt.text(j, i, cm[i, j],
    #                          horizontalalignment="center",
    #                          color="white" if cm[i, j] > thresh else "black")

    #         plt.tight_layout()
    #         plt.ylabel('True label')
    #         plt.xlabel('Predicted label')
    #         plt.show()

    #     # creating the objects
    #     logreg_cv = LogisticRegression(random_state=0)
    #     dt_cv = DecisionTreeClassifier()
    #     knn_cv = KNeighborsClassifier()
    #     svc_cv = SVC()
    #     nb_cv = BernoulliNB()
    #     cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',
    #                2: 'KNN', 3: 'SVC', 4: 'Naive Bayes'}
    #     cv_models = [logreg_cv, dt_cv, knn_cv, svc_cv, nb_cv]

    #     for i, model in enumerate(cv_models):
    #         print("{} Test Accuracy: {}".format(cv_dict[i], cross_val_score(
    #             model, X, y, cv=10, scoring='accuracy').mean()))

    #     param_grid = {'C': np.logspace(-4, 4, 50),
    #                   'penalty': ['l1', 'l2']}
    #     clf = GridSearchCV(LogisticRegression(random_state=0),
    #                        param_grid, cv=5, verbose=0, n_jobs=-1)
    #     best_model = clf.fit(X_train, y_train)
    #     print(best_model.best_estimator_)
    #     print("The mean accuracy of the model is:",
    #           best_model.score(X_test, y_test))

    #     logreg = LogisticRegression(C=10000.0, random_state=0)
    #     logreg.fit(X_train, y_train)
    #     y_pred = logreg.predict(X_test)
    #     print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    #         logreg.score(X_test, y_test)))

    #     logreg = LogisticRegression(C=10000.0, random_state=0)
    #     logreg.fit(X_train, y_train)
    #     y_pred = logreg.predict(X_test)
    #     print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    #         logreg.score(X_test, y_test)))

    #     cm = metrics.confusion_matrix(y_test, y_pred)
    #     plot_confusion_matrix(cm, classes=['Negative', 'Neutral', 'Positive'])


# cleaned_data = preprocessing_data()
# data_visualization(cleaned_data)
# feature_extraction(cleaned_data)
new_obj = SentimentAnalysis('Musical_instruments_reviews.csv')
