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
import itertools
from sklearn.metrics import accuracy_score

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

    def __init__(self, data: pd.DataFrame, logger):
        self.raw_reviews = data
        self.logger = logger
        self.logger.log("Columns present in the dataset: {}".format(
            self.raw_reviews.columns), "yellow")
        self.logger.log("Preprocessing dataset...")
        self.preprocessing_data()
        self.logger.log("Preprocessing complete...")
        self.logger.log("Shape of the dataset: {}".format(
            self.process_reviews.shape), "yellow")
        self.logger.log("Columns present in the dataset: {}".format(
            self.process_reviews.columns), "yellow")
        self.logger.log(
            "Commencing Data visualization tasks and image generation")
        self.data_visualization()
        self.logger.log("Data visualization complete...")
        self.logger.log("Commencing Model training experiments...")
        self.feature_extraction_experiment()
        self.logger.log("Model training experiments complete...")

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
        self.logger.log(
            ">Converting NA values in review column to 'Missing'", 'lightgreen')
        self.process_reviews['content'] = self.process_reviews['content'].fillna(
            'Missing')

        # Creating a new column called 'reviews', combining the reviewText and summary columns
        self.logger.log(
            ">Creating a new column called 'reviews', combining the content and title columns", 'lightgreen')
        self.process_reviews['reviews'] = self.process_reviews['content'] + \
            self.process_reviews['title']
        self.process_reviews = self.process_reviews.drop(
            ['content', 'title'], axis=1)

        # Figuring out the distribution of categories
        self.process_reviews['rating'] = pd.to_numeric(
            self.process_reviews['rating'])

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
        self.logger.log(
            ">Converting star values to ['Positive', 'Negative', 'Neutral'] Sentiments", 'lightgreen')
        self.process_reviews['sentiment'] = self.process_reviews.apply(
            sentiment_value, axis=1)

        # print(self.process_reviews.info())
        # print(self.process_reviews['sentiment'].value_counts())

        # print(self.process_reviews['date'][5])
        self.logger.log(
            ">Splitting date into year, date and adding them as seperate columns", 'lightgreen')
        new = self.process_reviews["date"].str.split(
            " ", n=2, expand=True)
        self.process_reviews["day"] = new[0]
        self.process_reviews["month"] = new[1]
        self.process_reviews["year"] = new[2]
        self.logger.log(
            ">Dropping irrelevant columns and renaming others", 'lightgreen')
        self.process_reviews = self.process_reviews.drop(
            ['date', 'url', 'author', 'variant', 'images', 'verified', 'product'], axis=1)

        self.logger.log(
            ">Cleaning review text by removing punctuation and stop words", 'lightgreen')

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
        self.logger.log('>Plotting Sentiment vs Helpful Rate', 'lightgreen')
        sns.violinplot(x=senti_help["sentiment"], y=senti_help["helpful_rate"])
        plt.title('Sentiment vs Helpfulness')
        plt.xlabel('Sentiment categories')
        plt.ylabel('helpful rate')
        plt.savefig('images/'+self.logger.key+'_sentiment_helpful_rate.png')
        plt.clf()
        self.logger.create_output(
            'sentiment_helpful_rate', 'images/'+self.logger.key+'_sentiment_helpful_rate.png', 'sentiment_helpful_rate.png')

        self.logger.log('>Plotting Year vs Number of Reviews', 'lightgreen')
        self.process_reviews.groupby(['year', 'sentiment'])[
            'sentiment'].count().unstack().plot(legend=True)
        plt.title('Year and Sentiment count')
        plt.xlabel('Year')
        plt.ylabel('Sentiment count')
        plt.savefig('images/'+self.logger.key+'_year_sentiment_count.png')
        plt.clf()
        self.logger.create_output('year_sentiment_count', 'images/'+self.logger.key +
                                  '_year_sentiment_count.png', 'year_sentiment_count.png')

        # Creating a dataframe
        day = pd.DataFrame(self.process_reviews.groupby(
            'day')['reviews'].count()).reset_index()
        day['day'] = day['day'].astype('int64')
        day.sort_values(by=['day'])

        # Plotting the graph
        self.logger.log('>Plotting Day vs Number of Reviews', 'lightgreen')
        sns.barplot(x="day", y="reviews", data=day)
        plt.title('Day vs Reviews count')
        plt.xlabel('Day')
        plt.ylabel('Reviews count')
        plt.savefig('images/'+self.logger.key+'_day_reviews_count.png')
        plt.clf()
        self.logger.create_output('day_reviews_count', 'images/' +
                                  self.logger.key+'_day_reviews_count.png', 'day_reviews_count.png')

        self.process_reviews['polarity'] = self.process_reviews['reviews'].map(
            lambda text: TextBlob(text).sentiment.polarity)
        self.process_reviews['review_len'] = self.process_reviews['reviews'].astype(
            str).apply(len)
        self.process_reviews['word_count'] = self.process_reviews['reviews'].apply(
            lambda x: len(str(x).split()))

        cf.go_offline()
        cf.set_config_file(offline=False, world_readable=True)
        self.logger.log('>Plotting Polarity Distribution', 'lightgreen')
        self.process_reviews['polarity'].plot(
            kind='hist',
            bins=50,
            title='Sentiment Polarity Distribution')
        plt.savefig('images/'+self.logger.key+'_polarity_distribution.png')
        plt.clf()
        self.logger.create_output('polarity_distribution', 'images/'+self.logger.key +
                                  '_polarity_distribution.png', 'polarity_distribution.png')

        self.logger.log('>Plotting Rating Distribution', 'lightgreen')
        self.process_reviews['overall'] = pd.to_numeric(
            self.process_reviews['overall'])
        self.process_reviews['overall'].plot(
            kind='hist',
            title='Review Rating Distribution')
        plt.savefig('images/'+self.logger.key+'_rating_distribution.png')
        plt.clf()
        self.logger.create_output('rating_distribution', 'images/'+self.logger.key +
                                  '_rating_distribution.png', 'rating_distribution.png')

        self.logger.log('>Plotting Review Length Distribution', 'lightgreen')
        self.process_reviews['review_len'].plot(
            kind='hist',
            bins=100,
            title='Review Text Length Distribution')
        plt.savefig('images/'+self.logger.key+'_review_len_distribution.png')
        plt.clf()
        self.logger.create_output('review_len_distribution', 'images/'+self.logger.key +
                                  '_review_len_distribution.png', 'review_len_distribution.png')

        self.logger.log('>Plotting Word Count Distribution', 'lightgreen')
        self.process_reviews['word_count'].plot(
            kind='hist',
            bins=100,
            title='Review Text Word Count Distribution')
        plt.savefig('images/'+self.logger.key +
                    '_review_word_count_distribution.png')
        plt.clf()
        self.logger.create_output('review_word_count_distribution', 'images/'+self.logger.key +
                                  '_review_word_count_distribution.png', 'review_word_count_distribution.png')

        # Filtering data
        print(self.process_reviews['sentiment'].value_counts())
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

            print(fd_sorted.info())
            print(fd_sorted.head())
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
            fig.write_image("images/"+self.logger.key+"_word_count_plots_" +
                            str(number_of_words)+".png")
            self.logger.create_output('word_count_plots_'+str(number_of_words), 'images/'+self.logger.key +
                                      '_word_count_plots_'+str(number_of_words)+'.png', 'word_count_plots_'+str(number_of_words)+'.png')
        self.logger.log('>Plotting Word Count Plots', 'lightgreen')
        word_count(1)
        word_count(2)
        word_count(3)

        # def plot_word_cloud(sentiment, text):
        #     wordcloud = WordCloud(
        #         width=3000,
        #         height=2000,
        #         background_color='black',
        #         stopwords=STOPWORDS).generate(str(text))
        #     fig = plt.figure(
        #         figsize=(40, 30),
        #         facecolor='k',
        #         edgecolor='k')
        #     plt.imshow(wordcloud, interpolation='bilinear')
        #     plt.axis('off')
        #     plt.tight_layout(pad=0)
        #     plt.savefig('images/'+self.logger.key +
        #                 '_wordcloud_'+sentiment+'.png')
        #     plt.clf()
        #     self.logger.create_output('wordcloud_'+sentiment, 'images/'+self.logger.key +
        #                               '_wordcloud_'+sentiment+'.png', 'wordcloud_'+sentiment+'.png')
        # self.logger.log('>Plotting Word Cloud', 'lightgreen')
        # plot_word_cloud('positive', review_pos['reviews'])
        # plot_word_cloud('neutral', review_neu['reviews'])
        # plot_word_cloud('negative', review_neg['reviews'])

    def feature_extraction_experiment(self):
        '''
        This function is used to perform feature extraction experiment
        This function is used to perform feature extraction experiment
        1) Encodes sentiment into numbers
        2) Stems the words of the reviews
        3) Removes stop words and punctuations
        4) Vectorizes words
        5) Performs SMOTE to balance the data
        6) Splits the data into training and testing sets
        7) Performs Grid Search to find the best parameters and algorithm
        8) Performs cross validation to find the best parameters and algorithm
        9) plots confusion matrix
        '''
        # calling the label encoder function
        label_encoder = preprocessing.LabelEncoder()

        # Encode labels in column 'sentiment'.
        self.process_reviews['sentiment'] = label_encoder.fit_transform(
            self.process_reviews['sentiment'])
        # Extracting 'reviews' for processing
        review_features = self.process_reviews.copy()
        review_features = review_features[['reviews']].reset_index(drop=True)
        # Performing stemming on the review dataframe
        self.logger.log('>Performing stemming on the reviews', 'lightgreen')
        ps = PorterStemmer()
        global stop_words
        # splitting and adding the stemmed words except stopwords
        corpus = []
        for i in range(0, len(review_features)):
            review = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])
            review = review.split()
            review = [ps.stem(word)
                      for word in review if not word in stop_words]
            review = ' '.join(review)
            corpus.append(review)
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(2, 2))
        # TF-IDF feature matrix
        self.logger.log('>Performing TF-IDF on the reviews', 'lightgreen')
        X = tfidf_vectorizer.fit_transform(review_features['reviews'])

        # Getting the target variable(encoded)
        y = self.process_reviews['sentiment']
        self.logger.log('>Performing SMOTE on the reviews', 'lightgreen')
        self.logger.log(f'>>Original dataset shape: {Counter(y)}', 'yellow')

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        self.logger.log(
            f'>>Resampled dataset shape: {Counter(y_res)}', 'yellow')

        # Divide the dataset into Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.25, random_state=0)

        print('Shape: ', X_train.shape, y_train.shape)

        def plot_confusion_matrix(cm,
                                  target_names,
                                  title='Confusion matrix',
                                  cmap=None,
                                  normalize=True):
            accuracy = np.trace(cm) / np.sum(cm).astype('float')
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
                accuracy, misclass))
            plt.savefig('images/'+self.logger.key + '_confusion_matrix.png')
            plt.clf()
            self.logger.create_output(
                'confusion_matrix', 'images/'+self.logger.key + '_confusion_matrix.png', 'confusion_matrix.png')

        # creating the objects
        logreg_cv = LogisticRegression(random_state=0)
        dt_cv = DecisionTreeClassifier()
        knn_cv = KNeighborsClassifier()
        svc_cv = SVC()
        nb_cv = BernoulliNB()
        self.logger.log('>Performing Grid Search on models', 'lightgreen')
        cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',
                   2: 'KNN', 3: 'SVC', 4: 'Naive Bayes'}
        cv_models = [logreg_cv, dt_cv, knn_cv, svc_cv, nb_cv]

        for i, model in enumerate(cv_models):
            self.logger.log(">>{} Test Accuracy: {}".format(cv_dict[i], cross_val_score(
                model, X, y, cv=10, scoring='accuracy').mean()), 'yellow')
        self.logger.log(
            '>Performing Grid Search on hyperparameters (Logistic Regression)', 'lightgreen')
        param_grid = {'C': np.logspace(-4, 4, 50),
                      'penalty': ['l1', 'l2']}
        clf = GridSearchCV(LogisticRegression(random_state=0),
                           param_grid, cv=5, verbose=0, n_jobs=-1)
        best_model = clf.fit(X_train, y_train)
        print(best_model.best_estimator_)
        self.logger.log(">>The mean accuracy of the model is: "+str(
                        best_model.score(X_test, y_test)), 'lightgreen')

        logreg = LogisticRegression(C=10000.0, random_state=0)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        self.logger.log('>Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            logreg.score(X_test, y_test)), 'lightgreen')

        cm = metrics.confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ['Negative', 'Neutral', 'Positive'])

    def create_model(self, X, y):
        '''
        This function creates a model using the training data
        '''
        self.mlmodel = LogisticRegression(C=10000.0, random_state=0)
        self.mlmodel.fit(X, y)

    def predict_review_sentiment(self, review: str) -> str:
        '''
        This function predicts the sentiment of the review
        '''
        ##########################################
        # Optimization is required for this code #
        ##########################################
        label_encoder = preprocessing.LabelEncoder()
        self.process_reviews['sentiment'] = label_encoder.fit_transform(
            self.process_reviews['sentiment'])
        # Extracting 'reviews' for processing
        review_features = self.process_reviews.copy()
        review_features = review_features[['reviews']].reset_index(drop=True)
        # Performing stemming on the review dataframe
        ps = PorterStemmer()
        global stop_words
        # splitting and adding the stemmed words except stopwords
        corpus = []
        for i in range(0, len(review_features)):
            reviewx = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])
            reviewx = reviewx.split()
            reviewx = [ps.stem(word)
                       for word in reviewx if not word in stop_words]
            reviewx = ' '.join(reviewx)
            corpus.append(reviewx)
        tfidf_vectorizer = TfidfVectorizer(
            max_features=200, ngram_range=(2, 2))
        # TF-IDF feature matrix
        all_reviews = list(review_features['reviews'])
        all_reviews.append(review)
        print(all_reviews[-1])
        All_X = tfidf_vectorizer.fit_transform(all_reviews)
        X = All_X[:-1]
        y = self.process_reviews['sentiment']
        smol_X = All_X[-1]
        try:
            self.mlmodel
        except AttributeError:
            self.create_model(X, y)

        dict_of_deconstruct = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return dict_of_deconstruct[self.mlmodel.predict(smol_X)[0]]

    def mass_predict_review_sentiment(self, list_of_reviews: list) -> list:
        # '''
        # This function predicts the sentiment of the list of reviews by calling the predict_review_sentiment function
        # '''
        # lst_of_predictions = []
        # for review in list_of_reviews:
        #     lst_of_predictions.append(self.predict_review_sentiment(review))
        # return lst_of_predictions
        self.logger.log(
            '>Preprocessing test set (stemming and removal of stopwords', 'lightgreen')
        label_encoder = preprocessing.LabelEncoder()
        self.process_reviews['sentiment'] = label_encoder.fit_transform(
            self.process_reviews['sentiment'])
        # Extracting 'reviews' for processing
        review_features = self.process_reviews.copy()
        review_features = review_features[['reviews']].reset_index(drop=True)
        # Performing stemming on the review dataframe
        ps = PorterStemmer()
        global stop_words
        # splitting and adding the stemmed words except stopwords
        corpus = []
        for i in range(0, len(review_features)):
            reviewx = re.sub('[^a-zA-Z]', ' ', review_features['reviews'][i])
            reviewx = reviewx.split()
            reviewx = [ps.stem(word)
                       for word in reviewx if not word in stop_words]
            reviewx = ' '.join(reviewx)
            corpus.append(reviewx)
        self.logger.log('>Creating TF-IDF feature matrix', 'lightgreen')
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2000, ngram_range=(2, 2))
        # TF-IDF feature matrix
        all_reviews = list(review_features['reviews'])
        # print(len(all_reviews))
        all_reviews.extend(list_of_reviews)
        # print(len(list_of_reviews))
        All_X = tfidf_vectorizer.fit_transform(all_reviews)
        X = All_X[:-(len(list_of_reviews))]
        y = self.process_reviews['sentiment']
        smol_X = All_X[-(len(list_of_reviews)):]
        # print(smol_X.shape)
        # print(All_X.shape)
        self.logger.log(
            '>Creating model with new TF-IDF feature matrix', 'lightgreen')
        self.create_model(X, y)

        dict_of_deconstruct = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        y_pred = [dict_of_deconstruct[i] for i in self.mlmodel.predict(smol_X)]
        self.logger.log('>Prediction complete', 'lightgreen')
        return y_pred

    def accuracy_score(self, y_true, y_pred):
        '''
        This function calculates the accuracy of the model
        '''
        return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    livedata = pd.read_csv('livedata.csv', index_col=0)
    new_obj = SentimentAnalysis_Live(livedata)
    livedata = pd.read_csv('Musical_instruments_reviews.csv')
    # print(new_obj.mass_predict_review_sentiment(livedata['reviewText']))
    print(new_obj.mass_predict_review_sentiment(
        ['i love this phone. i spend lots of time with it. its fast and it works']))
