from reviews.reviews import ReviewScraper
from sentiment_analysis import LiveSentimentAnalysis
from sentiment_analysis import MusicalSentimentAnalysis
import pandas as pd


class logger:
    def __init__(self, config: list):
        self.counter = config[0]
        self.db = config[1]
        self.key = config[2]
        self.colors = {'lightgreen': '#14FCB9', 'green': '#57FC14',
                       'red': '#FF0000', 'yellow': '#FFFF00'}

    def log(self, message, color='green', end=False, error=False):
        self.db.child("livedata").child(self.key).update({self.counter: {'message': message,
                                                                         'color': self.colors[color], 'error': error, 'end': end}})
        self.counter += 1

    def close(self):
        self.db.child("livedata").child(self.key).remove()


def handler(data, logger_data):
    log_object = logger(logger_data)
    if data['train']['mode'] == 1:
        log_object.log('Mode 1 chosen for Training', 'yellow')
        model = MusicalSentimentAnalysis.SentimentAnalysis_Musical(log_object)
    else:
        log_object.log('Mode 2 chosen for Training', 'yellow')
        scraper = ReviewScraper(log_object)
        for i in range(len(data['train']['urls'])):
            log_object.log('Downloading reviews from url: ' +
                           data['train']['urls'][i], 'green')
            if scraper.valid_state:
                scraper.get_reviews(data['train']['urls'][i])
            else:
                # log_object.log('Invalid URL', 'red')
                break
        final_dataset = scraper.retrieve_data()
        scraper.clear_data()
        try:
            final_dataset = pd.DataFrame(final_dataset)
        except:
            pass
        log_object.log('Data downloaded', 'green')
        try:
            log_object.log('Shape of dataset: ' +
                           str(final_dataset.shape), 'yellow')
        except:
            pass
        if len(final_dataset) < 1000:
            log_object.log(
                'Dataset too small. Please try again with more links', 'red')
            log_object.log('Exiting', 'red', end=True, error=True)
            return
        model = LiveSentimentAnalysis.SentimentAnalysis_Live(
            final_dataset, log_object)

    # Create test set
    if data['test']['mode'] == 1:
        log_object.log('Mode 1 chosen for Testing', 'yellow')
        log_object.log('Creating test set', 'green')
        test_set = pd.read_csv('Musical_instruments_reviews.csv')
        test_set = test_set.dropna()

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
        test_set['sentiment'] = test_set.apply(sentiment_value, axis=1)
        y_actual = list(test_set['sentiment'])
        test_set = test_set[['reviewText', 'summary']]
        test_set['reviews'] = test_set['reviewText'] + \
            ' ' + test_set['summary']
        test_set = test_set.drop(['reviewText', 'summary'], axis=1)
        test_set = list(test_set['reviews'])
        log_object.log('Test set created', 'green')
        log_object.log('Shape of test set: (' +
                       str(len(test_set))+",1)", 'yellow')
        log_object.log('Running test set on model..', 'green')
        y_pred = model.mass_predict_review_sentiment(test_set)
        accuracy = model.accuracy_score(y_actual, y_pred)
        log_object.log('>Accuracy of model on test set: ' +
                       str(accuracy), 'yellow')
    else:
        # Create dataset, and run it against the model
        pass

    log_object.log('All operations completed successfully. Exiting program..',
                   'green', end=True)
    log_object.close()
