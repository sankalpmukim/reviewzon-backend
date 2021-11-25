from typing import final
from reviews.reviews import ReviewScraper
from sentiment_analysis import LiveSentimentAnalysis
from sentiment_analysis import LocalSentimentAnalysis
import pandas as pd
import json
import time
from datetime import datetime
import os


class logger:
    activator = True
    output_activator = True

    def __init__(self, config: list):
        self.counter = config[0]
        self.db = config[1].database()
        self.storage = config[1].storage()
        self.key = str(config[2])
        self.colors = {'lightgreen': '#14FCB9', 'green': '#57FC14',
                       'red': '#FF0000', 'yellow': '#FFFF00'}
        self.strings = json.load(open('strings.json'))
        self.output_counter = 0

    def log(self, message, color='green', end=False, error=False):
        if self.activator:
            self.db.child("livedata").child(self.key).update({self.counter: {'message': message,
                                                                             'color': self.colors[color], 'error': error, 'end': end}})
            self.counter += 1

    def close(self):
        if self.activator:
            self.db.child("livedata").child(self.key).remove()

    def create_output(self, prompt, file_path=None, file_name=None):
        if self.activator and self.output_activator:
            try:
                cloud_file_name = 'files/'+self.key+"/"+file_name
            except TypeError:
                pass
            url = "None"
            data = self.strings[prompt]['text']
            title = self.strings[prompt]['title']

            if file_path is not None:
                self.storage.child(cloud_file_name).put(file_path)
                url = self.storage.child(cloud_file_name).get_url(None)
            self.db.child("output").child(self.key).update(
                {title: {'data': data, 'url': url, 'counter': self.output_counter}})
            self.output_counter += 1

    def create_static(self, data):
        if self.activator and self.output_activator:
            self.db.child("output").child(
                self.key).child('static').update(data)


def organizer(data, logger_data, doExperiment=False):
    time_1 = datetime.now()
    log_object = logger(logger_data)
    log_object.create_static(data)
    if data['train']['mode'] == 1:
        log_object.log('Mode 1 chosen for Training', 'yellow')
        model = LocalSentimentAnalysis.SentimentAnalysis_Local(
            logger=log_object, doExperiment=doExperiment)
    else:
        log_object.log('Mode 2 chosen for Training', 'yellow')
        time.sleep(1)
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
            log_object.close()
            return
        model = LiveSentimentAnalysis.SentimentAnalysis_Live(
            final_dataset, log_object)

    # Create test set
    if data['test']['mode'] == 1:
        log_object.log('Mode 1 chosen for Testing', 'yellow')
        log_object.log('Creating test set', 'green')
        test_set = pd.read_csv('Musical_instruments_reviews.csv')
        test_set = test_set[:250]
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
        ret = model.accuracy_score(y_actual, y_pred)
        log_object.log('>Accuracy of model on test set: ' +
                       str(ret[0]), 'yellow')
        log_object.log('>Precision of model on test set: ' +
                       str(ret[1]), 'yellow')
        log_object.log('>Recall of model on test set: ' +
                       str(ret[2]), 'yellow')
        log_object.log('>F1 of model on test set: ' +
                       str(ret[3]), 'yellow')

    else:
        # Create dataset, and run it against the model
        log_object.log('Mode 2 chosen for Testing', 'yellow')
        scraper = ReviewScraper(log_object)
        for i in range(len(data['test']['urls'])):
            log_object.log('Downloading reviews from url: ' +
                           data['test']['urls'][i], 'green')
            if scraper.valid_state:
                scraper.get_reviews(data['test']['urls'][i])
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
        if len(final_dataset) == 0:
            log_object.log(
                'Dataset too small. Please try again with more links', 'red')
            log_object.log('Exiting', 'red', end=True, error=True)
            log_object.close()

        log_object.log('Preprocessing test data..')
        final_dataset = final_dataset.dropna()
        final_dataset['content'] = final_dataset['content'].fillna(
            'Missing')
        final_dataset['reviews'] = final_dataset['content'] + \
            final_dataset['title']
        final_dataset = final_dataset.drop(['content', 'title'], axis=1)
        final_dataset['rating'] = pd.to_numeric(
            final_dataset['rating'])

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
        final_dataset['sentiment'] = final_dataset.apply(
            sentiment_value, axis=1)

        y_actual = list(final_dataset['sentiment'])
        test_set = final_dataset['reviews']
        log_object.log('Test set created', 'green')
        log_object.log('Running test set on model..', 'green')
        y_pred = model.mass_predict_review_sentiment(test_set)
        ret = model.accuracy_score(y_actual, y_pred)
        log_object.log('>Accuracy of model on test set: ' +
                       str(ret[0]), 'yellow')
        log_object.log('>Precision of model on test set: ' +
                       str(ret[1]), 'yellow')
        log_object.log('>Recall of model on test set: ' +
                       str(ret[2]), 'yellow')
        log_object.log('>F1 of model on test set: ' +
                       str(ret[3]), 'yellow')

    log_object.log('All operations completed successfully. Exiting program..',
                   'green', end=True)
    td = datetime.now() - time_1
    # function to convert seconds into hours, minutes and seconds

    def convert_seconds(seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        hours, minutes, seconds = str(hours), str(minutes), str(seconds)
        if len(hours) < 2:
            hours = '0' + hours
        if len(minutes) < 2:
            minutes = '0' + minutes
        if len(seconds) < 2:
            seconds = '0' + seconds
        return hours, minutes, seconds
    td = convert_seconds(td.seconds)
    log_object.log('Time taken: ' + td[0] +
                   ':' + td[1] + ':' + td[2], 'yellow')
    log_object.close()

    for filename in os.listdir('images/'):
        if log_object.key in filename:
            os.remove('images/' + filename)
