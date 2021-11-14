from reviews.reviews import ReviewScraper
from sentiment_analysis import LiveSentimentAnalysis
from sentiment_analysis import MusicalSentimentAnalysis


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


def handler(data, logger_data):
    log_object = logger(logger_data)
    # Initialize models
    # Validate urls provided in data
    if data['train']['mode'] == 1:
        log_object.log('Mode 1 chosen for Training', 'yellow')
        model = MusicalSentimentAnalysis.SentimentAnalysis_Musical(log_object)
    else:
        # Create dataset before initializing model
        model = LiveSentimentAnalysis.SentimentAnalysis_Live(log_object)

    # Run experiments
    # Create test set
    if data['test']['mode'] == 1:
        pass
    else:
        # Create dataset, and run it against the model
        pass
