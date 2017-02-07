from cmv.preprocessing.sentimentPredictor import SentimentPredictor

class SentimentClassifier:
    def __init__(self, sentiment_predictor=None, verbose=False):
        self.sentiment_predictor = sentiment_predictor
        if sentiment_predictor is None:
            sentiment_predictor = SentimentPredictor()
       
    def addSentiment(self, preprocessed_post):
        metadata = dict(sentiment=[])
        for words in preprocessed_post['words']:
            metadata['sentiment'].append(self.sentiment_predictor.predict(words))
        return metadata

        
