import json
import re

from pycorenlp import StanfordCoreNLP

class SentimentPredictor:
    def __init__(self, host='localhost', port=9000):
        self.nlp = StanfordCoreNLP('http://{}:{}'.format(host, port))

    def predict(self, words):
        words = ' '.join(words)
        words = re.sub(r'[^\x00-\x7f]',r' ',words)

        output = nlp.annotate(str(words),
                              properties={'annotators': 'sentiment',
                                          'outputFormat': 'json',
                                          'tokenize.whitespace': True
                                          })

        if type(output) == dict and len(output['sentences']) > 0:
            return output['sentences'][0]['sentiment']
        else:
            return  'Neutral'
            
    
