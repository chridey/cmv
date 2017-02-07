import re

from spacy.en import English

from cmv.preprocessing.preprocess import normalize_from_body
from cmv.preprocessing.preprocess import Metadata
from cmv.preprocessing.discourseClassifier import DiscourseClassifier
from cmv.preprocessing.frameClassifier import FrameClassifier
from cmv.preprocessing.sentimentClassifier import SentimentClassifier

class PostPreprocessor:
    cmv_pattern = re.compile('cmv:?', re.IGNORECASE)
    nlp = English()
    
    def __init__(self, data, op=False, lower=False, discourse=True, frames=True, sentiment=False):
        self.data = data
        self.op = op
        self.lower = lower

        self.metadata = Metadata()
        
        self.discourseclassifier = None 
        if discourse:
            self.discourseClassifier = DiscourseClassifier()

        self.frameClassifier = None
        if frames:
            self.frameClassifier = FrameClassifier()

        self.sentimentClassifier = None
        if sentiment:
            self.sentimentClassifier = SentimentClassifier()
        
        self._processedData = None
        
    def cleanup(self, text):

        cleaned_text = normalize_from_body(text, op=self.op, lower=self.lower)
        cleaned_text = self.cmv_pattern.sub('', cleaned_text)
        cleaned_text = cleaned_text.replace('\t', ' ')
    
        parsed_text = [self.nlp(unicode(i)) for i in cleaned_text.split('\n')]
        return parsed_text
        
    def preprocess(self, text):
        parsed_text = self.cleanup(text)
        split_sentences = []
        for paragraph in parsed_text:
            for sent in paragraph.sents:
                split_sentences.append(sent)

        processed_post = self.metadata.addMetadata(split_sentences)
        if self.discourse:
            processed_post.update(self.discourseClassifier.addDiscourse(processed_post))
        if self.frames:
            processed_post.update(self.frameClassifier.addFrames(processed_post))
        if self.sentiment:
            processed_post.update(self.sentimentClassifier.addSentiment(processed_post))

        return processed_post

    @property
    def processedData(self):
        if self._processedData is None:
            self._processedData = self.preprocess(self.data)
        return self._processedData
    
    @property
    def length(self):
        return len(self.processedData['words'])
    
    def __iter__(self):
        for i in self.length:
            yield {key: self.processedData[key][i] for key in self.processedData}

    def iterMetadata(self):
        for key in self.processedData:
            for metadata in self.processedData[key]:
                yield key,metadata
                
    
    
