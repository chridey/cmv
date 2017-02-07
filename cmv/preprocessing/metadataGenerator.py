import bz2
import json

from cmv.preprocessing.postPreprocessor import PostPreprocessor

class MetadataGenerator:
    def __init__(self, train_filename, val_filename, num_responses=2**32, extend=True,
                 discourse=True, frames=True, sentiment=False):
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.num_responses = num_responses
        self.extend = extend
        self.border = 'INTERMEDIATE_DISCUSSION'

        self.discourse = discourse
        self.frames = frames
        self.sentiment = sentiment
        
        self._data = None
                
    def _load_file(self, filename):
        pairs = []
        with bz2.BZ2File(filename) as f:
            for line in f:
                pairs.append(json.loads(line))
        return pairs

    @property
    def data(self):
        if self._data is not None:
            return self._data

        train = self._load_file(self.train_filename)
        val = self._load_file(self.val_filename)
        
        train_op, train_titles, train_pos, train_pos_indices, train_neg, train_neg_indices = self.processData(train)
        val_op, val_titles, val_pos, val_pos_indices, val_neg, val_neg_indices = self.processData(val)

        self._data = dict(train_op=train_op,
                          train_titles=train_titles,
                          train_pos=train_pos,
                          train_pos_indices=train_pos_indices,
                          train_neg=train_neg,
                          train_neg_indices=train_neg_indices,
                          val_op=val_op,
                          val_titles=val_titles,
                          val_pos=val_pos,
                          val_pos_indices=val_pos_indices,
                          val_neg=val_neg,
                          val_neg_indices=val_neg_indices)
        
        return self._data
    
    def processData(self, pairs):
        op = []
        titles = []
        pos = []
        pos_indices = []
        neg = []
        neg_indices = []
        
        for pair_index,pair in enumerate(pairs):
            op.append(PostPreprocessor(pair['op_text'], op=True,
                                       discourse=False, frames=False).processedData)

            post = ''
            for comment_index,comment in enumerate(pair['negative']['comments'][:self.num_responses]):
                if self.extend:
                    if comment_index > 0:
                        post += '\n' + self.border + '\n'
                    post += comment['body']
                else:
                    neg.append(PostPreprocessor(comment['body'],
                                                discourse=self.discourse, frames=self.frames,
                                                sentiment=self.sentiment).processedData)
                    neg_indices.append(pair_index)
                    
            if self.extend:
                neg.append(PostPreprocessor(comment['body'],
                                            discourse=self.discourse, frames=self.frames,
                                            sentiment=self.sentiment).processedData)
                neg_indices.append(pair_index)
                
            post = ''
            for comment_index,comment in enumerate(pair['positive']['comments'][:self.num_responses]):
                if self.extend:
                    if comment_index > 0:
                        post += '\n' + self.border + '\n'
                    post += comment['body']
                else:
                    pos.append(PostPreprocessor(comment['body'],
                                                discourse=self.discourse, frames=self.frames,
                                                sentiment=self.sentiment).processedData)
                    pos_indices.append(pair_index)

            if self.extend:
                pos.append(PostPreprocessor(comment['body'],
                                            discourse=self.discourse, frames=self.frames,
                                            sentiment=self.sentiment).processedData)
                pos_indices.append(pair_index)
                
            titles.append(PostPreprocessor(pair['op_title'], discourse=False, frames=False).processedData)
        
        return op, titles, pos, pos_indices, neg, neg_indices

        
