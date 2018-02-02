import bz2
import json

from cmv.preprocessing.postPreprocessor import PostPreprocessor

class MetadataGenerator(object):
    def __init__(self, train_filename, val_filename, test_filename=None,
                 num_responses=15, extend=True,
                 discourse=True, frames=True, num_examples=None):
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.test_filename = test_filename
        self.num_responses = num_responses
        self.extend = extend
        self.border = 'INTERMEDIATE_DISCUSSION'

        self.discourse = discourse
        self.frames = frames

        if num_examples is not None:
            self.num_examples = int(num_examples)
        else:
            self.num_examples = 2**32
            
        self._data = None
                
    def _load_file(self, filename):
        pairs = []
        with bz2.BZ2File(filename) as f:
            for index,line in enumerate(f):
                if index >= self.num_examples:
                    break
                pairs.append(json.loads(line))
        return pairs

    @property
    def data(self):
        if self._data is not None:
            return self._data

        train = self._load_file(self.train_filename)
        val = self._load_file(self.val_filename)
        
        train_metadata = self.processData(train)
        val_metadata = self.processData(val)

        self._data = dict(train=train_metadata,
                          val=val_metadata)

        if self.test_filename is not None:
            test = self._load_file(self.test_filename)
            test_metadata = self.processData(test)
            self._data.update(test=test_metadata)

        return self._data
    
    def processData(self, pairs):
        op = []
        titles = []
        pos = []
        pos_indices = []
        neg = []
        neg_indices = []
    
        for pair_index,pair in enumerate(pairs):
            if not(len(pair['op_text'])) or '[deleted]' in pair['op_text'] or '[removed]' in pair['op_text']:
                continue
            
            op.append(PostPreprocessor(pair['op_text'], op=True,
                                       discourse=self.discourse, frames=self.frames).processedData)

            post = ''
            for comment_index,comment in enumerate(pair['negative']['comments'][:self.num_responses]):
                if '[deleted]' in comment['body'] or '[removed]' in comment['body']:
                    continue
                if self.extend:
                    if comment_index > 0:
                        post += '\n' + self.border + '\n'
                    post += comment['body']
                else:
                    neg.append(PostPreprocessor(comment['body'],
                                                discourse=self.discourse, frames=self.frames).processedData)
                    neg_indices.append(pair_index)
                    
            if self.extend:
                neg.append(PostPreprocessor(post,
                                            discourse=self.discourse, frames=self.frames).processedData)
                neg_indices.append(pair_index)
                
            post = ''
            for comment_index,comment in enumerate(pair['positive']['comments'][:self.num_responses]):
                if '[deleted]' in comment['body'] or '[removed]' in comment['body']:
                    continue
                
                if self.extend:
                    if comment_index > 0:
                        post += '\n' + self.border + '\n'
                    post += comment['body']
                else:
                    pos.append(PostPreprocessor(comment['body'],
                                                discourse=self.discourse, frames=self.frames).processedData)
                    pos_indices.append(pair_index)

            if self.extend:
                pos.append(PostPreprocessor(post,
                                            discourse=self.discourse, frames=self.frames).processedData)
                pos_indices.append(pair_index)
                
            titles.append(PostPreprocessor(pair['op_title'], discourse=self.discourse, frames=self.frames).processedData)
        
        return dict(op=op, titles=titles, pos=pos,
                    pos_indices=pos_indices, neg=neg,
                    neg_indices=neg_indices)

        
