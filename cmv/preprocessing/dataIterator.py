from cmv.preprocessing.thread import Thread

class DataIterator:
    def __init__(self, data):
        self.data = data

    def iterPosts(self):
        for label in ('pos', 'neg'):
            for index,thread in enumerate(self.data[label]):
                #also get the original post and title if they exist
                originalPost = None
                title = None
                indices = '{}_indices'.format(label) 
                if indices in self.data:
                    if 'op' in self.data:
                        originalPost = self.data['op'][self.data[indices][index]]
                    
                        if 'titles' in self.data:
                            title = self.data['titles'][self.data[indices][index]]
                        
                    yield Thread(thread, originalPost, title),label=='pos'

    @property
    def labels(self):
        for thread,label in self.iterPosts():
            yield label

    @property
    def types(self):
        return tuple(self.data.keys())

    def responses(self, key=None, lower=False):
        for thread, label in self.iterPosts():
            yield thread.response.post(key, lower)

    def originalPosts(self, key, lower=False):
        for thread, label in self.iterPosts():
            yield thread.originalPost.post(key, lower)        
    
class PairedDataIterator(DataIterator):
    def iterPosts(self):
        '''
        iterate over posts in paired order, positives and negatives
        '''
        assert(len(self.data['pos']) == len(self.data['neg']))
        for i in range(len(self.data['pos'])):
            originalPost = self.data['op'][i]
            title = self.data['titles'][i]
            pos = self.data['pos'][i]
            neg = self.data['neg'][i]
                
            yield Thread(pos, originalPost, title), True
            yield Thread(neg, originalPost, title), False
