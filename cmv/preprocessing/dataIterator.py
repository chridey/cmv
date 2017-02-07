from cmv.preprocessing.thread import Thread

class DataIterator:
    def __init__(self, data, subsets=('train', 'val')):
        self.data = data

        assert(set(subsets).issubset({'train', 'val'}))
        self.subsets = subsets
        self._keys_2d = None
        self._keys_3d = None
        
    def iterPosts(self):
        for subset in self.subsets:
            for label in ('pos', 'neg'):
                for index,thread in enumerate(self.data['{}_{}'.format(subset, label)]):
                    #also get the original post and title if they exist
                    originalPost = None
                    title = None
                    indices = '{}_{}_indices'.format(subset, label) 
                    if indices in self.data:
                        if subset+'_op' in self.data:
                            originalPost = self.data[subset+'_op'][self.data[indices][index]]
                    
                        if subset+'_titles' in self.data:
                            title = self.data[subset+'_titles'][self.data[indices][index]]
                        
                    yield Thread(thread, originalPost, title),label=='pos'

    @property
    def labels(self):
        for thread,label in self.iterPosts():
            yield label

    def _get_keys(self):
        self._keys_2d = []
        self._keys_3d = []
        g = self.iterPosts()
        thread = next(g)
        for key in thread.response.keys:
            if type(thread.response.metadata[key][0]) == str:
                self._keys_2d.append(key)
            elif type(thread.response.metadata[key][0]) == list:
                self._keys_3d.append(key)
                
    @property
    def keys_2d(self):
        if self._keys_2d is None:
            self._get_keys()
        return self._keys_2d
        
    @property
    def keys_3d(self):
        if self._keys_3d is None:
            self._get_keys()
        return self._keys_3d

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
        for subset in self.subsets:
            assert(len(self.data[subset+'_pos']) == len(self.data[subset+'_neg']))
            for i in range(len(self.data[subset+'_pos'])):
                originalPost = self.data[subset+'_op'][i]
                title = self.data[subset+'_titles'][i]
                pos = self.data[subset+'_pos'][i]
                neg = self.data[subset+'_neg'][i]
                
                yield Thread(pos, originalPost, title), True
                yield Thread(neg, originalPost, title), False
