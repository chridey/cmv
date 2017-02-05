from cmv.preprocessing.thread import Thread

class DataIterator:
    def __init__(self, data, subsets=('train', 'val')):
        self.data = data

        assert(set(subsets).issubset({'train', 'val'}))
        self.subsets = subsets
        
    def iterPosts(self):
        for subset in self.subsets:
            for label in ('pos', 'neg'):
                for thread in self.data['{}_{}'.format(subset, label)]:
                    #TODO: also get the original post and title if they exist
                    
                    yield Thread(thread),label=='pos'
    
    def iterWords(self):
        pass

class PairedDataIterator(DataIterator):
    def iterPosts(self):
        '''
        iterate over posts in paired order, positives and negatives
        '''
        #TODO
        pass
