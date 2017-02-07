import collections

import gensim
import numpy as np

from cmv.preprocessing import utils

class IndexGenerator:
    def __init__(self, iteratorType, metadata, embeddings=None, min_count=0, lower=False,
                 max_sentence_length=256,
                 max_post_length=40,
                 dimension=0,
                 indices=None):
        
        self.train = iteratorType(metadata, 'train')
        self.val = iteratorType(metadata, 'val')
         
        self.embeddings = embeddings
        self.min_count = min_count
        self.lower = lower
        self.max_sentence_length
        self.max_post_length
        self.dimension = dimension
        
        self._indices = indices
        self._data = None
        self._model = None
        
    @property
    def data(self):
        if self._data is None:
            self.processIndices()
        return self._data
    
    @property
    def indices(self):
        if self._indices is None:
            self.processIndices()
        return self._indices

    def _map_embeddings(self):
        embeddings_array = [None] * len(self._indices)
        if self._model is None:
            self._model = {}
            if self.embeddings is not None:
                self._model = gensim.models.Doc2Vec.load_word2vec_format(self.embeddings, binary=False)

        for word in self._indices:
            if word in self._model:
                embeddings_array[self._indices[word]] = self._model[word]
            else:
                embeddings_array[self._indices[word]] = np.random.uniform(-1, 1, (self.dimension,))
                
        return np.array(embeddings_array)
    
    def processIndices(self):
        self._data = {}
        if self._indices is None:
            self._indices = collections.defaultdict(dict)
        
        if self.min_count:
            self._indices['words'] = utils.preprocess_indices_min_count(self.train.responses('words'),
                                                                        self.lower,
                                                                        self.min_count)

        for subset in ('train', 'val'):
            data = self.train
            if subset == 'val':
                data = self.val
                
            for which in data.types:
                f = data.responses
                if which == 'op':
                    f = data.originalPosts

                for key in data.keys_3d:
                    data['{}_{}_{}'.format(subset, which, key)], mask, mask_s = utils.build_indices(f(key,self.lower),
                                                                    indices=indices[key],
                                                                    max_sentence_length=self.max_sentence_length,
                                                                    max_post_length=self.max_post_length,
                                                                    mask=True,
                                                                    add=self.min_count==0 or key != 'words'))
                    if key == 'words':
                        self._data['{}_mask_{}_s'.format(subset, which)] = mask_s
                        self._data['{}_mask_{}_w'.format(subset, which)] = mask

                for key in self.train.keys_2d:
                    data['{}_{}_{}'.format(subset, which, key)] = utils.build_indices_2d(f(key,self.lower),
                                                                                         indices=indices[key],
                                                                                         max_length=self.max_post_length)
            
        self._data['embeddings'] = self._map_embeddings()
        self._data['train_labels'] = list(self.train.labels)
        self._data['val_labels'] = list(self.val.labels)
            
    def save(self, filename):
        np.savez_compressed(filename, **self.data)
        with open(filename + '.vocab.json', 'w') as f:
            json.dump(self.indices, f)

