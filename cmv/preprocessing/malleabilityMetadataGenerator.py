import json
import bz2

from cmv.preprocessing.postPreprocessor import PostPreprocessor
from cmv.preprocessing.metadataGenerator import MetadataGenerator

class MalleabilityMetadataGenerator(MetadataGenerator):

    @property
    def data(self):
        if self._data is not None:
            return self._data

        train = self._load_file(self.train_filename)
        val = self._load_file(self.val_filename)

        train_metadata = self.processData(train)
        val_metadata = self.processData(val)

        metadata =  dict(train=train_metadata,
                         val=val_metadata)

        if self.test_filename is not None:
            test = self._load_file(self.test_filename)
            test_metadata = self.processData(test)
    
            metadata.update(test=test_metadata)

        return metadata
    
    def processData(self, data):
        neg_text = []
        pos_text = []
        
        for i,datum in enumerate(data):
            label = bool(datum['delta_label'])
            text = PostPreprocessor(datum['selftext'], op=True,
                                 discourse=self.discourse, frames=self.frames).processedData
            if label:
                pos_text.append(text)
            else:
                neg_text.append(text)

        return dict(pos=pos_text, neg=neg_text)
