from cmv.preprocessing.postPreprocessor import PostPreprocessor

class MalleabilityMetadataGenerator:
    def __init__(self, train_filename, val_filename, num_responses=2**32, extend=True):
        self.train_filename = train_filename
        self.val_filename = val_filename
        self.num_responses = num_responses
        self.extend = extend
        self.border = 'INTERMEDIATE_DISCUSSION'
        
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

        train_pos, train_neg = processData(train)
        val_pos, val_neg = processData(val)

        return dict(train_pos=train_pos,
                    train_neg=train_neg,
                    val_pos=val_pos,
                    val_neg=val_neg)
    
    def processData(self, data):
        neg_text = []
        pos_text = []
        
        for datum in data:
            label = bool(datum['delta_label'])
            text = PostProcessor(datum['selftext'], op=True).processedData
            if label:
                pos_text.append([text])
            else:
                neg_text.append([text])

        return pos_text, neg_text
