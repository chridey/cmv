from cmv.preprocessing.preprocess import normalize_from_body

class PostPreprocessor:
    def __init__(self, data, op=False, lower=False):
        self.data = data
        self.op = op
        self.lower = lower
    
    def __iter__(self):
        for datum in self.data:
            
