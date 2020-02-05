class Post:
    def __init__(self, metadata):
        self.metadata = metadata

    def getAll(self, key, lower=False):
        words = []
        for sentence in self.__iter__():
            for word in sentence[key]:
                word = word.strip()
                if lower:
                    words.append(word.lower())
                else:
                    words.append(word)
        return words
        
    def getAllWords(self, lower=False):
        words = []
        for sentence in self.__iter__():
            for word in sentence['words']:
                word = word.strip()
                if lower:
                    words.append(word.lower())
                else:
                    words.append(word)
        return words

    def getAllLemmas(self, lower=False):
        return self.getAll('lemmas', lower)

    def getAllPos(self, lower=False):
        return self.getAll('pos', lower)
        
    def __iter__(self):
        sentences = self.metadata
        if 'data' in self.metadata:
            sentences = self.metadata['data']
            
        for sentence in sentences:
            yield sentence

    @property
    def info(self):
        if 'metadata' in self.metadata:
            return self.metadata['metadata']
        else:
            return {}
    
    @property
    def keys(self):
        return self.metadata.keys()
        
class Thread:
    def __init__(self, response, originalPost=None, title=None):
        self.response = Post(response)

        self.originalPost = None
        if originalPost is not None:
            self.originalPost = Post(originalPost)

        self.title = None
        if title is not None:
            self.title = Post(title)
        

    
