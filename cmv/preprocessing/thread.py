class Post:
    def __init__(self, metadata):
        self.metadata = metadata

    def getAllWords(self, lower=False):
        words = []
        for sentence in self.metadata['words']:
            if lower:
                for word in sentence:
                    words.append(word.lower())
            else:
                words.extend(sentence)
        return words

    @property
    def keys(self):
        return self.metadata.keys()

    def post(self, key=None, lower=False):
        if key is None:
            return self.metadata
        if not lower:
            return self.metadata[key]
        return [[i.lower() for i in j] for j in self.metadata[key]]
        
class Thread:
    def __init__(self, response, originalPost=None, title=None):
        self.response = Post(response)

        self.originalPost = None
        if originalPost is not None:
            self.originalPost = Post(originalPost)

        self.title = None
        if title is not None:
            self.title = Post(title)
        

    
