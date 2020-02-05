class Post:
    def __init__(self, metadata):
        self.metadata = metadata

    def getAllSentences(self, lower=False):
        sentences = []
        for sentence in self.metadata:
            if lower:
                sentences.append(' '.join(sentence['words']).lower())
            else:
                sentences.append(' '.join(sentence['words']))
        return sentences        
        
    def getAllWords(self, lower=False):
        words = []
        for sentence in self.metadata:
            for word in sentence['words']:
                word = word.strip()
                if lower:
                    words.append(word.lower())
                else:
                    words.append(word)
        return words

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
        

    
