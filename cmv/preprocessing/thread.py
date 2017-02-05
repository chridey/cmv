class Post:
    def __init__(self, metadata):
        self.metadata = metadata

    def getAllWords(self, lower=False):
        words = []
        for sentence in self.metadata:
            if lower:
                for word in sentence['words']:
                    words.append(word.lower())
            else:
                words.extend(sentence['words'])
        return words
    
class Thread:
    def __init__(self, response, originalPost=None, title=None):
        self.response = Post(response)

        self.originalPost = None
        if originalPost is not None:
            self.originalPost = Post(originalPost)

        self.title = None
        if title is not None:
            self.title = Post(title)
        

    
