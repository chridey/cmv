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
    
class DocumentData:
    def __init__(self, title, originalPost, response):
        #for each sentence, we want the original sentence as well as the sentence split into segments based on the altlex
        self.title = Post(title)
        self.originalPost = Post(originalPost)
        self.response = Post(response)
        
    
