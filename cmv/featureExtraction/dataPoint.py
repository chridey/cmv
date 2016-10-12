
class DocumentData:
    def __init__(self, title, originalPost, response):
        #for each sentence, we want the original sentence as well as the sentence split into segments based on the altlex
        self.title = title
        self.originalPost = originalPost
        self.response = response
        
    
