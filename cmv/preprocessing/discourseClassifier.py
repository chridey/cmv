from cmv.preprocessing.discourseParser import DiscourseParser

class DiscourseClassifier:
    def __init__(self, discourse_parser=None, verbose=None):
        self.discourse_parser = discourse_parser
        if discourse_parser is None:
            self.discourse_parser = DiscourseParser()
        self.verbose = verbose
        
    def addDiscourse(self, preprocessed_post):
        original_text = '\n'.join([i['original'] for i in preprocessed_post])
        inter_sentence, intra_sentence = self.discourse_parser.parse(original_text)

        assert(len(inter_sentence) == len(preprocessed_post)-1)
        inter_sentence = ['norel'] + inter_sentence
        ret = []
        for i in range(len(inter_sentence)):
            metadata = preprocessed_post[i]
            metadata.update(inter_discourse=inter_sentence[i])
            ret.append(metadata)
            
        return ret
        
