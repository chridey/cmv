from cmv.preprocessing.discourseParser import DiscourseParser

class DiscourseClassifier:
    def __init__(self, discourse_parser=None, verbose=None):
        self.discourse_parser = discourse_parser
        if discourse_parser is None:
            self.discourse_parser = DiscourseParser()
        self.verbose = verbose
        
    def addDiscourse(self, preprocessed_post):
        inter_sentence, intra_sentence = self.discourse_parser.parse('\n'.join(preprocessed_post['original']))
        
        intra_discourse = []
        for i in range(len(inter_sentence)):
            split_sentence = preprocessed_post['words'][i]
            intra_discourse_list = [None for i in range(len(split_sentence))]

            #for each item in the list, need to find it in the tokenized sentence
            for relation, connective, connective_start in intra_sentence[i]:
                counter = 0
                split_connective = connective.split()
                for index,word in enumerate(parsed_sentence):
                    if counter == connective_start or counter + len(word) > connective_start:
                        if self.verbose:
                            print(split_sentence, index, split_connective)
                            
                        for j in range(len(split_connective)):                    
                            assert(split_connective[j] in parsed_sentence[index+j].text.lower())
                            intra_discourse_list[index+j] = relation
                        break
                    
                    counter += len(word.string)

                if counter >= len(sentence):
                    print('cant find', connective, connective_start, sentence, split_sentence)
                    raise Exception

                if self.verbose:
                    print(intra_discourse_list)
                    
            intra_discourse.append(intra_discourse_list)
            
        return dict(inter_discourse=inter_sentence,
                    intra_discourse=intra_discourse)
        
