import nltk

class Metadata(object):
    stemmer = nltk.stem.SnowballStemmer('english')
        
    def addMetadata(self, parsed_sentences):
        full_metadata = []
        
        for parsed_sentence in parsed_sentences:
            metadata = dict(original=[],
                            words=[],
                            lemmas=[],
                            stems=[],
                            dependencies=[],
                            ner=[],
                            pos=[])                    
            
            try:
                offset = parsed_sentence[0].i
            except IndexError:
                continue
            
            metadata['original'].append(unicode(parsed_sentence.string))
            
            empty = {i for (i,word) in enumerate(parsed_sentence) if not len(word.string.strip())}

            words = []
            lemmas = []
            stems = []
            pos = []
            ner = []
            dependencies = []
            
            for index in range(len(parsed_sentence)):
                word = parsed_sentence[index]
                if not len(word.string.strip()):
                    continue

                words.append(unicode(word.string))
                lemmas.append(word.lemma_)
                stems.append(self.stemmer.stem(unicode(word.string)))
                pos.append(word.tag_)
                ner.append('O')
                
                head = word.head.i-offset-sum(1 for x in empty if x < word.head.i-offset)
                if word.dep_ == 'ROOT':
                    head = -1
                dependencies.append((word.dep_.lower(),head))

            metadata['words'] = words
            metadata['lemmas'] = lemmas
            metadata['stems'] = stems
            metadata['pos'] = pos
            metadata['ner'] = ner
            metadata['dependencies'] = dependencies

            full_metadata.append(metadata)
            
        return full_metadata
            

