import nltk

class Metadata(object):
    stemmer = nltk.stem.SnowballStemmer('english')
        
    def addMetadata(self, parsed_sentences):
        metadata = dict(original=[],
                        words=[],
                        lemmas=[],
                        stems=[],
                        dependencies=[],
                        ner=[],
                        pos=[])                    
                    
        for parsed_sentence in parsed_sentences:
            metadata['original'].append(unicode(parsed_sentence))
            offset = parsed_sentence[0].i
            empty = {i for (i,word) in enumerate(sent) if not len(word.string.strip())}

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

                words.append(unicode(word))
                lemmas.append(word.lemma_)
                stems.append(self.stemmer.stem(unicode(word)))
                pos.append(word.tag_)
                ner.append('O')
                
                head = word.head.i-offset-sum(1 for x in empty if x < word.head.i-offset)
                if word.dep_ == 'ROOT':
                    head = -1
                dependencies.append((word.dep_.lower(),head))

            metadata['words'].append(words)
            metadata['lemmas'].append(lemmas)
            metadata['stems'].append(stems)
            metadata['pos'].append(pos)
            metadata['ner'].append(ner)
            metadata['dependencies'].append(dependencies)

        return metadata
            

