import nltk

from cmv.preprocessing import conll
from cmv.preprocessing import semafor

stemmer = nltk.stem.SnowballStemmer('english')
frame_parser = semafor.TCPClientSemaforParser()

def getSentenceMetadata(docs, add_frames=True, adjustment=True, verbose=False):
    ret_docs = []
    for doc in docs:
        ret_sents = []
        offset = 0
        conll_string = u''
        for sent in doc:
            ret_words = {'lemmas': [],
                         'words': [],
                         'stems': [],
                         'dependencies': [],
                         'ner': [],
                         'pos': [],
                         'frames': [],
                         'intra_discourse': []}
            if getattr(sent, 'inter_discourse', None) is not None:
                ret_words['inter_discourse'] = sent.inter_discourse
                if verbose:
                    print(sent.inter_discourse)
                
            empty = {i for (i,word) in enumerate(sent) if not len(word.string.strip())}
            #print(offset, empty)
            for index,word in enumerate(sent):
                if not len(word.string.strip()):
                    continue
                
                ret_words['lemmas'].append(word.lemma_)
                ret_words['pos'].append(word.tag_)
                ret_words['ner'].append('O')
                ret_words['stems'].append(stemmer.stem(unicode(word)))
                ret_words['words'].append(unicode(word))
                
                head = word.head.i-offset-sum(1 for x in empty if x < word.head.i-offset)
                if word.dep_ == 'ROOT':
                    head = -1
                ret_words['dependencies'].append((word.dep_.lower(),head))
                
                ret_words['frames'].append(None)
                
                if getattr(sent, 'intra_discourse', None) is not None:
                    ret_words['intra_discourse'].append(sent.intra_discourse[index])
                else:
                    ret_words['intra_discourse'].append(None)
                    
                conll_string += conll.to_conll(index-sum(1 for x in empty if x < word.head.i-offset), word, head)
                
            if adjustment:
                offset += len(sent)
            if not len(ret_words):
                continue
            conll_string += u'\n'
            ret_sents.append(ret_words)

        if verbose:
            print(conll_string)
        if add_frames:
            if len(conll_string.strip()):
                frames = frame_parser.get_frames(conll_string)
                if verbose:
                    print(len(frames), len(doc), len(ret_sents))
                assert(len(frames) == len(ret_sents))

                for dindex,sent in enumerate(ret_sents):
                    #print(frames[dindex])
                    if verbose:
                        print(frames[dindex].tokens)
                        print(ret_sents[dindex]['words'])
                        print(len(frames[dindex].tokens), len(ret_sents[dindex]['words']))
                    assert(len(frames[dindex].tokens) == len(ret_sents[dindex]['words']))
                    for frame,windex in frames[dindex].iterTargets():
                        ret_sents[dindex]['frames'][windex] = frame
            else:
                if verbose:
                    print('conll_string is empty')
        ret_docs.append(ret_sents)
    return ret_docs
