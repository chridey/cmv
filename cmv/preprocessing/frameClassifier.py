from cmv.preprocessing import semafor

def to_conll(index, word, pos, head, dep):
    '''
    Create a string for a token and metadata in CONLL format:
    1       My      _       PRP$    PRP$    _       2       NMOD    _       _
    2       kitchen _       NN      NN      _       5       SBJ     _       _
    3       no      _       RB      RB      _       5       ADV     _       _
    4       longer  _       RB      RB      _       3       AMOD    _       _
    5       smells  _       VBZ     VBZ     _       0       ROOT    _       _
    6       .       _       .       .       _       5       P       _       _
    
    '''
    return u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'.format(index+1,
                                                                       word,
                                                                       '_',
                                                                       pos,
                                                                       pos,
                                                                       '_',
                                                                       head+1,
                                                                       dep,
                                                                       '_',
                                                                       '_')

class FrameClassifier(object):
    def __init__(self, frame_parser=None, verbose=False):
        self.frame_parser = frame_parser
        if frame_parser is None:
            self.frame_parser = semafor.TCPClientSemaforParser()
        self.verbose = verbose
        
    def addFrames(self, preprocessed_post):
        conll_string = ''                
        for sent in preprocessed_post:
            for i in range(len(sent['words'])):
                word = sent['words'][i]
                pos = sent['pos'][i]
                dep,head = sent['dependencies'][i]
                
                conll_string += to_conll(i, word, pos, head, dep)
            
            conll_string += '\n'

        if self.verbose:
            print (conll_string)

        ret = []            
        metadata = dict(frames=[])
                        
        if len(conll_string.strip()):
            frames = self.frame_parser.get_frames(conll_string)
            
            if self.verbose:
                print(len(frames), len(preprocessed_post))
            assert(len(frames) == len(preprocessed_post))

            for dindex,sent_frames in enumerate(frames):
                sent_frames = frames[dindex]
                metadata = preprocessed_post[dindex]
                
                if self.verbose:
                    print(sent_frames.tokens)
                    print(preprocessed_post['words'][dindex])
                    print(len(sent_frames.tokens), len(metadata['words']))
                assert(len(sent_frames.tokens) == len(metadata['words']))

                metadata['frames'] = [None for i in range(len(sent_frames.tokens))]
                for frame,windex in sent_frames.iterTargets():
                    metadata['frames'][windex] = frame
                ret.append(metadata)
        else:
            if self.verbose:
                print('conll_string is empty')

        return ret
