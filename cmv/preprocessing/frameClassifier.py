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
        self.verbose = True
        
    def addFrames(self, preprocessed_post):
        metadata = dict(frames=[])
        print(preprocessed_post['words'], len(preprocessed_post['words']))
        conll_string = ''        
        for i in range(len(preprocessed_post['words'])):
            for j in range(len(preprocessed_post['words'][i])):
                word = preprocessed_post['words'][i][j]
                pos = preprocessed_post['words'][i][j]
                dep,head = preprocessed_post['dependencies'][i][j]
                
                conll_string += to_conll(j, word, pos, head, dep)
            
            conll_string += '\n'

        if self.verbose:
            print (conll_string)
            
        if len(conll_string.strip()):
            frames = self.frame_parser.get_frames(conll_string)
            
            if self.verbose:
                print(len(frames), len(preprocessed_post['words']))
            assert(len(frames) == len(preprocessed_post['words']))

            for dindex,sent_frames in enumerate(frames):
                if self.verbose:
                    print(sent_frames.tokens)
                    print(preprocessed_post['words'][dindex])
                    print(len(sent_frames.tokens), len(preprocessed_post['words'][dindex]))
                assert(len(sent_frames.tokens) == len(preprocessed_post['words'][dindex]))

                metadata['frames'].append([None for i in range(len(sent_frames.tokens))])
                for frame,windex in frames[dindex].iterTargets():
                    metadata['frames'][dindex][windex] = frame
        else:
            if self.verbose:
                print('conll_string is empty')

        return metadata
