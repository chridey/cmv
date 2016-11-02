def to_conll(index, word, head):
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
                                                                       unicode(word),
                                                                       '_',
                                                                       word.tag_,
                                                                       word.tag_,
                                                                       '_',
                                                                       head+1,
                                                                       word.dep_,
                                                                       '_',
                                                                       '_')
                                                                             
                                                                               
