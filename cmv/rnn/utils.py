import collections

import numpy as np

def prepare_data(metadata, vocab, lower,
                 max_post_length, max_sentence_length,
                 frames=False, discourse=False, words=True):

    size = len(metadata)
    shape_w = (size, max_post_length, max_sentence_length)
    shape_s = (size, max_post_length)
    data = np.zeros(shape_w)
    mask = np.zeros(shape_w)
    mask_s = np.zeros(shape_s)
    if frames:
        data_frames = np.zeros(shape_w)
    if discourse:
        data_discourse = np.zeros(shape_s)
       
    for d_idx, document in enumerate(metadata):
        for s_idx, sentence in enumerate(document):
            if s_idx >= max_post_length:
                break
                
            if words:
                for w_idx, word in enumerate(sentence['words']):
                    if lower:
                        word = word.lower()
                    if word in vocab:
                        if w_idx < max_sentence_length:
                            data[d_idx][s_idx][w_idx] = vocab[word]
                            mask[d_idx][s_idx][w_idx] = 1
            if frames:
                assert('frames' in sentence)
                for w_idx, frame in enumerate(sentence['frames']):
                    if frame is None:
                        continue
                    frame = 'FRAME_' + frame
                    if frame in vocab:
                        if w_idx < max_sentence_length:
                            data_frames[d_idx][s_idx][w_idx] = vocab[frame]
                            mask[d_idx][s_idx][w_idx] = 1
            if discourse:
                assert('inter_discourse' in sentence)
                rel = sentence['inter_discourse']
                if rel is not None:
                    rel = 'DISCOURSE_' + rel
                    if rel in vocab:
                        data_discourse[d_idx][s_idx] = vocab[rel]
                        mask_s[d_idx][s_idx] = 1
            
            if mask[d_idx][s_idx].sum() > 0:
                mask_s[d_idx][s_idx] = 1
                            
        mask[d_idx][0][0] = 1
        mask_s[d_idx][0] = 1

    ret = [mask, mask_s]
    if words:
        ret = [data] + ret
    if frames:
        ret.append(data_frames)
    if discourse:
        ret.append(data_discourse)
    return ret
    
def prepare_embeddings(embeddings, vocab):
    '''
    initialize the embeddings array, setting unmatched words to random
    '''
    
    dimension = len(embeddings.values()[0])
    print('embeddings', len(vocab), dimension)
    embeddings_array = [None] * len(vocab)
    for word in vocab:
        if word in embeddings:
            assert(len(embeddings[word]) == dimension)
            embeddings_array[vocab[word]] = np.array(embeddings[word])
        else:
            embeddings_array[vocab[word]] = np.random.uniform(-1, 1, (dimension,))
   
    embeddings_array = np.array(embeddings_array)
    print(embeddings_array.shape)
    
    return embeddings_array
        
