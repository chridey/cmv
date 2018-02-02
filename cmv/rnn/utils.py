import numpy as np

def prepare_data(metadata, vocab, lower,
                 max_post_length, max_sentence_length,
                 frames=False, discourse=False, words=True):

    size = len(metadata)
    shape_w = (size, max_post_length, max_sentence_length)
    shape_s = (size, max_post_length)
    data = np.zeros(shape_w)
    mask = np.zeros(shape_w)
    mask_s = np.zeros(shape_w)

    for d_idx, document in enumerate(metadata):
        for s_idx, sentence in enumerate(document):
            if s_idx >= max_post_length:
                break
                
            ptr = 0
            if words:
                for w_idx, word in enumerate(sentence['words']):
                    if lower:
                        word = word.lower()
                    if word in vocab:
                        if ptr < max_sentence_length:
                            data[d_idx][s_idx][ptr] = vocab[word]
                            mask[d_idx][s_idx][ptr] = 1
                            ptr += 1

        mask_s[d_idx][0] = 1
        mask[d_idx][0][0] = 1

    return data, mask, mask_s
    
def prepare_embeddings(embeddings, vocab):
    #finally, do embeddings
    dimension = len(embeddings.values()[0])
    print('embeddings', len(vocab), dimension)
    embeddings_array = [None] * len(vocab)
    for word in vocab:
        if word in embeddings:
            embeddings_array[vocab[word]] = embeddings[word]
        else:
            embeddings_array[vocab[word]] = np.random.uniform(-1, 1, (dimension,))
   
    embeddings_array = np.array(embeddings_array)
    print(embeddings_array.shape)
    
    return embeddings_array
        
def build_data(metadata, vocab, lower, 
               max_post_length, max_sentence_length, max_title_length,
               frames=False, discourse=False, sentiment=False, words=True):
    print(max_post_length, max_sentence_length, max_title_length)
    
    data = collections.defaultdict(list)    
    new_metadata = collections.defaultdict(list)
    for split in ('train', 'val'):
        for name in ('pos', 'neg'):
            for index, post in enumerate(metadata[split+'_'+name]):
                if split+'_'+name+'_indices' in metadata:
                    op_index = metadata[split+'_'+name+'_indices'][index]
                    op = metadata[split+'_op'][op_index]
                    if split+'_titles' in metadata:
                        title = metadata[split+'_titles'][op_index]
                    else:
                        title = []
                else:
                    op = []
                    title = []
                new_metadata[split+'_op'].append(op)
                new_metadata[split+'_titles'].append(title)
                new_metadata[split+'_rr'].append(post)
                data[split+'_labels'].append(name=='pos')
                
    for split in ('train', 'val'):
        for name in ('op', 'rr', 'titles'):
            key = split+'_'+name
            if name == 'titles':
                shape_w = (len(new_metadata[key]), max_post_length, max_title_length)
            else:
                shape_w = (len(new_metadata[key]), max_post_length, max_sentence_length)

            data[key] = np.zeros(shape_w)
            data[split+'_mask_'+name+'_w'] = np.zeros(shape_w)
            data[split+'_mask_'+name+'_s'] = np.zeros((len(new_metadata[key]), max_post_length))
            print(name, data[key].shape, data[split+'_mask_'+name+'_w'].shape, data[split+'_mask_'+name+'_s'].shape)
            
            for pindex,post in enumerate(new_metadata[key]):
                wctr = 0
                for sindex,sentence in enumerate(post):
                    if sindex >= max_post_length:
                        continue
                    
                    if words:
                        for windex,word in enumerate(sentence['words']):
                            if lower:
                                word = word.lower()
                            if word in vocab:
                                vindex = vocab[word]

                                if name == 'titles':
                                    if wctr < max_title_length:
                                        data[key][pindex][0][wctr] = vindex
                                        data[split+'_mask_'+name+'_w'][pindex][0][wctr] = 1
                                        wctr += 1
                                else:
                                    if windex < max_sentence_length:
                                        data[key][pindex][sindex][windex] = vindex
                                        data[split+'_mask_'+name+'_w'][pindex][sindex][windex] = 1
                            
                    #TODO: add frames and index
                    if frames and 'frames' in sentence:
                        for findex,frame in enumerate(sentence['frames']):
                            if frame is None or findex >= max_sentence_length:
                                continue
                            if frame not in vocab:
                                print(frame)
                                continue
                            vindex = vocab[frame]
                            data[key][pindex][sindex][findex] = vindex
                            data[split+'_mask_'+name+'_w'][pindex][sindex][findex] = 1

                    if discourse:
                        if 'inter_discourse' in sentence:
                            rel = sentence['inter_discourse']
                            vindex = vocab[rel]
                            data[key][pindex][sindex][0] = vindex

                        data[split+'_mask_'+name+'_s'][pindex][sindex] = 1
                        data[split+'_mask_'+name+'_w'][pindex][sindex][0] = 1

                    if data[split+'_mask_'+name+'_w'][pindex][sindex].sum() > 0:
                        data[split+'_mask_'+name+'_s'][pindex][sindex] = 1

                #make sure to add frames if there is no frame index between two
                if frames:
                    for sindex in range(max_post_length):
                        if data[split+'_mask_'+name+'_s'][pindex][sindex] and (sindex==0 or data[split+'_mask_'+name+'_s'][pindex][sindex-1]==1) and (sindex==max_post_length-1 or data[split+'_mask_'+name+'_s'][pindex][sindex+1]==1):
                            data[split+'_mask_'+name+'_s'][pindex][sindex] = 1
                            data[split+'_mask_'+name+'_w'][pindex][sindex][0] = 1
                            
                data[split+'_mask_'+name+'_w'][pindex][0][0] = 1
                data[split+'_mask_'+name+'_s'][pindex][0] = 1                
                        
    #finally, do embeddings
    dimension = len(metadata['embeddings'].values()[0])
    print('embeddings', len(vocab), dimension)
    embeddings_array = [None] * len(vocab)
    for word in vocab:
        if word in metadata['embeddings']:
            embeddings_array[vocab[word]] = metadata['embeddings'][word]
        else:
            embeddings_array[vocab[word]] = np.random.uniform(-1, 1, (dimension,))
   
    data['embeddings'] = np.array(embeddings_array)
    print(data['embeddings'].shape)
    
    return data
