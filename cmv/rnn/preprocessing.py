MAX_POST_LENGTH = 40
MAX_SENTENCE_LENGTH = 256

def make_sentence_indices(sentence, indices, max_sentence_length):
    sentence_indices = []
    for word in sentence:
        #word = word.lower()
        if word not in indices:
            indices[word] = len(indices)
        sentence_indices.append(indices[word])

    #pad out to max sentence length
    return sentence_indices[:max_sentence_length] + [0]*max(0, max_sentence_length-len(sentence_indices))

def make_post_indices(post, indices, max_sentence_length, max_post_length):
    post_indices = []

    for sentence in post:
        post_indices.append(make_sentence_indices(sentence, indices, max_sentence_length))

    #pad out to max post length
    return post_indices[:max_post_length] + [[0]*max_sentence_length]*(max_post_length-len(post_indices))

def make_mask(post_length, max_post_length):
    return [1]*min(max_post_length, post_length) + [0]*max(0,max_post_length-post_length)

def build_indices(op, pos, neg, indices=None, mask=False):
    if not indices:
        indices = {None:0}

    op_ret = []
    resp_ret = []
    op_mask = []
    resp_mask = []
    gold = []

    for i in range(len(op)):
        curr_indices = make_post_indices(op[i], indices, MAX_SENTENCE_LENGTH, MAX_POST_LENGTH)
        if mask:
            indices_mask = make_mask(len(op[i]), MAX_POST_LENGTH)
            op_mask.extend([indices_mask, indices_mask])
        op_ret.extend([curr_indices, curr_indices])

        curr_indices = make_post_indices(pos[i], indices, MAX_SENTENCE_LENGTH, MAX_POST_LENGTH)
        if mask:
            indices_mask = make_mask(len(pos[i]), MAX_POST_LENGTH)
            resp_mask.append(indices_mask)
        resp_ret.append(curr_indices)
    
        curr_indices = make_post_indices(neg[i], indices, MAX_SENTENCE_LENGTH, MAX_POST_LENGTH)
        if mask:
            indices_mask = make_mask(len(neg[i]), MAX_POST_LENGTH)
            resp_mask.append(indices_mask)
        resp_ret.append(curr_indices)

        gold.extend([1,0])

    return op_ret, resp_ret, gold, op_mask, resp_mask, indices
                                                                                                                                                                                                                                                                                                                            
