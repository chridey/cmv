MAX_POST_LENGTH = 40
MAX_SENTENCE_LENGTH = 256

def make_sentence_indices(sentence, indices, max_sentence_length):
    sentence_indices = []
    for word in sentence:
        #word = word.lower()
        word = unicode(word)
        if not len(word):
            continue
        if word not in indices:
            indices[word] = len(indices)
        sentence_indices.append(indices[word])

    #pad out to max sentence length
    return sentence_indices[:max_sentence_length] + [0]*max(0, max_sentence_length-len(sentence_indices))

def make_post_indices(post, indices, max_sentence_length, max_post_length):
    post_indices = []

    for sentence in post:
        sentence_indices = make_sentence_indices(sentence, indices, max_sentence_length)
        if len(sentence_indices):
            post_indices.append(sentence_indices)

    #pad out to max post length
    return post_indices[:max_post_length] + [[0]*max_sentence_length]*(max_post_length-len(post_indices))


def make_mask(post_length, max_post_length):
    return [1]*min(max_post_length, post_length) + [0]*max(0,max_post_length-post_length)

def make_sentence_mask(post_length, max_post_length, sentence_lengths, max_sentence_length):
    ret = []
    for i in range(min(post_length, max_post_length)):
        ret.append([1]*min(sentence_lengths[i], max_sentence_length) + [0]*max(0,max_sentence_length-sentence_lengths[i]))
    for i in range(max_post_length-post_length):
        ret.append([0]*max_sentence_length)
    return ret
                                        
def build_indices(op, pos, neg, indices=None, mask=False,
                  max_sentence_length=MAX_SENTENCE_LENGTH,
                  max_post_length=MAX_POST_LENGTH):
    if not indices:
        indices = {None:0}

    op_ret = []
    resp_ret = []
    op_mask = []
    resp_mask = []
    op_mask_s = []
    resp_mask_s = []
    gold = []

    for i in range(len(op)):
        curr_indices = make_post_indices(op[i], indices, max_sentence_length, max_post_length)
        if mask:
            indices_mask_s = make_sentence_mask(len(curr_indices), max_post_length,
                                                [len(j) for j in curr_indices], max_sentence_length)
            op_mask_s.extend([indices_mask_s, indices_mask_s])
            indices_mask = make_mask(len(curr_indices), max_post_length)
            op_mask.extend([indices_mask, indices_mask])
        op_ret.extend([curr_indices, curr_indices])

        curr_indices = make_post_indices(pos[i], indices, max_sentence_length, max_post_length)
        if mask:
            indices_mask_s = make_sentence_mask(len(curr_indices), max_post_length,
                                                [len(j) for j in curr_indices], max_sentence_length)
            resp_mask_s.append(indices_mask_s)
            indices_mask = make_mask(len(curr_indices), max_post_length)
            resp_mask.append(indices_mask)
        resp_ret.append(curr_indices)
    
        curr_indices = make_post_indices(neg[i], indices, max_sentence_length, max_post_length)
        if mask:
            indices_mask_s = make_sentence_mask(len(curr_indices), max_post_length,
                                                [len(j) for j in curr_indices], max_sentence_length)
            resp_mask_s.append(indices_mask_s)
            indices_mask = make_mask(len(curr_indices), max_post_length)
            resp_mask.append(indices_mask)
        resp_ret.append(curr_indices)

        gold.extend([1,0])

    return op_ret, resp_ret, gold, op_mask, resp_mask, op_mask_s, resp_mask_s, indices                                                                                                                                                                                                                                                                                                                            
