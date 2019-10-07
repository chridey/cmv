import collections

MAX_POST_LENGTH = 40
MAX_SENTENCE_LENGTH = 256

def make_sentence_indices(sentence, indices, max_sentence_length, add=True):
    sentence_indices = []
    for word in sentence:
        #word = word.lower()
        word = unicode(word)
        if not len(word):
            continue
        if word not in indices:
            if not add:
                continue
            indices[word] = len(indices)
        sentence_indices.append(indices[word])

    #pad out to max sentence length
    return sentence_indices[:max_sentence_length] + [0]*max(0, max_sentence_length-len(sentence_indices))

def make_post_indices(post, indices, max_sentence_length, max_post_length, add=True):
    post_indices = []

    for sentence in post:
        sentence_indices = make_sentence_indices(sentence, indices, max_sentence_length, add)
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

def preprocess_indices_min_count(data, min_count):
    #first go through training and count the occurrences of each word
    counts = collections.defaultdict(int)
    for i in range(len(data)):
        for sentence in data[i]:
            for token in sentence:
                counts[token] += 1

    #then add to indices only if the count is above the threshold
    indices = {None:0}
    for token in counts:
        if counts[token] >= min_count and token not in indices:
            indices[token] = len(indices)

    return indices

def make_post_indices_and_masks(post, indices, max_sentence_length, max_post_length, mask, add):
    curr_indices = make_post_indices(post, indices, max_sentence_length, max_post_length, add)

    post_length = len(curr_indices)
    sentence_lengths = [len(i) for i in curr_indices]
    if mask:
        post_length = len(post)
        sentence_lengths = [len(i) for i in post]
        
    indices_mask_s = make_sentence_mask(post_length, max_post_length,
                                        sentence_lengths, max_sentence_length)
    indices_mask = make_mask(post_length, max_post_length)

    return curr_indices, indices_mask_s, indices_mask    

#TODO: lowercase or not                                        
def build_indices(data, indices=None, mask=False,
                  max_sentence_length=MAX_SENTENCE_LENGTH,
                  max_post_length=MAX_POST_LENGTH, add=True):
    if indices is None:
        indices = {None:0}
    if len(indices) == 0:
        indices[None] = 0
        
    ret = []
    mask = []
    mask_s = []
    
    for datum in data:
        curr_indices, indices_mask_s, indices_mask = make_post_indices_and_masks(datum,
                                                                                 indices,
                                                                                 max_sentence_length,
                                                                                 max_post_length,
                                                                                 mask,
                                                                                 add)
        mask_s.append(indices_mask_s)
        mask.append(indices_mask)
        ret.append(curr_indices)

    return ret, mask, mask_s

def build_indices_2d(data, indices=None, max_length=MAX_SENTENCE_LENGTH, add=True):
    if indices is None:
        indices = {None:0}
    if len(indices) == 0:
        indices[None] = 0

    ret = []

    for datum in data:
        op_indices = make_sentence_indices(datum, indices, max_length, add)
        ret.append(op_indices)

    return ret
