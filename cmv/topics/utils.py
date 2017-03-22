import collections

import numpy as np

#read in all OPs
#generate lists of lower-case words filtered by count or by top N

#encourage sparse topic vectors? additional constraint to maximize l2 norm

#for persuasiveness loss, use max margin with negative samples from same post up to N
#if less than N, sample from posts with similar topic vectors up to N

UNKNOWN = '___UNK___'

MAX_SENTENCE_LENGTH = 32
MAX_POST_LENGTH = 40

def load_data(metadata, embeddings,
              min_count=0, max_count=2**32,
              min_rank=0, max_rank=1,
              indices=None, add=False,
              counts=None,
              hierarchical=False,
              key='train_op'):
    
    #go over this once for counts
    if counts is None:
        counts = collections.Counter()
        for post in metadata[key]:
            for sentence in post:
                for word in sentence['words']:
                    counts[word.lower()] += 1
                
    #go over again for max length
    posts = []
    max_len = 0
    max_sentence_len = 0
    keys = np.array(counts.keys())
    values = np.array(counts.values())
    bottom_k_keys = set(keys[np.argpartition(values, min_rank)[:min_rank]])
    top_k_keys = set(keys[np.argpartition(values, len(values)-max_rank)[len(values)-max_rank:]])

    print(bottom_k_keys)
    print(top_k_keys)
    
    for index,post in enumerate(metadata[key]):
        words = []
        sentences = []
        for sentence in post:
            for word in sentence['words']:
                word = word.lower()
                if word in embeddings and counts[word] >= min_count and counts[word] <= max_count and word not in bottom_k_keys and word not in top_k_keys:
                    words.append(word)
            if len(words) > max_sentence_len:
                max_sentence_len = len(words)
            if hierarchical:
                sentences.append(words)
                words = []
        if hierarchical:
            words = sentences
        if len(words) > max_len:
            max_len = len(words)
        if len(words) > 0:
            posts.append(words)
        else:
            print(index)
            
    print(len(posts), max_len)
    
    if indices is None:
        indices = {0:UNKNOWN}

    #finally convert to indices
    if hierarchical:
        words, mask, indices, embeddings_array = make_indices_mask_3d(posts, max_len, max_sentence_len,
                                                                      indices, add, embeddings)
    else:
        words, mask, indices, embeddings_array = make_indices_mask_2d(posts, max_len,
                                                                      indices, add, embeddings)
        
    print(len(indices))
            
    return words, mask, indices, counts, np.array(embeddings_array, dtype='float32')

def make_indices_mask_2d(posts, max_len, indices, add, embeddings):
    embeddings_array = [np.random.uniform(-.2, .2, embeddings.layer1_size)]

    words = np.zeros((len(posts), max_len))
    mask = np.zeros((len(posts), max_len))
    for i,post in enumerate(posts):
        for j,word in enumerate(post):
            if word not in indices and add:
                indices[word] = len(indices)
                embeddings_array.append(embeddings[word])
            words[i,j] = indices[word]
            mask[i,j] = 1
            
    return word, mask, indices, embeddings_array

def make_indices_mask_3d(posts, max_len, max_sentence_len, indices, add, embeddings):
    embeddings_array = [np.random.uniform(-.2, .2, embeddings.layer1_size)]

    words = np.zeros((len(posts), min(max_len, MAX_POST_LENGTH),
                      min(max_sentence_len, MAX_SENTENCE_LENGTH)))
    mask = np.zeros((len(posts), min(max_len, MAX_POST_LENGTH),
                      min(max_sentence_len, MAX_SENTENCE_LENGTH)))

    for i,post in enumerate(posts):
        for j,sentence in enumerate(post):
            if j >= MAX_POST_LENGTH:
                continue
            for k,word in enumerate(sentence):
                if k >= MAX_SENTENCE_LENGTH:
                    continue
                if word not in indices and add:
                    indices[word] = len(indices)
                    embeddings_array.append(embeddings[word])
                words[i,j,k] = indices[word]
                mask[i,j,k] = 1
                
    return word, mask, indices, embeddings_array

def generate_negative_samples(batch_size, negs, max_len, words, mask):
    inds = np.random.randint(0, words.shape[0], (batch_size, negs))
    neg_words = np.zeros((batch_size, negs, max_len)).astype('int32')
    neg_masks = np.zeros((batch_size, negs, max_len)).astype('float32')
    for index, i in enumerate(inds):
        neg_words[index] = words[i]
        neg_masks[index] = mask[i]

    return neg_words, neg_masks



