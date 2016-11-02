import argparse
import json
import re
import collections

from spacy.en import English
import gensim
import nltk

import numpy as np

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input,handle_titles
from cmv.preprocessing import conll
from cmv.preprocessing import semafor

from cmv.rnn.preprocessing import build_indices, preprocess_indices_min_count

from cmv.preprocessing.preprocess import normalize_from_body

#embeddings_file = '/proj/nlp/corpora/GloVE/twitter/glove.twitter.27B.200d.txt.out.gensim'
embeddings_file = '/proj/nlpdisk3/nlpusers/chidey/cmv/glove.twitter.27B.200d.txt.out.gensim'

def map_embeddings(indices, embeddings, d):
    embeddings_array = [None] * len(indices)
    for word in indices:
        if word in embeddings:
            embeddings_array[indices[word]] = embeddings[word]
        else:
            embeddings_array[indices[word]] = np.random.uniform(-1, 1, (d,))
    return np.array(embeddings_array)

def getToken(sentence, index, label):
    if label == 'deps':
        return sentence['dependencies'][index][0]
    if label == 'govs':
        gov = sentence['dependencies'][index][1]
        if gov == -1:
            return 'R_O_O_T'
        return sentence['words'][gov]
    return sentence[label][index]    
    
def getMetadataType(data, label, lower=False):

    ret = []
    for post in data:
        ret_post = []
        for sentence in post:
            ret_sentence = []
            for index in range(len(sentence['words'])):
                token = getToken(sentence, index, label)
                if lower and token is not None:
                    token = token.lower()
                ret_sentence.append(token)
            ret_post.append(ret_sentence)
        ret.append(ret_post)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data')
    parser.add_argument('metadata')
    parser.add_argument('outfile')    
    parser.add_argument('-d', '--dimension', type=int, default=200)
    parser.add_argument('--max_sentence_length', type=int, default=256)
    parser.add_argument('--max_post_length', type=int, default=40)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--min_count', type=int, default=0)
    args = parser.parse_args()

    print('loading metadata...')
    with open(args.metadata) as f:
        metadata = json.load(f)

    indices = collections.defaultdict(dict)
    if args.min_count:
        indices['words'] = preprocess_indices_min_count(getMetadataType(metadata['train_pos'],
                                                                        'words', args.lower),
                                                        getMetadataType(metadata['train_neg'],
                                                                        'words', args.lower),
                                                        args.min_count)
        indices['govs'] = indices['words'].copy()
    print('converting to indices...')
    data = {}    
    for i in ('words', 'pos', 'frames', 'deps', 'govs'):
        data['train_op_'+i], data['train_rr_'+i], train_labels, train_mask_op_s, train_mask_rr_s, train_mask_op_w, train_mask_rr_w, indices[i] = build_indices(getMetadataType(metadata['train_op'],
                                                                                     i,
                                                                                     args.lower),
                                                                     getMetadataType(metadata['train_pos'],
                                                                                     i,
                                                                                     args.lower),
                                                                     getMetadataType(metadata['train_neg'],
                                                                                     i,
                                                                                     args.lower),
                                                                     max_sentence_length=args.max_sentence_length,
                                                                     max_post_length=args.max_post_length,
                                                                     mask=True,
                                                                     indices=indices[i],
                                                                     add=args.min_count==0 or i != 'words')
    for i in ('words', 'pos', 'frames', 'deps', 'govs'):
        data['val_op_'+i], data['val_rr_'+i], val_labels, val_mask_op_s, val_mask_rr_s, val_mask_op_w, val_mask_rr_w, indices[i] = build_indices(getMetadataType(metadata['val_op'],
                                                                                     i,
                                                                                     args.lower),
                                                                     getMetadataType(metadata['val_pos'],
                                                                                     i,
                                                                                     args.lower),
                                                                     getMetadataType(metadata['val_neg'],
                                                                                     i,
                                                                                     args.lower),
                                                                     max_sentence_length=args.max_sentence_length,
                                                                     max_post_length=args.max_post_length,
                                                                     mask=True,
                                                                     indices=indices[i],
                                                                     add=args.min_count==0)

    print('loading embeddings...')
    model = gensim.models.Doc2Vec.load_word2vec_format(embeddings_file, binary=False)
    #map embeddings from indices to their word vectors
    embeddings = map_embeddings(indices['words'], model, args.dimension)

    data['embeddings'] = embeddings
    data['train_labels'] = train_labels
    data['train_mask_op_s'] = train_mask_op_s
    data['train_mask_rr_s'] = train_mask_rr_s
    data['train_mask_op_w'] = train_mask_op_w
    data['train_mask_rr_w'] = train_mask_rr_w
    data['val_labels'] = val_labels
    data['val_mask_op_s'] = val_mask_op_s
    data['val_mask_rr_s'] = val_mask_rr_s
    data['val_mask_op_w'] = val_mask_op_w
    data['val_mask_rr_w'] = val_mask_rr_w

    np.savez_compressed(args.outfile, **data)
    with open(args.outfile + '.vocab.json', 'w') as f:
        json.dump(indices, f)

