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

from cmv.rnn.preprocessing import build_indices, build_indices_2d, build_title_indices, preprocess_indices_min_count, build_indices_unbalanced,build_indices_2d_unbalanced

from cmv.preprocessing.preprocess import normalize_from_body

#embeddings_file = '/proj/nlp/corpora/GloVE/twitter/glove.twitter.27B.200d.txt.out.gensim'
#embeddings_file = '/proj/nlpdisk3/nlpusers/chidey/cmv/glove.twitter.27B.200d.txt.out.gensim'

#cluster_file = '/proj/nlp/corpora/GloVE/glove.42B.300d.txt.gensim.clusters.500.json'
cluster_file = '/proj/nlp/corpora/GloVE/glove.42B.300d.txt.gensim.clusters.1000.json'
with open(cluster_file) as f:
    clusters = json.load(f)
unk_cluster = max(clusters.values())+1

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
    
    if label == 'clusters':
        if sentence['words'][index] in clusters:
            return str(clusters[sentence['words'][index]])
        if sentence['words'][index].lower() in clusters:
            return str(clusters[sentence['words'][index].lower()])
        print(sentence['words'][index], 'not found')
        return str(unk_cluster)
    
    if label == 'intra_discourse' and 'intra_discourse' not in sentence:
            return None
        
    return sentence[label][index]    
    
def getMetadata(data, label, lower=False, edus=False):

    ret = []
    for post in data:
        ret_post = []
        for sentence in post:
            if label == 'inter_discourse':
                ret_post.append(sentence.get('inter_discourse', 'norel'))
                continue
            if label == 'sentiment':
                ret_post.append(sentence.get('sentiment', 'Neutral'))
                continue
            if label == 'chars':
                ret_post.append(sentence.get('chars', []))
                continue
            
            ret_sentence = []
            curr_edu = None
                            
            for index in range(len(sentence['words'])):
                token = getToken(sentence, index, label)
                if lower and token is not None and type(token) != int:
                    token = token.lower()
                if edus:
                    edu = getToken(sentence, index, 'edus')
                    if edu != curr_edu and len(ret_sentence):
                        ret_post.append(ret_sentence)
                        curr_edu = edu
                        ret_sentence = []
                ret_sentence.append(token)
            ret_post.append(ret_sentence)
        ret.append(ret_post)
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data')
    parser.add_argument('metadata')
    parser.add_argument('outfile')
    parser.add_argument('embeddings_file')
    parser.add_argument('-d', '--dimension', type=int, default=200)
    parser.add_argument('--max_sentence_length', type=int, default=256)
    parser.add_argument('--max_post_length', type=int, default=40)
    parser.add_argument('--max_chars_length', type=int, default=256)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--edus', action='store_true')    
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--unbalanced', action='store_true')    
    
    args = parser.parse_args()

    print('loading metadata...')
    with open(args.metadata) as f:
        metadata = json.load(f)

    for key in ('train_neg', 'train_pos', 'val_neg', 'val_pos'):
        if key in metadata and key +'1' not in metadata:
            metadata[key +'1'] = metadata[key]
        metadata[key] = []
    
    for key in sorted(metadata):
        print (key, len(metadata[key]))
        if key[-1].isdigit():
            base = key[:-1]
            if key[-1].isdigit():
                base = key[:-1]
            print(base)
            metadata[base].extend(metadata[key])
            print(len(metadata[base]))
            
    indices = collections.defaultdict(dict)
    if args.min_count:
        indices['words'] = preprocess_indices_min_count(getMetadata(metadata['train_pos'],
                                                                        'words', args.lower),
                                                        getMetadata(metadata['train_neg'],
                                                                        'words', args.lower),
                                                        args.min_count)
        indices['govs'] = indices['words'].copy()
        
    print('converting to indices...')
    data = {}

    if args.unbalanced or 'train_op' not in metadata:
        print('assuming unbalanced data, no original post...')
        build_indices = build_indices_unbalanced
        build_indices_2d = build_indices_2d_unbalanced
        
    for i in ('chars', 'words', 'pos', 'frames', 'deps', 'govs', 'clusters', 'intra_discourse'): #, 'causality'):
        data['train_op_'+i], data['train_rr_'+i], train_labels, train_mask_op_s, train_mask_rr_s, train_mask_op_w, train_mask_rr_w, indices[i] = build_indices(getMetadata(metadata.get('train_op',
                                                                                                                                                                                        []),
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     getMetadata(metadata['train_pos'],
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     getMetadata(metadata['train_neg'],
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     max_sentence_length=args.max_sentence_length if i != 'chars' else args.max_chars_length,
                                                                     max_post_length=args.max_post_length,
                                                                     mask=True,
                                                                     indices=indices[i],
                                                                     add=args.min_count==0 or i != 'words')
        
    for i in ('chars', 'words', 'pos', 'frames', 'deps', 'govs', 'clusters', 'intra_discourse'): #, 'causality'):
        data['val_op_'+i], data['val_rr_'+i], val_labels, val_mask_op_s, val_mask_rr_s, val_mask_op_w, val_mask_rr_w, indices[i] = build_indices(getMetadata(metadata.get('val_op',
                                                                                                                                                                          []),
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     getMetadata(metadata['val_pos'],
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     getMetadata(metadata['val_neg'],
                                                                                     i,
                                                                                     args.lower,
                                                                                     args.edus),
                                                                     max_sentence_length=args.max_sentence_length if i != 'chars' else args.max_chars_length,
                                                                     max_post_length=args.max_post_length,
                                                                     mask=True,
                                                                     indices=indices[i],
                                                                     add=args.min_count==0)

    for i in ('inter_discourse', 'sentiment'):
        data['train_op_'+i], data['train_rr_'+i], indices[i] = build_indices_2d(getMetadata(metadata.get('train_op',
                                                                                                         []),
                                                                                     i),
                                                                         getMetadata(metadata['train_pos'],
                                                                                     i),
                                                                         getMetadata(metadata['train_neg'],
                                                                                     i),
                                                                            indices=indices[i],
                                                                            max_length=args.max_post_length)
        data['val_op_'+i], data['val_rr_'+i], indices[i] = build_indices_2d(getMetadata(metadata.get('val_op',
                                                                                                     []),
                                                                                     i),
                                                                         getMetadata(metadata['val_pos'],
                                                                                     i),
                                                                         getMetadata(metadata['val_neg'],
                                                                                     i),
                                                                            indices=indices[i],
                                                                            max_length=args.max_post_length)
                                                                                 

    #data['train_titles'], data['train_mask_titles'], indices['words'] = build_title_indices(getMetadata(metadata['train_titles'], 'words', args.lower), indices=indices['words'], max_length=args.max_sentence_length)
    #data['val_titles'], data['val_mask_titles'], indices['words'] = build_title_indices(getMetadata(metadata['val_titles'], 'words', args.lower), indices=indices['words'], max_length=args.max_sentence_length)
                
    print('loading embeddings...')
    model = gensim.models.Doc2Vec.load_word2vec_format(args.embeddings_file, binary=False)
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

