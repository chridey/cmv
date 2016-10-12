import argparse
import json

from spacy.en import English
import gensim
import nltk

import numpy as np

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input,handle_titles

from cmv.rnn.preprocessing import build_indices

from cmv.preprocessing.preprocess import normalize_from_body

nlp = English()
stemmer = nltk.stem.SnowballStemmer('english')

embeddings_file = '/proj/nlp/corpora/GloVE/twitter/glove.twitter.27B.200d.txt.out.gensim'

def cleanup(text):

    cleaned_text = normalize_from_body(text)
    cleaned_text = cleaned_text.replace('cmv:', '').replace('cmv', '').replace('\n', '  ')
    parsed_text = nlp(unicode(cleaned_text))
    return list(parsed_text.sents)
    
    ret_post = []
    for sent in parsed_text.sents:
        ret_sent = []
        for word in sent:
            ret_word = word.string.strip()
            if len(ret_word):
                ret_sent.append(word)
        if len(ret_sent):
            ret_post.append(ret_sent)
    return ret_post

def map_embeddings(indices, embeddings, d):
    embeddings_array = [None] * len(indices)
    for word in indices:
        if word in embeddings:
            embeddings_array[indices[word]] = embeddings[word]
        else:
            embeddings_array[indices[word]] = np.random.uniform(-1, 1, (d,))
    return np.array(embeddings_array)

def getSentenceMetadata(docs):
    ret_docs = []
    for doc in docs:
        ret_sents = []
        offset = 0
        for sent in doc:
            ret_words = {'lemmas': [],
                         'words': [],
                         'stems': [],
                         'dependencies': [],
                         'ner': [],
                         'pos': []}
            empty = {i for (i,word) in enumerate(sent) if not len(word.string.strip())}
            #print(offset, empty)
            for word in sent:
                if not len(word.string.strip()):
                    continue
                ret_words['lemmas'].append(word.lemma_)
                ret_words['pos'].append(word.tag_)
                ret_words['ner'].append('O')
                ret_words['stems'].append(stemmer.stem(unicode(word)))
                ret_words['words'].append(unicode(word))
                index = word.head.i-offset-sum(1 for x in empty if x < word.head.i-offset)
                if word.dep_ == 'ROOT':
                    index = -1
                ret_words['dependencies'].append((word.dep_.lower(),index))
            offset += len(sent)
            ret_sents.append(ret_words)
        ret_docs.append(ret_sents)
    return ret_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data')
    parser.add_argument('outfile')
    parser.add_argument('--root_replies', type=int, default=1)
    parser.add_argument('-d', '--dimension', type=int, default=200)
    parser.add_argument('--max_sentence_length', type=int, default=256)
    parser.add_argument('--max_post_length', type=int, default=40)        
    args = parser.parse_args()
    
    train_pairs = load_train_pairs()
    heldout_pairs = load_test_pairs()

    train_titles = handle_titles(train_pairs, cleanup=cleanup)
    heldout_titles = handle_titles(heldout_pairs, cleanup=cleanup)        
    train_op, train_neg, train_pos = handle_pairs_input(train_pairs, args.root_replies,
                                                        cleanup=cleanup)
    heldout_op, heldout_neg, heldout_pos = handle_pairs_input(heldout_pairs, args.root_replies,
                                                              cleanup=cleanup)
    
    metadata = {}
    for name,docs in (('train_op', train_op),
                    ('train_pos', train_pos),
                    ('train_neg', train_neg),
                    ('train_titles', train_titles),
                    ('val_op', heldout_op),
                    ('val_pos', heldout_pos),
                    ('val_neg', heldout_neg),
                    ('val_titles', heldout_titles)):
        metadata[name] = getSentenceMetadata(docs)
    with open(args.outfile + '.metadata.json', 'w') as f:
        json.dump(metadata, f)

    train_op, train_rr, train_labels, train_mask_op_s, train_mask_rr_s, train_mask_op_w, train_mask_rr_w, indices = build_indices(train_op, train_pos, train_neg, max_sentence_length=args.max_sentence_length, max_post_length=args.max_post_length, mask=True)

    val_op, val_rr, val_labels, val_mask_op_s, val_mask_rr_s, val_mask_op_w, val_mask_rr_w, indices = build_indices(heldout_op, heldout_pos, heldout_neg, indices=indices, max_sentence_length=args.max_sentence_length, max_post_length=args.max_post_length, mask=True,)

    model = gensim.models.Doc2Vec.load_word2vec_format(embeddings_file, binary=False)
    #map embeddings from indices to their word vectors
    embeddings = map_embeddings(indices, model, args.dimension)

    data = {}
    data['embeddings'] = embeddings
    data['train_op'] = train_op
    data['train_rr'] = train_rr
    data['train_labels'] = train_labels
    data['train_mask_op_s'] = train_mask_op_s
    data['train_mask_rr_s'] = train_mask_rr_s
    data['train_mask_op_w'] = train_mask_op_w
    data['train_mask_rr_w'] = train_mask_rr_w
    data['val_op'] = val_op
    data['val_rr'] = val_rr
    data['val_labels'] = val_labels
    data['val_mask_op_s'] = val_mask_op_s
    data['val_mask_rr_s'] = val_mask_rr_s
    data['val_mask_op_w'] = val_mask_op_w
    data['val_mask_rr_w'] = val_mask_rr_w

    np.savez_compressed(args.outfile, **data)
    with open(args.outfile + '.vocab.json', 'w') as f:
        json.dump(indices, f)

