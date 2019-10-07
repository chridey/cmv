import json
import argparse

import gensim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data for training')
    parser.add_argument('infile')
    parser.add_argument('embeddings')    
    parser.add_argument('outfile')
        
    args = parser.parse_args()

    with open(args.infile) as f:
        metadata = json.load(f)
        
    #model = gensim.models.Doc2Vec.load_word2vec_format(args.embeddings, binary=False)
    model = gensim.models.KeyedVectors.load_word2vec_format(args.embeddings, binary=False)
    
    embeddings = {}
    for name in ('op', 'pos', 'neg', 'titles'):
        print(name)
        if 'train_' + name not in metadata:
            print("ERROR: {} not in metadata".format(name))
            continue
        for post in metadata['train_' + name]:
            for sentence in post:
                for word in sentence['words']:
                    if word in model:
                        embeddings[word] = list(model[word])
                    if word.lower() in model:
                        embeddings[word.lower()] = map(float, list(model[word.lower()]))
                        
    metadata['embeddings'] = embeddings
    
    with open(args.outfile, 'w') as f:
        json.dump(metadata, f)
