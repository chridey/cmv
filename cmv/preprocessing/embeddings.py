import gensim

def preprocess_embeddings(embeddings_file, metadata):
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
    
    embeddings = {}
    for name in ('op', 'pos', 'neg', 'titles'):
        if name not in metadata:
            print("WARNING: {} not in metadata".format(name))
            continue
        for post in metadata[name]:
            for sentence in post:
                for word in sentence['words']:
                    if word in model:
                        embeddings[word] = list(model[word])
                    if word.lower() in model:
                        embeddings[word.lower()] = map(float, list(model[word.lower()]))
                        
    return embeddings
