#get the embeddings only for those words that appear in the data

import sys
import json

import gensim

def getWords(data):

    ret = set()
    for post in data:
        for sentence in post:
            for word in sentence['words']:
                ret.add(word)
                ret.add(word.lower())
    return ret

if __name__ == '__main__':
    embeddings_file = sys.argv[1]
    data_file = sys.argv[2]
    if len(sys.argv) > 3:
        suffix = sys.argv[3]
    else:
        suffix = '.small'
        
    #get all words
    with open(data_file) as f:
        metadata = json.load(f)
        
    words = set()
    for key in metadata.keys():
        print(key)
        if '_indices' in key:
            continue
        words.update(getWords(metadata[key]))

    print(len(words))
    out = [None]
    with open(embeddings_file) as f:
        line = f.readline()
        print(line)
        orig_size,dim = line.split()
        for line in f:
            columns = line.split()
            if columns[0] in words:
                out.append(line)

    out[0] = '{} {}\n'.format(len(out)-1,dim)
    print(len(out))
    with open(embeddings_file + suffix, 'w') as f:
        f.writelines(out)
