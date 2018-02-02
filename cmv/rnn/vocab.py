def build_vocab(metadata, min_count, lower):
    print('min count is {}'.format(min_count))
    
    counts = collections.Counter()
    for name in ('op', 'pos', 'neg', 'titles'):
        if name not in metadata:
            print("ERROR: {} not in metadata".format(name))
            continue
        
        for post in metadata[name]:
            for sentence in post:
                for word in sentence['words']:
                    if lower:
                        word = word.lower()
                    counts[word] += 1

    vocab = {'UNK': 0}
    for name in ('op', 'pos', 'neg', 'titles'):
        print(name)
        if name not in metadata:
            print("ERROR: {} not in metadata".format(name))
            continue
        
        for post in metadata[name]:
            for index,sentence in enumerate(post):
                feature = 'INDEX_'+str(index)
                if feature not in vocab:
                    vocab[feature] = len(vocab)
                    
                for word in sentence['words']:
                    if lower:
                        word = word.lower()
                    if word not in vocab and counts[word] >= min_count:
                        vocab[word] = len(vocab)

                if 'frames' in sentence:
                    for frame in sentence['frames']:
                        if frame is None:
                            continue
                        if frame not in vocab:
                            vocab['FRAME_' + frame] = len(vocab)

                if 'inter_discourse' in sentence:
                    discourse = sentence['inter_discourse']
                    if discourse is None:
                        continue
                    if discourse not in vocab:
                        vocab['DISCOURSE_' + discourse] = len(vocab)
                    
    print('vocab size is {}'.format(len(vocab)))
    return vocab
