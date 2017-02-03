from __future__ import print_function

import sys
import re
import os

from spacy.en import English

from cmv.preprocessing.preprocess import normalize_from_body
from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input,handle_titles,load_train_op_malleability,load_heldout_op_malleability,handle_op_malleability

nlp = English()

cmv_pattern = re.compile('cmv:?', re.IGNORECASE)

def cleanup(text, op=False):

    cleaned_text = normalize_from_body(text, op=op, lower=False)
    cleaned_text = cmv_pattern.sub('', cleaned_text)
    cleaned_text = cleaned_text.replace('\t', ' ')
    
    parsed_text = [nlp(unicode(i)) for i in cleaned_text.split('\n')]
    return parsed_text



if __name__ == '__main__':

    base_dir = sys.argv[1] #'/proj/nlpdisk3/nlpusers/chidey/cmv/discourse/origplus/'
    datasets = []
    
    print('loading data...')
    pairs = True
    if len(sys.argv) > 2:
        print('loading ', sys.argv[2])
        if 'op_task' not in sys.argv[2]:
            train = load_train_pairs(sys.argv[2])
        else:
            pairs = False
            train = load_train_op_malleability(sys.argv[2])
    else:
        train = load_train_pairs()
        
    print('loaded {} items'.format(len(train)))
    
    if len(sys.argv) > 3:
        print('loading ', sys.argv[3])
        if 'op_task' not in sys.argv[3]:
            heldout = load_test_pairs(sys.argv[3])            
        else:
            assert(not pairs)
            heldout = load_heldout_op_malleability(sys.argv[3])
    else:
        heldout = load_test_pairs()
    print('loaded {} items'.format(len(heldout)))

    extend=False
    if len(sys.argv) > 4:
        extend = int(sys.argv[4]) == 1
    
    border = 'INTERMEDIATE_DISCUSSION'
    print(extend)

    if pairs:
        subsets = ('titles', 'op', 'neg', 'pos')
        print('handling train titles...')
        datasets.append(handle_titles(train, cleanup=cleanup))

        print('handling train posts...')
        datasets.extend(handle_pairs_input(train,
                                           cleanup=cleanup, border=border, extend=extend))

        print('handling val...')    
        datasets.append(handle_titles(heldout, cleanup=cleanup))
        datasets.extend(handle_pairs_input(heldout,
                                           cleanup=cleanup, border=border, extend=extend))
        
    else:
        subsets = ('neg', 'pos')
        datasets.extend(handle_op_malleability(train, cleanup=cleanup))
        datasets.extend(handle_op_malleability(heldout, cleanup=cleanup))

    print(len(datasets))
    print('writing to file...')
    counter = 0
    for second_level in ('train', 'val'):
        for first_level in subsets:
            for thread_index,thread in enumerate(datasets[counter]):
                if extend or first_level not in ('pos', 'neg'):
                    thread = [thread]
                for post_index,post in enumerate(thread):
                    print(first_level, second_level, counter, thread_index, post_index)
                    out_dir = os.path.join(os.path.join(base_dir, first_level), second_level)
                    if first_level in ('pos', 'neg'):
                        #if post_index < 3:
                        out_dir = os.path.join(out_dir, str(post_index+1))
                        #else:
                        #    out_dir = os.path.join(out_dir, '4+')
                            
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                        
                    with open(os.path.join(out_dir, str(thread_index)), 'w') as f:
                        for paragraph in post:
                            for sent in list(paragraph.sents):
                                if len(sent):
                                    print(sent, file=f)
                            print(file=f)
            counter += 1
