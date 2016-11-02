train_pair_file = '/local/nlp/chidey/cmv/pair_task//train_pair_data.jsonlist.bz2'
heldout_pair_file = '/local/nlp/chidey/cmv/pair_task//heldout_pair_data.jsonlist.bz2'

import bz2
import json

from cmv.preprocessing.cleanup import cleanup

def _load_file(filename):
    pairs = []
    with bz2.BZ2File(filename) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def load_train_pairs(filename=train_pair_file):
    return _load_file(filename)

def load_test_pairs(filename=heldout_pair_file):
    return _load_file(filename)

def handle_pairs_input(pairs, num_responses=2**32, cleanup=cleanup):
    op_text = []
    neg_text = []
    pos_text = []

    for pair in pairs:
        op_text.append(list(cleanup(pair['op_text'], op=True)))

        post = []
        for comment in pair['negative']['comments'][:num_responses]:
            post.extend(list(cleanup(comment['body'])))
        neg_text.append(post)

        post = []
        for comment in pair['positive']['comments'][:num_responses]:
            post.extend(list(cleanup(comment['body'])))
        pos_text.append(post)

    return op_text,neg_text,pos_text

def handle_titles(pairs, cleanup=cleanup):
    titles = []
    for pair in pairs:
        titles.append(list(cleanup(pair['op_title'])))
    return titles
