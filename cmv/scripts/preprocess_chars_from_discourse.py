import sys
import os
import json
import collections
import re

tab_re = re.compile("\t+")
punct_sub = re.compile('[^a-zA-Z]*([a-zA-Z]+)[^a-zA-Z]*')
starting_punct_re = re.compile('^[^a-zA-z]+')

import spacy
nlp = spacy.en.English()

from metadata import getSentenceMetadata
from add_sentiment import add_sentiment

def makeCharacterMetadata(string):
    return {'chars': ['START'] + list(unicode(string, encoding='utf-8')) + ['END']}

def iterDiscourseDocs(parse_dir, size, skip_unfound=False):
    for filename in range(size):
        full_path_filename = os.path.join(parse_dir, str(filename))
        adj = False
        init = False
        doc = []
        
        if not os.path.exists(full_path_filename):
            print('cant find {}'.format(full_path_filename))
            if skip_unfound:
                continue
            raise Exception
        
        with open(full_path_filename) as f:
            print(filename)
            data = []
            intra = collections.defaultdict(list)
            for line in f:
                if line.startswith('INTRA_SENTENCE') or not len(line.strip()):
                    continue
                
                if line.startswith('ADJACENT_SENTENCES'):
                    adj = True
                    init = True
                    continue
                
                if adj:
                    try:
                        relation, i1, i2, first, j1, j2, second = tab_re.split(line)
                    except Exception:
                        print(line)
                        raise Exception
                    assert((i1 == '-1' or i1.isdigit()) and (i2 == '-1' or i2.isdigit()) and (j1 == '-1' or j1.isdigit()) and (j2 == '-1' or j2.isdigit()))
                    
                    if init:
                        doc.append(makeCharacterMetadata(first))

                        init = False
                    doc.append(makeCharacterMetadata(second))
            
        yield doc,filename
        
if __name__ == '__main__':
    base_dir = sys.argv[1]
    out_file = sys.argv[2]
    #read in the discourse tagged files
    #parse using spacy
    #optionally, load the preprocessed metadata for the OP
    
    #need to add the discourse tag between each sentence
    #call getSentenceMetadata

    all_dirs = False
    if len(sys.argv) > 3:
        if sys.argv[3] == 'all':
            suffix = ''
            all_dirs = True
        else:
            suffix = sys.argv[3]
    else:
        suffix = '1'

    print('getting metadata...')
    metadata = {}

    for data_dir in ('pos.dp/train/'+suffix, 'neg.dp/train/'+suffix, 'pos.dp/val/'+suffix, 'neg.dp/val/'+suffix):
        full_path_data_dir = os.path.join(base_dir, data_dir)
        size = len(os.listdir(full_path_data_dir))
        print(data_dir, size)
        if size < 1:
            continue
        size = max(map(int,os.listdir(full_path_data_dir)))+1
        print(data_dir, size)

    with open(out_file + '.metadata.json') as f:
        old_metadata = json.load(f)
    
    for name,data_dir in (('train_neg', 'neg.dp/train/'+suffix),
                          ('train_pos', 'pos.dp/train/'+suffix),
                          ('val_pos', 'pos.dp/val/'+suffix),
                          ('val_neg', 'neg.dp/val/'+suffix)):
        full_path_data_dir = os.path.join(base_dir, data_dir)
        if all_dirs:
            iter_dirs = os.listdir(full_path_data_dir)
        else:
            iter_dirs = ['']

        for this_dir in iter_dirs:
            this_name = name + str(this_dir)
            this_out_file = out_file + '_' + this_name + '.metadata.json'
            print(this_out_file)
            #if os.path.exists(this_out_file + '.chars'):
            #    continue
            
            full_path_sub_dir = os.path.join(full_path_data_dir, this_dir)
            if len(os.listdir(full_path_sub_dir)) < 1:
                continue
            size = max(map(int,os.listdir(full_path_sub_dir)))+1
            print(name, this_dir, size)
            
            metadata[this_name] = []
            metadata[this_name + '_indices'] = []
            for doc,filename in iterDiscourseDocs(full_path_sub_dir,
                                                  size,
                                                  True):
                
                metadata[this_name].append(doc)
                metadata[this_name + '_indices'].append(filename)

            print(this_name)
            if this_name[-1] == '1' and this_name[-2].isalpha():
                old_name = this_name[:-1]
            else:
                old_name = this_name
                
            for post_index,post in enumerate(old_metadata[old_name]):
                print(post_index)
                assert(len(post) == len(metadata[this_name][post_index]))
                for sentence_index,sentence in enumerate(old_metadata[old_name][post_index]):
                    for key in old_metadata[old_name][post_index][sentence_index]:
                        metadata[this_name][post_index][sentence_index][key] = old_metadata[old_name][post_index][sentence_index][key]
                            
        with open(out_file + '_chars.metadata.json', 'w') as f:
                json.dump(metadata, f)
       
