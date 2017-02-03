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

class DiscourseSpan:
    def __init__(self, span, inter_discourse, intra_discourse):
        self.inter_discourse = inter_discourse
        self.intra_discourse = intra_discourse
        self.span = span

    def __getattr__(self, attr):
        if attr == 'inter_discourse':
            return self.inter_discourse        
        if attr == 'intra_discourse':
            return self.intra_discourse        
        return getattr(self.span, attr)

with open(os.path.join(os.path.split(__file__)[0], 'markers_big')) as f:
    valid_connectives = set(f.read().splitlines())
    
MAX_CONNECTIVE_LENGTH = 6

def getConnective(start, arg1_start, arg1_end, arg2_start, arg2_end, sentence):
    if arg2_start < arg1_start:
        #try for lengths up to 5
        for i in range(MAX_CONNECTIVE_LENGTH):
            connective_start = 0            
            connective = sentence.split()[:i]
            connective = [punct_sub.sub('\\1', loc) for loc in connective]

            if not all(loc.isalpha() for loc in connective):
                connective = sentence.split()[1:i+1]
                connective = [punct_sub.sub('\\1', loc) for loc in connective]
                connective_start += len(sentence.split()[0]) + 1
            else:
                match = starting_punct_re.match(sentence)
                if match is not None:
                    starting_punct = match.group()
                    connective_start += len(starting_punct)
                
            connective = ' '.join(connective).lower()
            if connective in valid_connectives:
                break
    else:
        connective_start = arg1_end-start
        connective_end = arg2_start-start
        connective = unicode(sentence, encoding='utf-8')[connective_start:connective_end].lower()
        if connective not in valid_connectives:
            connective = unicode("\t" + sentence,
                                 encoding='utf-8')[connective_start:connective_end].lower()
            connective_start -= 1
            
    if connective not in valid_connectives:
        print(connective, connective_start, sentence)
        return sentence.split()[0].lower(), 0
        #raise Exception

    return connective, connective_start

def makeDiscourseSpan(sentence, inter_discourse, intra_discourse):
    parsed_sentence = nlp(unicode(sentence, encoding='utf-8'))

    split_sentence = list(parsed_sentence)
    intra_discourse_list = [None for i in range(len(split_sentence))]
    #for each item in the list, need to find it in the tokenized sentence
    for relation, connective, connective_start in intra_discourse:
        counter = 0
        split_connective = connective.split()
        for index,word in enumerate(parsed_sentence):
            if counter == connective_start or counter + len(word.string) > connective_start:
                print(split_sentence, index, split_connective)
                for j in range(len(split_connective)):                    
                    assert(split_connective[j] in parsed_sentence[index+j].text.lower())
                    intra_discourse_list[index+j] = relation
                break
            counter += len(word.string)

        if counter >= len(sentence):
            print('cant find', connective, connective_start, sentence, split_sentence)
            raise Exception
        print(intra_discourse_list)
        
    return DiscourseSpan(parsed_sentence, inter_discourse, intra_discourse_list)

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
                        print('init', init)
                        doc.append(makeDiscourseSpan(first, None, intra[i1]))
                        #doc.append(DiscourseSpan(nlp(unicode(first, encoding='utf-8')), None, None))
                        init = False
                    doc.append(makeDiscourseSpan(second, relation, intra[j1]))
                    #doc.append(DiscourseSpan(nlp(unicode(second, encoding='utf-8')), relation, None))
                else:
                    relation, start, end, arg1_start, arg1_end, arg2_start, arg2_end, sentence = tab_re.split(line)
                    if int(arg1_start) != -1:
                        connective, connective_start = getConnective(int(start), int(arg1_start),
                                                                     int(arg1_end), int(arg2_start),
                                                                     int(arg2_end), sentence)
                        intra[start].append((relation, connective, connective_start))
                    
            print('adj', adj)
            print(intra)
            
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
            if os.path.exists(this_out_file):
                continue
            
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
                
                sentence_metadata = getSentenceMetadata([doc],
                                                        True,
                                                        False)
                sentence_metadata = add_sentiment(sentence_metadata)
                
                metadata[this_name].append(sentence_metadata[0])
                metadata[this_name + '_indices'].append(filename)
            
            with open(this_out_file, 'w') as f:
                json.dump(metadata, f)
            metadata = {}
