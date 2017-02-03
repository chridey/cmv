import sys
import json
import re

from pycorenlp import StanfordCoreNLP

def add_sentiment(metadata):
    nlp = StanfordCoreNLP('http://localhost:9000')

    ret = []
    for post_index,post in enumerate(metadata):
        ret_post = []
        for sentence_index,sentence in enumerate(post):
            print(post_index, sentence_index)

            words = ' '.join(sentence['words'])
            #print(words)
            words = re.sub(r'[^\x00-\x7f]',r' ',words)
            
            output = nlp.annotate(str(words),
                                  properties={'annotators': 'sentiment',
                                              'outputFormat': 'json',
                                              'tokenize.whitespace': True
                                              })

            #print(words, len(output['sentences']))
            #assert(len(output['sentences'])==1)
            if type(output) == dict and len(output['sentences']) > 0:
                sentence['sentiment'] = output['sentences'][0]['sentiment']
            else:
                sentence['sentiment'] = 'Neutral'
                
            ret_post.append(sentence)
        ret.append(ret_post)
    return ret
    
if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    with open(infile) as f:
        metadata = json.load(f)

    for key in metadata:
        print(key)
        metadata[key] = add_sentiment(metadata[key])
    
    with open(outfile, 'w') as f:
        json.dump(metadata, f)
