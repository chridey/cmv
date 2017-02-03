import sys
import json

from cmv.featureExtraction.altlexHandler import AltlexHandler

def add_causal(altlex_handler, metadata):
    ret = []
    for post_index,post in enumerate(metadata):
        ret_post = []
        for sentence_index,sentence in enumerate(post):
            if 'intra_discourse' not in sentence:
                continue
            
            print(post_index, sentence_index)
            sentence['causality'] = [0] * len(sentence['words'])
            
            data = [i for i in altlex_handler.getCausalConnectiveSentences([sentence]) if i.altlexLength]
            if not len(data):
                continue
            #print(len(data),[i._dataDict for i in data])
            predictions = altlex_handler.causalPredictions(data)

            #now we need to match up the data with the predictions
            for dp,label in zip(data, predictions):
                if label == 1:
                    start = len(dp.getPrevLemmas())
                    for i in range(start, start+dp.altlexLength):
                        sentence['causality'][i] = 1
                        sentence['intra_discourse'][i] = 'contingency.cause'
            ret_post.append(sentence)
        ret.append(ret_post)
    return ret

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    
    altlex_handler = AltlexHandler(cache=False)

    with open(infile) as f:
        metadata = json.load(f)

    for key in metadata:
        print(key)
        metadata[key] = add_causal(altlex_handler, metadata[key])
    
    with open(outfile, 'w') as f:
        json.dump(metadata, f)
