import pandas as pd
import nltk

def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

class ArgumentFeatureExtractor:
    '''features for an entire document'''
    def __init__(self,
                 settings=None,
                 verbose=False):

        if settings is not None:
            self.settings = settings
        else:
            self.settings = {'featureSettings': {'interplay': True}}
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
                
        self.validFeatures = {'interplay': self.getInterplay,
                              }
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

    def getInterplay(self, dataPoint):
        op_all = set(dataPoint.originalPost.getAllWords(True))
        rr_all = set(dataPoint.response.getAllWords(True))
        op_stop = op_all & self.stopwords
        rr_stop = rr_all & self.stopwords
        op_content = op_all - self.stopwords
        rr_content = rr_all - self.stopwords

        key = self.functionFeatures[self.getInterplay]
        all_interplay = calculate_interplay(op_all, rr_all)
        stop_interplay = calculate_interplay(op_stop, rr_stop)
        content_interplay = calculate_interplay(op_content, rr_content)
        keys = [key + '_int', key + '_reply_frac', key + '_op_frac', key + '_jaccard']
        keys = ['all_' + i for i in keys] + ['stop_' + i for i in keys] + ['content_' + i for i in keys]
        return zip(keys, all_interplay + stop_interplay + content_interplay)
        
    def addFeatures(self, dataPoint, featureSettings=None):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']
            
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint))
        return features
        
        return features

if __name__ == '__main__':
    import json
    import sys
    from cmv.featureExtraction.dataPoint import DocumentData

    infile = sys.argv[1]
    outfile = sys.argv[2]

    settings = config.defaultConfig
                                   
    argfe = ArgumentFeatureExtractor()
    
    #load metadata
    with open(infile) as f:
        j = json.load(f)

    training = zip(j['train_titles'], j['train_op'], j['train_pos'], j['train_neg'])
    heldout = zip(j['val_titles'], j['val_op'], j['val_pos'], j['val_neg'])

    featureLabels = {'train_features': pd.DataFrame(), 'val_features': pd.DataFrame()}
    for dataname, dataset in (('train_features', training),
                              ('val_features', heldout)):
        for count,thread in enumerate(dataset):
            if count % 10 == 0:
                print(dataname, count, len(featureLabels[dataname]))
            #for each thread, create a DocumentData object
            pos = DocumentData(thread[0], thread[1], thread[2])
            neg = DocumentData(thread[0], thread[1], thread[3])
        
            #get features for this object and add to the list
            pos_features = argfe.addFeatures(pos)
            neg_features = argfe.addFeatures(neg)
            
            #for sentences, dont forget interaction features
            #TODO
            pos_features['label'] = 1
            pos_features['index'] = len(featureLabels)
            featureLabels[dataname] = featureLabels[dataname].append(pos_features,
                                                                     ignore_index=True)
            neg_features['label'] = 0
            neg_features['index'] = len(featureLabels)            
            featureLabels[dataname] = featureLabels[dataname].append(neg_features,
                                                                     ignore_index=True)            

    train_features = featureLabels['train_features'].fillna(0).to_json()
    val_features = featureLabels['val_features'].fillna(0).to_json()
            
    #save extracted features
    with open(outfile, 'w') as f:
        json.dump({'train_features': train_features,
                   'val_features': val_features,
                   'train_labels': [1,0]*(len(featureLabels['train_features'])//2),
                   'val_labels': [1,0]*(len(featureLabels['val_features'])//2)},
                   f)


