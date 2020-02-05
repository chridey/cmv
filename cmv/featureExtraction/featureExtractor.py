import collections
import json

import pandas as pd
import nltk
import numpy as np

from cmv.preprocessing.thread import Post,Thread

#doc features - total sentences, total paragraphs, total posts, total quotes, total URLs
#sentence features - distance from start/end of paragraph/post, paragraph distance from start/end of post, distance from previous/next quote/url
                                
def extract_features(metadata):
    features = []
    #for each sentence, extract distance from start/end of paragraph/post/thread                                                                                                                                                 
    #also extract paragraph distance from start/end of post/thread and post distance from start/end of thread                                                                                                                    
    #distance from previous/next quote/url (only do this within a single post or across posts? i think only within a single post)                                                                                                
    #0 if N/A                                                                                                                                                                                                                    
    #also is_title feature                                                                                                                                                                                                       

    #keep track of breaks, the location of each of these                                                                                                                                                                         
    paragraph_breaks = [0]
    post_breaks = []
    quote_breaks = []
    url_breaks = []

    #for each quote, url, and paragraph, what post does it belong to
    paragraph_post_breaks = [0]
    
    thread = []
    
    sentence_index = 0
    paragraph_index = 0
    for sentence in metadata:
        if not len(sentence['words']):
            continue
        if sentence['words'][0].lower() == 'intermediate_discussion':
            post_breaks.append(sentence_index)
            paragraph_post_breaks.append(len(paragraph_breaks))
        elif sentence['words'][0].lower() == 'q_u_o_t_e':
            quote_breaks.append(sentence_index)
        elif sentence['words'][0].lower() == 'u_r_l':
            url_breaks.append(sentence_index)
        else:
            thread.append(' '.join(sentence['words']))
            if sentence['paragraph_index'] != paragraph_index:
                paragraph_breaks.append(sentence_index)
                paragraph_index = sentence['paragraph_index']
            sentence_index += 1
            
    paragraph_breaks.append(len(thread))
    post_breaks.insert(0,0)
    post_breaks.append(len(thread))
    paragraph_post_breaks.append(len(paragraph_breaks))
    
    #print(paragraph_breaks, post_breaks, quote_breaks, url_breaks, paragraph_post_breaks)
    paragraph_marker, post_marker, quote_marker, url_marker = 0, 0, 0, 0
    for i in range(len(thread)):
        #print(i, paragraph_marker, post_marker, quote_marker, url_marker)
        
        d_from_start_of_paragraph = i-paragraph_breaks[paragraph_marker] + 1
        d_from_start_of_post = i-post_breaks[post_marker] + 1
        d_from_start_of_thread = i + 1
        
        d_from_end_of_paragraph = paragraph_breaks[paragraph_marker+1]-i
        d_from_end_of_post = post_breaks[post_marker+1]-i
        d_from_end_of_thread = len(thread)-i
        
        paragraph_d_from_start_of_post = paragraph_marker-paragraph_post_breaks[post_marker] + 1
        paragraph_d_from_end_of_post = paragraph_post_breaks[post_marker+1]-paragraph_marker
        
        d_from_previous_quote = 0
        if quote_marker != 0 and quote_marker-1 < len(quote_breaks) and quote_breaks[quote_marker-1] > post_breaks[post_marker]:
            d_from_previous_quote = i-quote_breaks[quote_marker-1] + 1
        d_from_next_quote = 0
        if quote_marker < len(quote_breaks) and quote_breaks[quote_marker] < post_breaks[post_marker+1]:
            d_from_next_quote = quote_breaks[quote_marker]-i
        
        d_from_previous_url = 0
        if url_marker != 0 and url_marker-1 < len(url_breaks) and url_breaks[url_marker-1] > post_breaks[post_marker]:
            d_from_previous_url = i-url_breaks[url_marker-1] + 1
        d_from_next_url = 0
        if url_marker < len(url_breaks) and url_breaks[url_marker] < post_breaks[post_marker+1]:
            d_from_next_url = url_breaks[url_marker]-i
        
        features.append([d_from_start_of_paragraph, d_from_start_of_post, d_from_start_of_thread,
                        d_from_end_of_paragraph, d_from_end_of_post, d_from_end_of_thread,
                        paragraph_marker+1, len(paragraph_breaks)-1-paragraph_marker,
                        paragraph_d_from_start_of_post, paragraph_d_from_end_of_post,
                        post_marker+1, len(post_breaks)-1-post_marker,
                        d_from_previous_quote, d_from_next_quote,
                        d_from_previous_url, d_from_next_url])
        
        if paragraph_breaks[paragraph_marker+1] == i+1:
            paragraph_marker += 1
        if post_breaks[post_marker+1] == i+1:
            post_marker += 1
        
        if quote_marker < len(quote_breaks) and quote_breaks[quote_marker] == i+1:
            quote_marker += 1
        if url_marker < len(url_breaks) and url_breaks[url_marker] == i+1:
            url_marker += 1
            
    return thread, features, [len(thread), len(paragraph_breaks)-1, len(post_breaks)-1, len(quote_breaks), len(url_breaks)]


def calculate_interplay(op, rr):
    int_int = 1.*len(set(op) & set(rr))
    if len(set(op)) == 0 or len(set(rr)) == 0:
        return [0,0,0,0]
    return [int_int, int_int/len(set(rr)), int_int/len(set(op)), int_int/len(set(op) | set(rr))]

def calculate_argumentation_features(features, label_dictionary):
    label_count_keys = [i for i in features if i.startswith('arg_comp_count_')]
    values = sum(features[i] for i in features if i in label_count_keys)
    for key in label_count_keys:
        label = key.split('_')[-1]
        features['arg_comp_pct_' + label] = 1.*features[key]/values

    nonarg = 0
    claim = 1
    premise = 2
    if type(label_dictionary) == dict:
        reverse_labels = {j:i for i,j in label_dictionary.items()}
        claim = reverse_labels['claim']
        premise = reverse_labels['premise']
        nonarg = reverse_labels['nonarg']

    premise_counts = features['arg_comp_count_' + str(premise)]
    claim_counts = features['arg_comp_count_' + str(claim)]

    #ratios and indicator variables reflecting whether claims are supported and premises are supporting
    features['premise_claim_ratio'] = premise_counts / (1+claim_counts) #TODO: this has been calculated wrong this whole time
    features['more_than_premise_claim'] = premise_counts >= claim_counts
    features['all_and_none_premise_claim'] = premise_counts > 0 and claim_counts == 0
    features['all_and_none_claim_premise'] = claim_counts > 0 and premise_counts == 0

    #print(sequence)
    #print(chains)
    #print(relation_counts)

    features['pct_supported'] = 0
    if features['supported'] or features['unsupported']:
        features['pct_supported'] = 1.*features['supported']/(features['supported']+features['unsupported'])
    features['pct_supporting'] = 0
    if features['supporting'] or features['unsupporting']:
        features['pct_supporting'] = 1.*features['supporting']/(features['supporting']+features['unsupporting'])

    total = 1.*sum(sum(features['chains'][label]) for label in features['chains'])
    for label in label_dictionary:
        total_label = 1.*sum(features['chains'][label])
        for key,f in (('max',max), ('min',min), ('mean',np.mean), ('sum',sum), ('num',len)):
            features[key + '_chain_' + str(label)] = f(features['chains'][label]) if label in features['chains'] and len(features['chains'][label]) else 0
            features[key + '_chain_' + str(label) + '_pct_of_' + str(label)] = f(features['chains'][label])/total_label if label in features['chains'] and len(features['chains'][label]) else 0
            features[key + '_chain_' + str(label) + '_pct_of_total'] = f(features['chains'][label])/total if label in features['chains'] and len(features['chains'][label]) else 0
        #features['num_chain_' + str(label)] = len(features['chains'][label]) if label in features['chains'] else 0

    return features

class ArgumentFeatureExtractor:
    '''features for an entire document or sentence'''

    emotion_embeddings_file = '/proj/nlp/users/chidey/cmv/ewe_uni.txt'
    
    def __init__(self,
                 settings=None,
                 verbose=False):

        if settings is not None:
            self.settings = settings
        else:
            self.settings = {'featureSettings': {'interplay': True,
                                                 'structure': True}}
            
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        #self.stopwords = set(json.load(open('/proj/nlpdisk3/nlpusers/chidey/cmv/fusion/argument_words.json')))
                
        self.validFeatures = {'interplay': self.getInterplay,
                              'emotion': self.getEmotions,
                              'structure': self.getStructure,
                              'argumentation': self.getArgumentation,
                              'paragraph': self.getParagraph,
                              'coherence': self.getCoherence,
                              }
                              
        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

        self._emotionEmbeddings = None
        self._emotionEmbeddingsDim = None

        self._embeddings = None
        self._embedding_dim = None
        self._embedding_cache = {}

    def _load_embeddings(self, feature_settings):
        self._embeddings = {}
        self._embedding_dim = {}
        for key in feature_settings:
            if type(feature_settings[key]) == dict and 'embedding_file' in feature_settings[key]:
                filename = feature_settings[key]['embedding_file']
                if filename not in self._embedding_cache:
                    embeddings, embedding_dim = self._read_embeddings(filename)
                    self._embedding_cache[filename] = embeddings, embedding_dim
                embeddings, embedding_dim = self._embedding_cache[filename]
                self._embeddings[key] = embeddings
                self._embedding_dim[key] = embedding_dim
                
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._load_embeddings(self.settings['featureSettings'])
        return self._embeddings

    @property
    def embedding_dim(self):
        if self._embedding_dim is None:
            self._load_embeddings(self.settings['featureSettings'])
        return self._embedding_dim    

    def _read_embeddings(self, filename):
        embeddings = {}
        embedding_dim = None
        with open(filename) as f:
            for line in f:
                word, embedding = line.split(' ', 1)
                embedding = embedding.split()
                embeddings[word] = np.array(list(map(float,embedding)))
                if embedding_dim is None:
                    embedding_dim = embeddings[word].shape[0]
                assert(embedding_dim == embeddings[word].shape[0])        
        return embeddings, embedding_dim
    
    @property
    def emotionEmbeddings(self):
        if self._emotionEmbeddings is None:
            self._emotionEmbeddings, self._emotionEmbeddingsDim = self._read_embeddings(self.emotion_embeddings_file)
                    
        return self._emotionEmbeddings

    def getEmotionRepresentation(self, words):
        emotionFunctionByName = self.functionFeatures[self.getEmotions]
        embeddings = [self.emotionEmbeddings.get(word, np.zeros(self._emotionEmbeddingsDim)) for word in words]
        if len(embeddings):
            embedded_sentence = np.stack(embeddings)
            features = np.concatenate([embedded_sentence.max(axis=0),
                                    embedded_sentence.min(axis=0),
                                    embedded_sentence.mean(axis=0)])            
        else:
            if emotionFunctionByName in self.settings.get('featureSettings', {}) and 'pooling' in self.settings['featureSettings'].get(emotionFunctionByName, {}):
                features = np.zeros(self._emotionEmbeddingsDim*len(self.settings['featureSettings'][emotionFunctionByName]['pooling'].split(',')))
            else:
                features = np.zeros(self._emotionEmbeddingsDim*3)
        return features
    
    def getEmotions(self, dataPoint, *args):
        words = dataPoint.response.getAllWords(True)
        features = self.getEmotionRepresentation(words)
        
        keys = ['EMAX' + str(i) for i in range(self._emotionEmbeddingsDim)] + ['EMIN' + str(i) for i in range(self._emotionEmbeddingsDim)] + ['EMEAN' + str(i) for i in range(self._emotionEmbeddingsDim)]
        return dict(zip(keys, features))

    def getEmbedding(self, words, embedding_type, pooling=('mean', 'max', 'min')):
        embeddings = [self.embeddings[embedding_type][word] for word in words if word in self.embeddings[embedding_type]]
        if len(embeddings):
            embedded_sentence = np.stack(embeddings)
            features = []
            for pool in pooling:
                if pool == 'max':
                    features.append(embedded_sentence.max(axis=0))
                elif pool == 'min':
                    features.append(embedded_sentence.min(axis=0))
                elif pool == 'mean':
                    features.append(embedded_sentence.mean(axis=0))
            features = np.concatenate(features)
        else:
            features = np.zeros(self.embedding_dim[embedding_type]*len(pooling))
        return features
    
    def getCoherence(self, dataPoint, *args):
        op_lemmas = dataPoint.originalPost.getAllLemmas(True)
        op_pos = dataPoint.originalPost.getAllPos()
        rr_lemmas = dataPoint.response.getAllLemmas(True)
        rr_pos = dataPoint.response.getAllPos()

        stopwords = self.stopwords
        if self.settings['featureSettings'][self.functionFeatures[self.getCoherence]].get('argument_words') is not None:
            stopwords = self.settings['featureSettings'][self.functionFeatures[self.getCoherence]]['argument_words']
        
        op_content = []
        for lemma, pos in zip(op_lemmas, op_pos):
            if lemma in stopwords or not lemma.isalnum():
                continue
            op_content.append((lemma, pos[0]))
        rr_content = []
        for lemma, pos in zip(rr_lemmas, rr_pos):
            if lemma in stopwords or not lemma.isalnum():
                continue
            rr_content.append((lemma, pos[0]))

        #print(op_content)
        #print(rr_content)
        #get most similar mean/max/min for V/N/J            
        op_embedding, rr_embedding, keys = [], [], []
        for pooling in ('mean', 'min', 'max'):
            for pos in ({'V'}, {'N'}, {'J'}, {'V', 'N', 'J'},
                        set(map(chr, range(65,91)))):
                if self.settings['featureSettings'][self.functionFeatures[self.getCoherence]].get('has_pos', True):
                    op = {'_'.join(i) for i in op_content if i[1][0] in pos}
                    rr = {'_'.join(i) for i in rr_content if i[1][0] in pos}
                else:
                    op = {i[0] for i in op_content if i[1][0] in pos}
                    rr = {i[0] for i in rr_content if i[1][0] in pos}
                #print(pos, op, rr)
                op_embedding.append(self.getEmbedding(op,
                                                      self.functionFeatures[self.getCoherence],
                                                      pooling=(pooling,)))
                rr_embedding.append(self.getEmbedding(rr,
                                                      self.functionFeatures[self.getCoherence],
                                                      pooling=(pooling,)))
                keys.append(' '.join([pooling, ''.join(pos)]))

        a = np.array(op_embedding)
        b = np.array(rr_embedding)
        cos_sims = (a*b).sum(axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))

        return dict(zip(map(lambda x:'coherence_' + x, keys),
                        cos_sims))
        
    def getInterplay(self, dataPoint, *args):
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
        return dict(zip(keys, all_interplay + stop_interplay + content_interplay))

    def getArgumentation(self, dataPoint, startIndex, *args):
        if type(startIndex) != int:
            startIndex = 0
        argumentationFunctionByName = self.functionFeatures[self.getArgumentation]
        predictions = self.settings['featureSettings'][argumentationFunctionByName]['predictions']

        nonarg = 0
        claim = 1
        premise = 2
        if type(self.settings['featureSettings'][argumentationFunctionByName]['labels']) == dict:
            reverse_labels = {j:i for i,j in self.settings['featureSettings'][argumentationFunctionByName]['labels'].items()}
            claim = reverse_labels['claim']
            premise = reverse_labels['premise']
            nonarg = reverse_labels['nonarg']

        response_id = dataPoint.response.info['id']            
        #raw counts of number of claim/premise/nonarg        
        arg_comp_counts = collections.Counter()
        sequence = []
        for index,sentence in enumerate(dataPoint.response):
            if 'Q_U_O_T_E' in sentence['original'] or 'U_R_L' in sentence['original']:
                sequence.append(None)
                continue
            
            prediction = predictions[(response_id,index+startIndex)]
            arg_comp_counts[prediction] += 1
            sequence.append(prediction)

        features = {}
        for label in self.settings['featureSettings'][argumentationFunctionByName]['labels']:
            features['arg_comp_count_' + str(label)] = arg_comp_counts[label]
            #features['arg_comp_pct_' + str(label)] = 1.*arg_comp_counts[label] / sum(arg_comp_counts.values())
            
        #count longest, shortest, average length of chain of claim/premise/nonarg
        chains = collections.defaultdict(list)
        current_chain = 1
        #count supported claims (premise precedes or follows) and unsupported and supporting premises (claim/premise precedes or follows) and unsupporting
        relation_counts = dict(supported=0, unsupported=0, supporting=0, unsupporting=0)
        previous = None
        for i in range(len(sequence)):
            label = sequence[i]
            
            if previous != label:
                if previous is not None:
                    chains[previous].append(current_chain)
                current_chain = 1
            else:
                current_chain += 1
                
            if label == claim:
                if previous == premise or i+1 < len(sequence) and sequence[i+1] == premise:
                    relation_counts['supported'] += 1
                else:
                    relation_counts['unsupported'] += 1
            if label == premise:
                if previous in {claim, premise} or i+1 < len(sequence) and sequence[i+1] in {claim,premise}:
                    relation_counts['supporting'] += 1
                else:
                    relation_counts['unsupporting'] += 1
                    
            previous = label

        if previous is not None:
            chains[previous].append(current_chain)            
        features.update(relation_counts)
        features['chains'] = chains

        #TODO: move everything below this to a separate function
            
        return calculate_argumentation_features(features, self.settings['featureSettings'][argumentationFunctionByName]['labels'])
    
    def getStructure(self, dataPoint, structureFeatures, sentenceLevel):
        if sentenceLevel:
            features = structureFeatures.getFeatures()
        else:
            return structureFeatures.getDocFeatures()
        keys = ['STRUC' + str(i) for i in range(len(features))]
        return dict(zip(keys, features))

    def getParagraph(self, dataPoint, structureFeatures, *args):
        metadata = dataPoint.response.info
        aggregate_features = collections.defaultdict(list)
        data = list(dataPoint.response)

        for i,(start, end) in enumerate(zip(structureFeatures.paragraph_breaks, structureFeatures.paragraph_breaks[1:])):
            #the paragraph indices are not the same because of quotes/urls/etc            
            start = structureFeatures.original_index[start]
            end = structureFeatures.original_index[end]
            paragraph = Thread(dict(metadata=metadata, data=data[start:end]),
                               dataPoint.originalPost.metadata)

            #calculate interplay between paragraph and op
            interplay = self.getInterplay(paragraph)
            for key in interplay:
                aggregate_features['paragraph_' + key].append(interplay[key])

            #calculate length features for each paragraph
            lengths = self.getStructure(paragraph, StructureFeatures(paragraph.response), False)
            #print(lengths)
            for key in lengths:
                aggregate_features['length_' + key].append(lengths[key])

            #calculate argumentation features for each paragraph
            argumentation = self.getArgumentation(paragraph, start)
            for key in argumentation:
                aggregate_features['argumentation_' + key].append(argumentation[key])

            #calculate coherence features for each paragraph
            coherence = self.getCoherence(paragraph)
            for key in coherence:
                aggregate_features['op_' + key].append(coherence[key])
                            
            #calculate interplay (redundancy) between all paragraphs
            for j,(start2, end2) in enumerate(zip(structureFeatures.paragraph_breaks[i+1:], structureFeatures.paragraph_breaks[i+2:])):
                start2 = structureFeatures.original_index[start2]
                end2 = structureFeatures.original_index[end2]
                #print((start,end),(start2,end2))                
                paragraph_interaction = Thread(dict(metadata=metadata, data=data[start:end]),
                                               dict(metadata=metadata, data=data[start2:end2]))

                #only get coherence between consecutive paragraphs
                if j-i == 1:
                    coherence = self.getCoherence(paragraph_interaction)
                    for key in coherence:
                        aggregate_features['inter_' + key].append(coherence[key])
                
                redundancy = self.getInterplay(paragraph_interaction)
                for key in redundancy:
                    aggregate_features['redundancy_' + key].append(redundancy[key])

        features = {}
        for key in aggregate_features:
            if 'chains' in key:
                continue
            #if not(hasattr(aggregate_features[key], '__iter__') or hasattr(aggregate_features[key], '__getitem__')):                
            #    continue
            #if hasattr(aggregate_features[key][0], '__iter__') or hasattr(aggregate_features[key][0], '__getitem__'):
            #    continue
            features['max_' + key] = max(aggregate_features[key])
            features['min_' + key] = min(aggregate_features[key])
            features['sum_' + key] = sum(aggregate_features[key])
            features['mean_' + key] = np.mean(aggregate_features[key])
        return features

    def mergeFeatures(self, dataPoints, featureSets):
        #can sum all structure/length and argumentation* features
        #*except for chains
        #for paragraph features, can just sum/max/mean
        #need to recompute interplay 
        merged_features = {}
        length_features = ['num_paragraphs', 'num_sentences', 'num_words', 'num_characters', 'num_quotes', 'num_urls']
        for key in length_features + ['supported', 'unsupported', 'supporting', 'unsupporting']:
            merged_features[key] = sum(i[key] for i in featureSets)
        merged_features['num_posts'] = 1
        
        #count longest, shortest, average length of chain of claim/premise/nonarg
        chains = collections.defaultdict(list)
        label_names = self.settings['featureSettings'][self.functionFeatures[self.getArgumentation]]['labels']
        for label in label_names:
            key = 'arg_comp_count_' + str(label)
            merged_features[key] = sum(i[key] for i in featureSets)
            for features in featureSets:
                chains[label].extend(features['chains'][label])
        merged_features['chains'] = chains
        argumentation_features = calculate_argumentation_features(merged_features, label_names)
                                                           
        merged_features.update(argumentation_features)
        
        originalPost = dataPoints[0].originalPost
        data = []
        for dataPoint in dataPoints:
            data.extend(list(dataPoint.response))
        merged_response = Thread(data,
                               originalPost)
        interplay_features = self.getInterplay(merged_response)
        merged_features.update(interplay_features)

        #paragraph features
        aggregate_features = collections.defaultdict(list)
        for index,(paragraph,features) in enumerate(zip(dataPoints, featureSets)):
            #calculate interplay between paragraph and op
            for key in interplay_features:
                aggregate_features['paragraph_' + key].append(features[key])

            #calculate length features for each paragraph
            for key in length_features:
                aggregate_features['length_' + key].append(features[key])

            #calculate argumentation features for each paragraph
            for key in set(argumentation_features) - set(length_features):
                aggregate_features['argumentation_' + key].append(features[key])
                                
            #calculate interplay (redundancy) between all paragraphs
            for index2 in range(index+1, len(dataPoints)):                
                paragraph_interaction = Thread(list(paragraph.response), list(dataPoints[index2].response))

                redundancy = self.getInterplay(paragraph_interaction)
                for key in redundancy:
                    aggregate_features['redundancy_' + key].append(redundancy[key])

        for key in aggregate_features:
            if 'chains' in key:
                continue
            #if not(hasattr(aggregate_features[key], '__iter__') or hasattr(aggregate_features[key], '__getitem__')):                
            #    continue
            #if hasattr(aggregate_features[key][0], '__iter__') or hasattr(aggregate_features[key][0], '__getitem__'):
            #    continue
            merged_features['max_' + key] = max(aggregate_features[key])
            merged_features['min_' + key] = min(aggregate_features[key])
            merged_features['sum_' + key] = sum(aggregate_features[key])
            merged_features['mean_' + key] = np.mean(aggregate_features[key])        

        return merged_response, merged_features
            
    def addFeatures(self, dataPoint, featureSettings=None):
        '''add features for the given dataPoint according to which are
        on or off in featureList'''

        features = {}
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']

        structureFunctionByName = self.functionFeatures[self.getStructure]
        paragraphFunctionByName = self.functionFeatures[self.getParagraph]
        structureFeatures = None
        if len({structureFunctionByName, paragraphFunctionByName} & set(self.settings.get('featureSettings', {}))):
            structureFeatures = StructureFeatures(dataPoint.response)
                        
        for featureName in featureSettings:
            assert(featureName in self.validFeatures)
            if featureSettings[featureName]:
                features.update(self.validFeatures[featureName](dataPoint, structureFeatures, False))
        return features
        
        return features

    def extractSentenceFeatures(self, response, post=None, featureSettings=None):
        if featureSettings is None:
            featureSettings = self.settings['featureSettings']
            
        features = []
        structureFeatures = StructureFeatures(response)

        thread = []
        for sentence in response:
            if not len(sentence['words']) or sentence['words'][0].lower() in ('intermediate_discussion',
                                                                              'q_u_o_t_e',
                                                                              'u_r_l'):
                continue

            dataPoint = Thread([sentence], post)
            sentence_features = []
            for featureName in featureSettings:
                assert(featureName in self.validFeatures)
                if featureSettings[featureName]:
                    sentence_features.extend(self.validFeatures[featureName](dataPoint,
                                                                             structureFeatures, True).values())

            features.append(sentence_features)
            thread.append(' '.join(sentence['words']))
                                
        return thread, features
        
class StructureFeatures:
    def __init__(self, response):
        #for each sentence, extract distance from start/end of paragraph/post/thread                                                                                                                                                 
        #also extract paragraph distance from start/end of post/thread and post distance from start/end of thread                                                                                                                    
        #distance from previous/next quote/url (only do this within a single post or across posts? i think only within a single post)                                                                                                
        #0 if N/A                                                                                                                                                                                                                    
        #also is_title feature                                                                                                                                                                                                       

        #keep track of breaks, the location of each of these                                                                                                                                                                         
        self.paragraph_breaks = [0]
        self.post_breaks = []
        self.quote_breaks = []
        self.url_breaks = []
        #for each quote, url, and paragraph, what post does it belong to
        self.paragraph_post_breaks = [0]

        sentence_index = 0
        paragraph_index = 0
        self.thread_len = 0
        self.num_words = 0
        self.num_characters = 0
        self.original_index = []
        for index,sentence in enumerate(response):
            if not len(sentence['words']):
                continue
            if sentence['words'][0].lower() == 'intermediate_discussion':
                self.post_breaks.append(sentence_index)
                self.paragraph_post_breaks.append(len(self.paragraph_breaks))
            elif sentence['words'][0].lower() == 'q_u_o_t_e':
                self.quote_breaks.append(sentence_index)
            elif sentence['words'][0].lower() == 'u_r_l':
                self.url_breaks.append(sentence_index)
            else:
                self.thread_len += 1
                self.original_index.append(index)
                if sentence['paragraph_index'] != paragraph_index:
                    if sentence_index != 0:
                        self.paragraph_breaks.append(sentence_index)
                    paragraph_index = sentence['paragraph_index']
                sentence_index += 1
                self.num_words += len(sentence['words'])
                self.num_characters += len(sentence['original'][0])

        self.original_index.append(index+1)
        self.paragraph_breaks.append(self.thread_len)
        self.post_breaks.insert(0,0)
        self.post_breaks.append(self.thread_len)
        self.paragraph_post_breaks.append(len(self.paragraph_breaks))

        #print(paragraph_breaks, post_breaks, quote_breaks, url_breaks, paragraph_post_breaks)
        self.paragraph_marker = 0
        self.post_marker = 0
        self.quote_marker = 0
        self.url_marker = 0
        self.i = 0

    def getDocFeatures(self):
        keys = ['num_posts', 'num_paragraphs', 'num_sentences', 'num_words', 'num_characters', 'num_quotes', 'num_urls']
        features = [len(self.post_breaks)-1, len(self.paragraph_breaks)-1, self.thread_len, self.num_words, self.num_characters, len(self.quote_breaks), len(self.url_breaks)]
        return dict(zip(keys, features))
                
    def getFeatures(self):
        d_from_start_of_paragraph = self.i-self.paragraph_breaks[self.paragraph_marker] + 1
        d_from_start_of_post = self.i-self.post_breaks[self.post_marker] + 1
        d_from_start_of_thread = self.i + 1

        d_from_end_of_paragraph = self.paragraph_breaks[self.paragraph_marker+1]-self.i
        d_from_end_of_post = self.post_breaks[self.post_marker+1]-self.i
        d_from_end_of_thread = self.thread_len-self.i

        paragraph_d_from_start_of_post = self.paragraph_marker-self.paragraph_post_breaks[self.post_marker] + 1
        paragraph_d_from_end_of_post = self.paragraph_post_breaks[self.post_marker+1]-self.paragraph_marker

        d_from_previous_quote = 0
        if self.quote_marker != 0 and self.quote_marker-1 < len(self.quote_breaks) and self.quote_breaks[self.quote_marker-1] > self.post_breaks[self.post_marker]:
            d_from_previous_quote = self.i-self.quote_breaks[self.quote_marker-1] + 1
        d_from_next_quote = 0
        if self.quote_marker < len(self.quote_breaks) and self.quote_breaks[self.quote_marker] < self.post_breaks[self.post_marker+1]:
            d_from_next_quote = self.quote_breaks[self.quote_marker]-self.i

        d_from_previous_url = 0
        if self.url_marker != 0 and self.url_marker-1 < len(self.url_breaks) and self.url_breaks[self.url_marker-1] > self.post_breaks[self.post_marker]:
            d_from_previous_url = self.i-self.url_breaks[self.url_marker-1] + 1
        d_from_next_url = 0
        if self.url_marker < len(self.url_breaks) and self.url_breaks[self.url_marker] < self.post_breaks[self.post_marker+1]:
            d_from_next_url = self.url_breaks[self.url_marker]-self.i

        features = [d_from_start_of_paragraph, d_from_start_of_post, d_from_start_of_thread,
                    d_from_end_of_paragraph, d_from_end_of_post, d_from_end_of_thread,
                    self.paragraph_marker+1, len(self.paragraph_breaks)-1-self.paragraph_marker,
                    paragraph_d_from_start_of_post, paragraph_d_from_end_of_post,
                    self.post_marker+1, len(self.post_breaks)-1-self.post_marker,
                    d_from_previous_quote, d_from_next_quote,
                    d_from_previous_url, d_from_next_url]

        if self.paragraph_breaks[self.paragraph_marker+1] == self.i+1:
            self.paragraph_marker += 1
        if self.post_breaks[self.post_marker+1] == self.i+1:
            self.post_marker += 1

        if self.quote_marker < len(self.quote_breaks) and self.quote_breaks[self.quote_marker] == self.i+1:
            self.quote_marker += 1
        if self.url_marker < len(self.url_breaks) and self.url_breaks[self.url_marker] == self.i+1:
            self.url_marker += 1

        self.i += 1
        
        return features
