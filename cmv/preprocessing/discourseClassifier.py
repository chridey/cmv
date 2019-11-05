import json
import os
import pickle
import string
import re

import torch
import numpy as np
from sklearn.externals import joblib

from common.text_utils import WordEncoder, puncs_removed
from pytorch_impl.cnn_pdtb_arg_wp_multiclass_jl import PDTB_Classifier
from pytorch_impl.cnn_pdtb_classifier_utils_pytorch import make_idx_data_for_mtl, get_binary_class_index, get_class_names, get_class_weights, get_W
                                                                                          
class CMVWordEncoder(WordEncoder):
    def __init__(self, pickled_indexer_file):
        super(CMVWordEncoder, self).__init__()

        with open(pickled_indexer_file, 'rb') as f:
            word_map, pos_map, ner_map = pickle.load(f)
        
        self.encoder = word_map
        self.encoder_pos = pos_map
        self.encoder_ner = ner_map
        self.decoder = {j:i for i,j in word_map.items()}
        self.decoder_pos = {j:i for i,j in pos_map.items()}
        self.decoder_ner = {j:i for i,j in ner_map.items()}        

        self.current_args = []
        self.current_pos_args = []
        self.arg_index = 0

    def reset_args(self, args, pos_args):
        self.current_args = args
        self.current_pos_args = pos_args
        self.arg_index = 0

    def word2index(self, word):
        if word in self.encoder:
            return self.encoder[word]
        return 0

    def pos2index(self, pos):
        if pos in self.encoder_pos:
            return self.encoder_pos[pos]
        return 0    
                
    def _encode_args(self, text, max_arg_len = 100):
        text_tokens = []
        pos_tags = []
        ner_tags = []

        _len = 0    
        for ii, token in enumerate(self.current_args[self.arg_index]):
            # only retain most common english punctuation. That is ',.:;?'
            if len(token) == 1 and token in puncs_removed:
                continue
            pos_tags.append(self.pos2index(self.current_pos_args[self.arg_index][ii]))
            ner_tags.append(1)
            if self.nlp(token)[0].like_num:
                text_tokens.append(self.word2index(u"NUM"))
            else:
                text_tokens.append(self.word2index(token))

            _len += 1
            if _len >= max_arg_len:
                break

        self.arg_index += 1
        return text_tokens, pos_tags, ner_tags

    def _encode_wp(self, text):
        text_tokens = []
        pos_tags = []
        ner_tags = []
    
        for ii, token in enumerate(self.current_args[self.arg_index]):
            if len(token) < 3 and (token[0] not in string.ascii_uppercase) and (token not in ['as', 'or', 'so', 'if', 'no', 'do', 'he', 'we', 'me', 'us', 'it']):
                continue
            # if word contains anything other than A-Za-z' ignore it. ' is kept to retain negation
            if re.match('[A-Za-z\']+$', token) is None:
                continue
            pos_tags.append(self.pos2index(self.current_pos_args[self.arg_index][ii]))
            ner_tags.append(1)
            if self.nlp(token)[0].like_num:
                text_tokens.append(self.word2index(u"NUM"))
            else:
                text_tokens.append(self.word2index(token))

        self.arg_index += 1
        return text_tokens, pos_tags, ner_tags    

class WordPairCNNSettings:
    def __init__(self, class_dict, examples_per_class, w2v_file, encoder):
        self.emb_static = True
        self.reg_emb = False
        self.fsz_arg = [2, 3, 4, 5]
        self.nfmaps_arg = 50
        self.fsz_wp = [2, 4, 6, 8]
        self.nfmaps_wp = 100
        self.dsz_arg = []
        self.dsz_wp = []
        self.gate_units_arg = 300
        self.gate_units_wp = 300
        self.nclasses = len(class_dict)
        self.binarize = False
        self.binary_class = get_binary_class_index(class_dict)
        self.dropout_p = 0.5
        self.l2_weight = 1e-4
        self.batch_size = 200
        self.class_names = get_class_names(class_dict)
        self.n_epochs = 30
        self.patience = 5

        self.W = get_W(w2v_file, encoder.encoder)
        self.P = np.identity(len(encoder.encoder_pos))
        self.class_weights = get_class_weights(class_dict, examples_per_class)
        
aux_verbs = set('be can could do have may might must need ought shall should will would'.split())
def find_nonaux_verb(sent, start_index, end_index=0):
    while start_index >= end_index:
        if sent['pos'][start_index][0] == 'V' and sent['lemmas'][start_index] not in aux_verbs:
            return True
        start_index -= 1
    return False

def find_comma_index(sent, split_index):
    comma_index = -1
    for i in range(split_index, len(sent['words'])):
        if sent['words'][i] == ',':
            return i
    return comma_index

def find_nonaux_verb_both_sides_comma(sent, split_index):
    comma_index = find_comma_index(sent, split_index)
    if comma_index == -1:
        return False

    left_side_has_nonaux_verb = find_nonaux_verb(sent, comma_index)
    right_side_has_nonaux_verb = find_nonaux_verb(sent, len(sent['words'])-1, comma_index)

    return left_side_has_nonaux_verb and right_side_has_nonaux_verb
    
def find_nth_previous_word_index(sent, start_index, n):
    count = 0
    while start_index > -len(sent['words']):
        if sent['words'][start_index].isalnum():
            count += 1
            if count == n:
                return start_index
        start_index -= 1
    return None 

keys = ['discourse_label', 'sentence_label', 'C', 'P', 'F', 'P2', 'P+C', 'C+F', 'P2+C', 'CPOS', 'PPOS', 'FPOS', 'P2POS', 'PPOS+CPOS', 'CPOS+FPOS', 'P2POS+CPOS', 'has_nonaux_verb', 'C+has_nonaux_verb', 'has_nonaux_verb_both_sides_comma', 'C+has_nonaux_verb_both_sides_comma', 'class', 'sentence', 'sentence_prev', 'words', 'words_prev', 'POS', 'POS_prev', 'connective_index', 'comma_index']
def extract_metadata(sent, prev, location, length):
    words = sent['words']
    tags = sent['pos']
    prev_words = []
    prev_tags = []
    C = '_'.join(words[location:location+length])

    has_nonaux_verb = False
    has_nonaux_verb_both_sides_comma = find_nonaux_verb_both_sides_comma(sent, location)
    if location > 0:
        has_nonaux_verb = find_nonaux_verb(sent, location-1)
        previous_word_index = find_nth_previous_word_index(sent, location-1, 1)
        if previous_word_index is None:
            P = str(None)
            PPOS = str(None)
        else:
            P = str(words[previous_word_index])
            PPOS = tags[previous_word_index]
    elif prev is None or not len(' '.join(prev['words']).strip()):
        P = str(None)
        PPOS = str(None)
    else:
        prev_words = prev['words']
        prev_tags = prev['pos']        
        previous_word_index = find_nth_previous_word_index(prev, -1, 1)
        if previous_word_index is None:
            P = str(None)
            PPOS = str(None)
        else:                                
            P = prev_words[previous_word_index]
            PPOS = prev_tags[previous_word_index]

    F = words[location+length] if location + length < len(words) else 'EOS'
                        
    CPOS = tags[location]
    FPOS = tags[location+length] if location + length < len(words) else 'EOS'

    P2 = str(None)
    P2POS = str(None)
    discourse_label = 0
    sentence_label = 0
    discourse_class = 'NoRel'
    
    features = [C, P, F, P2, P+'-'+C, C+'-'+F, P2+'-'+C,
                CPOS, PPOS, FPOS, P2POS,
                PPOS+'-'+CPOS, CPOS+'-'+FPOS, P2POS+'-'+CPOS,
                has_nonaux_verb, C+'-'+str(has_nonaux_verb),
                has_nonaux_verb_both_sides_comma,
                C+'-'+str(has_nonaux_verb_both_sides_comma)]
                        
    values = [discourse_label,
              sentence_label] + features + [discourse_class,
                                            words, prev_words,
                                            words, prev_words,
                                            tags, prev_tags,
                                            location,
                                            find_comma_index(sent, location)]
    return dict(zip(keys, values))
                                    

class DiscourseClassifier:
    def __init__(self, discourse_parser=None, verbose=None):
        currdir = os.path.abspath(os.path.dirname(__file__))

        #load discourse markers
        self.markers = set()
        self.longest_marker = 0
        with open(os.path.join(currdir,'markers')) as f:
            for line in f:
                marker = tuple(line.strip().split())
                self.markers.add(marker)
                if len(marker) > self.longest_marker:
                    self.longest_marker = len(marker)        

        self.feature_names = ['C', 'P+C','C+F',
                              'CPOS','PPOS','FPOS',
                              'PPOS+CPOS','CPOS+FPOS',
                              'C+has_nonaux_verb']
        
        #load discourse usage classifier
        self.discourse_usage_classifier = joblib.load(os.path.join(currdir,
                                                                   'discourse_usage_prediction.model'))
        self.discourse_usage_vectorizer = joblib.load(os.path.join(currdir,
                                                                   'discourse_usage_prediction.vectorizer'))

        #load classifiers for finding the arguments (intra- or inter-sentence)
        self.sentence_classifier = joblib.load(os.path.join(currdir,
                                                                   'sentence_prediction.model'))
        self.sentence_vectorizer = joblib.load(os.path.join(currdir,
                                                                   'sentence_prediction.vectorizer'))
        self.sentence_classifier_ambiguous_connectives = joblib.load(os.path.join(currdir,
                                                                                  'sentence_prediction_ambiguous_connectives.model'))
        self.sentence_vectorizer_ambiguous_connectives = joblib.load(os.path.join(currdir,
                                                                                  'sentence_prediction_ambiguous_connectives.vectorizer'))
        self.ambiguous_connectives = json.load(open(os.path.join(currdir, 'ambiguous_connectives')))

        #finally, load the 15-way discourse relation classifier and other utilities
        #TODO: pytorch model, word indexer, embedding file
        self.word_encoder = CMVWordEncoder(os.path.join(currdir, 'WordEncoder.p'))
        with open(os.path.join(currdir, 'discourse_classes.p'), 'rb') as f:
            _, _, _, class_dict, examples_per_class, _, _ = pickle.load(f)
        self.args = WordPairCNNSettings(class_dict, examples_per_class,
                                        os.path.join(currdir,
                                                     'GoogleNews-vectors-negative300.bin'),
                                        self.word_encoder)
                                                     
        self.discourse_relation_classifier = PDTB_Classifier(self.args)
        self.discourse_relation_classifier.load_state_dict(torch.load('best_params_4'))
        self.discourse_relation_classifier = self.discourse_relation_classifier.cuda()
        
    def find_possible_discourse_connectives(self, words):
        words = list(map(lambda x: x.lower(), words))
        i = 0
        candidates = []
        while i < len(words):
            found = 0

            for j in range(1, self.longest_marker+1):
                if tuple(words[i:i+j]) in self.markers:
                    found = j
            
            if found:
                candidates.append((i, found))
            i += 1
        return candidates
        
    def extract_features(self, features, is_ambiguous=False):
        feature_names = self.feature_names
        if not is_ambiguous:
            feature_names = feature_names[:-1]
            
        return [{i:j for i,j in feature_map.items() if i in set(feature_names)} for feature_map in features]
        
    def identify_discourse_usage(self, discourse_usage_candidates):
        discourse_usage_candidate_locations, discourse_usage_features = zip(*discourse_usage_candidates)
        features = self.discourse_usage_vectorizer.transform(self.extract_features(discourse_usage_features))
        discourse_usage_predictions = self.discourse_usage_classifier.predict(features)
        
        next_candidates = []
        for location, features, prediction in zip(discourse_usage_candidate_locations,
                                        discourse_usage_features,
                                        discourse_usage_predictions):
            if prediction:
                next_candidates.append((location, features))
            #print(prediction, features['C'])
        return next_candidates

    def handle_ambiguous_connective(self, prediction, features):
        if prediction == 1 and features['has_nonaux_verb']:
            return 0
        if prediction == 0 and not (features['has_nonaux_verb'] or features['has_nonaux_verb_both_sides_comma']):
            return 1
        return prediction

    def handle_intra_sentence_arguments(self, features):
        words = features['words']
        tags = features['POS']
        location = features['connective_index']
        if features['has_nonaux_verb']:
            words_prev = words[:location]
            tags_prev = tags[:location]
            words_curr = words[location:]
            tags_curr = tags[location:]
        elif features['has_nonaux_verb_both_sides_comma']:
            location = features['comma_index']
            words_prev = words[location+1:]
            tags_prev = tags[location+1:]
            words_curr = words[:location+1]
            tags_curr = tags[:location+1]
        elif location > 0:
            words_prev = words[:location]
            tags_prev = tags[:location]
            words_curr = words[location:]
            tags_curr = tags[location:]            
        else:
            return features['words_prev'], words, features['POS_prev'], tags
        return words_prev, words_curr, tags_prev, tags_curr
    
    def identify_discourse_arguments(self, discourse_argument_candidates):
        discourse_argument_candidate_locations, discourse_argument_features = zip(*discourse_argument_candidates)
        features = self.sentence_vectorizer.transform(self.extract_features(discourse_argument_features))
        sentence_predictions = self.sentence_classifier.predict(features)
        features = self.sentence_vectorizer_ambiguous_connectives.transform(self.extract_features(discourse_argument_features, True))
        sentence_predictions_ambiguous_connectives = self.sentence_classifier_ambiguous_connectives.predict(features)

        next_candidates = []
        for features, sentence_prediction, sentence_prediction_ambiguous in zip(discourse_argument_features, sentence_predictions, sentence_predictions_ambiguous_connectives):
            #print(features['C'], sentence_prediction, sentence_prediction_ambiguous)
            if features['C'] in self.ambiguous_connectives:
                #special logic for this case
                sentence_prediction = self.handle_ambiguous_connective(sentence_prediction_ambiguous,
                                                                       features)
            if sentence_prediction:
                #if we predict the argument is in the previous sentence, just use the entire previous and current sentences as arg1 and arg2
                next_candidates.append((features['words_prev'], features['words'],
                                        features['POS_prev'], features['POS']))
            else:
                #we need to determine if the first argument is before the connective or after a comma
                next_candidates.append(self.handle_intra_sentence_arguments(features))
                
        return list(zip(discourse_argument_candidate_locations, next_candidates))

    def identify_discourse_relations(self, intra_discourse_candidates):
        locations, input_arguments = zip(*intra_discourse_candidates)
        encoded_arguments = []
        for arg1,arg2,tags1,tags2 in input_arguments:
            #print(arg1,arg2)
            
            self.word_encoder.reset_args([arg1, arg2], [tags1, tags2])
            token_ids, tag_ids, ner_ids = self.word_encoder.encode_args(['',''])
            
            self.word_encoder.reset_args([arg1, arg2], [tags1, tags2])            
            token_wp_ids, tag_wp_ids, ner_wp_ids = self.word_encoder.encode_wp(['',''])
            
            #print(token_ids, tag_ids, token_wp_ids, tag_wp_ids)
            encoded_arguments.append(dict(left_arg_words=token_ids[0],
                                          right_arg_words=token_ids[1],
                                          left_arg_pos=tag_ids[0],
                                          right_arg_pos=tag_ids[1],
                                          left_arg_ner=ner_ids[0],
                                          right_arg_ner=ner_ids[1],
                                          left_wp_words=token_wp_ids[0],
                                          right_wp_words=token_wp_ids[1],
                                          left_wp_pos=tag_wp_ids[0],
                                          right_wp_pos=tag_wp_ids[1],
                                          left_wp_ner=ner_wp_ids[0],
                                          right_wp_ner=ner_wp_ids[1],
                                          is_imp=0, y=[0]))
        test_data = make_idx_data_for_mtl(encoded_arguments, 100, 500,
                              self.args.fsz_arg[-1], self.args.fsz_wp[-1])
        X_te_larg, X_te_rarg, X_te_lpos, X_te_rpos, _, _, X_te_wp, X_te_wp_rev, X_te_wp_pos, X_te_wp_rev_pos, _, _, is_imp_te, _ = test_data
        test_data = [X_te_larg, X_te_lpos, X_te_rarg, X_te_rpos, X_te_wp, X_te_wp_pos, X_te_wp_rev, X_te_wp_rev_pos, is_imp_te]

        #print(self.discourse_relation_classifier.word_embed.num_embeddings,
        #      self.discourse_relation_classifier.word_embed.embedding_dim,
        #      self.discourse_relation_classifier.pos_embed.num_embeddings,
        #      self.discourse_relation_classifier.pos_embed.embedding_dim)
        #print(self.args.class_names)
              
        with torch.no_grad():
            self.discourse_relation_classifier.eval()
            test_data = [torch.LongTensor(data).cuda() for data in test_data]
            #print([i.shape for i in test_data])
            #print([(i.max(), i.min()) for i in test_data])
            predictions = self.discourse_relation_classifier(test_data)
        #print(predictions)

        intra_discourse = {}
        for index,((sentence_index, candidate_location, _),
                   prediction) in enumerate(zip(locations,
                                                predictions)):
            if sentence_index not in intra_discourse:
                sentence_length = len(input_arguments[index][0])+len(input_arguments[index][1])
                intra_discourse[sentence_index] = [None] * sentence_length
            intra_discourse[sentence_index][candidate_location] = self.args.class_names[prediction.argmax()]
        return intra_discourse
            
    def addDiscourse(self, preprocessed_post):
        discourse_usage_candidates = []
        prev = None
        #print([i['original'] for i in preprocessed_post])
        for sentence_index,sentence in enumerate(preprocessed_post):
            #first find all possible discourse markers
            candidate_locations = self.find_possible_discourse_connectives(sentence['words'])

            #then extract the features for each candidate location
            for candidate_location,length in candidate_locations:
                #print(candidate_location,length,sentence['words'])
                discourse_usage_candidates.append(((sentence_index, candidate_location, length),
                                                   extract_metadata(sentence, prev,
                                                                    candidate_location, length)))
            prev = sentence

        #print(discourse_usage_candidates)
        if not len(discourse_usage_candidates):
            return add_intra_discourse(preprocessed_post, {})
        
        #now we need to identify the cases where there is an explicit discourse relation
        discourse_argument_candidates = self.identify_discourse_usage(discourse_usage_candidates)

        if not len(discourse_argument_candidates):
            return add_intra_discourse(preprocessed_post, {})
        
        #given the existence of a relation, identify the arguments to the discourse relation
        intra_discourse_candidates = self.identify_discourse_arguments(discourse_argument_candidates)

        #finally, pass these results to the CNN classifier
        intra_discourse = self.identify_discourse_relations(intra_discourse_candidates)

        return add_intra_discourse(preprocessed_post, intra_discourse)

def add_intra_discourse(preprocessed_post, intra_discourse):
    ret = []
    for i in range(len(preprocessed_post)):
        metadata = preprocessed_post[i]
        metadata['intra_discourse'] = [None] * len(metadata['words'])
        if i in intra_discourse:
            metadata['intra_discourse'] = intra_discourse[i]
        ret.append(metadata)

    return ret                
