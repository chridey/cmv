from altlex.featureExtraction import config

defaultConfig = {
    'altlexSettings': {'featureSettings': config.defaultConfig,
                       'classifierFile': None,
                       'altlexFile': None},
    'featureSettings': {'causal_pct': True, #explicit causal connectives
                        'noncausal_pct': True, #explicit non-causal connectives
                        'causal_score': True, #log prob?
                        'causal_altlex_pct': True,
                        'noncausal_altlex_pct': True,
                        'wordnet_response': True,
                        'verbnet_response': True,
                        'wordnet_title_response_interaction': True, #title, all roots and arguments
                        'wordnet_post_response_interaction': True, #OP, all roots and arguments
                        'verbnet_title_response_interaction': True, #title, all roots and arguments
                        'verbnet_post_response_interaction': True, #OP, all roots and arguments
                        'framenet_response': True,
                        'framenet_altlex_sum': True}
                        #'connective_patterns': True}
                        #intersection of all these things
    }
    
class Config:
    def __init__(self, settings=defaultConfig):
        self.settings = settings

    @property
    def altlexSettings(self):
        return self.settings['altlexSettings']

    @property
    def featureSettings(self):
        return self.settings['featureSettings']
