import sys
import os

import torch

from allennlp.data import Vocabulary, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.common import Params
from allennlp.data.dataset import Batch
from allennlp.models import Model, archive_model, archival
from allennlp.training import Trainer
from allennlp.nn import util

from cmv.preprocessing.cmvReader import CMVReader
from cmv.rnn.cmvPredictor import CMVPredictor
from cmv.rnn.cmvExtractor import CMVExtractor, extract
from cmv.rnn.hierarchicalDocumentEmbedder import HierarchicalDocumentEmbedder
from cmv.rnn.cmvDiscriminator import CMVDiscriminator
from cmv.rnn.cmvActorCritic import CMVActorCritic
from cmv.rnn.cmvGeneratorTrainer import CMVGeneratorTrainer
from cmv.rnn.cmvDiscriminatorTrainer import CMVDiscriminatorTrainer
from cmv.rnn.cmvActorCriticTrainer import CMVActorCriticTrainer

model_types = {'actor_critic': CMVActorCriticTrainer, 'generator': CMVGeneratorTrainer}

def main(model_dir, model_type, compression_rate, max_sentences, model_index=None):
    print(compression_rate, max_sentences)

    i = 0
    if model_index:
        i = model_index
        
    params = Params.from_file(os.path.join(model_dir, 'model_params.json'))
    ds_params = params.pop('dataset_reader', {})
    data_params = ds_params.pop('data', {})    
    dataset_reader = CMVReader.from_params(ds_params)

    vocab = Vocabulary.from_params(Params({"directory_path": os.path.join(model_dir, 'vocabulary')}))

    val_iterator = DataIterator.from_params(params.pop('generator_iterator'))
        
    cmv_predictor = Model.from_params(params=params.pop('cmv_predictor'), vocab=vocab)
    document_embedder = Model.from_params(params=params.pop('document_embedder'), vocab=vocab)
    cmv_extractor = Model.from_params(params=params.pop('cmv_extractor'))

    cmv_actor_critic_params = params.pop('cmv_actor_critic', None)
    cmv_actor_critic = None
    if cmv_actor_critic_params is not None:
        cmv_actor_critic = Model.from_params(params=cmv_actor_critic_params)

    cmv_discriminator_params = params.pop('cmv_discriminator', None)
    cmv_discriminator = None
    if cmv_discriminator_params is not None:
        cmv_discriminator = Model.from_params(params=cmv_discriminator_params)

    params = dict(document_embedder=document_embedder, cmv_predictor=cmv_predictor,
                  cmv_extractor=cmv_extractor, cmv_actor_critic=cmv_actor_critic)
    if model_type == 'generator':
        params.update(dict(cmv_discriminator=cmv_discriminator))

    model = model_types[model_type](**params)

    data = dataset_reader.read('val', **data_params)    
    data.index_instances(vocab)
        
    while True:
        model_filename = 'model_state_epoch_{}.th'.format(i)
        model_filename = os.path.join(os.path.join(model_dir, model_type), model_filename)

        print(model_filename)
        if not os.path.exists(model_filename):
            break

        #load file then do forward_on_instance
        model_state = torch.load(model_filename, map_location=util.device_mapping(-1))
        model.load_state_dict(model_state)
        model.eval()
        
        val_generator = val_iterator(data,
                                     num_epochs=1,
                                     shuffle=False)

        model._cmv_extractor._compression_rate = compression_rate
        for batch in val_generator:
            #batch is a tensor dict
            document, mask = model._document_embedder(batch['original_post'])
            idxs, probs, gold_loss = model._cmv_extractor(document, mask, batch['label'],
                                                          gold_evidence=batch['weakpoints'],
                                                          n_abs=max_sentences)

            #extracted_sentences = extract(batch['original_post'], idxs)
            #fake_output = model._cmv_predictor(batch['response'], batch['label'], extracted_sentences)
            for bidx,e in enumerate(batch['weakpoints']):
                if int(e.ne(-1).sum()) == 0:
                    continue
                print(e.numpy().tolist())
                print(idxs[bidx].numpy().tolist())
                for idx,sentence in enumerate(batch['original_post']['tokens'][bidx]):
                    o = [model._cmv_predictor.vocab.get_token_from_index(int(index), 'tokens').replace('@@end@@', '').replace('@@UNKNOWN@@', 'UNK') for index in sentence if int(index)]
                    if len(o):
                        print(idx, ' '.join(o))
                print()
                
        #print(model._cmv_predictor.get_metrics(reset=True))
        print(model._cmv_extractor.get_metrics(reset=True))        
        
        i += 1        
        if model_index is not None:
            break
        
if __name__ == '__main__':
    model_dir = sys.argv[1]
    
    model_type = sys.argv[2]
    assert(model_type in model_types.keys())
    
    compression_rates = list(map(float, sys.argv[3].split(',')))
    assert(all(0 <= compression_rate < 1 for compression_rate in compression_rates))
    
    max_sentences = list(map(int, sys.argv[4].split(',')))
    assert(all(max_sentence > 0 for max_sentence in max_sentences))

    model_index = None
    if len(sys.argv) > 5:
        model_index = int(sys.argv[5])
    
    for compression_rate in compression_rates:
        for max_sentence in max_sentences:
            main(model_dir, model_type, compression_rate, max_sentence, model_index)
                
