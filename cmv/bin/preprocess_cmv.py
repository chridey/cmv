import json
import argparse

from cmv.preprocessing.metadataGenerator import MetadataGenerator
from cmv.preprocessing.malleabilityMetadataGenerator import MalleabilityMetadataGenerator
from cmv.preprocessing.dataIterator import DataIterator, PairedDataIterator
from cmv.preprocessing.indexGenerator import IndexGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data for training')
    parser.add_argument('trainfile')
    parser.add_argument('testfile')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--malleability', action='store_true')
    parser.add_argument('--load_metadata')
    parser.add_argument('--save_metadata')

    parser.add_argument('--discourse', action='store_true')
    parser.add_argument('--frames', action='store_true')    
    parser.add_argument('--sentiment', action='store_true')
    
    parser.add_argument('--indices')
    parser.add_argument('--embeddings')
    parser.add_argument('--min_count', type=int, default=0)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--max_sentence_length', type=int, default=256)
    parser.add_argument('--max_post_length', type=int, default=40)

    args = parser.parse_args()

    if args.load_metadata:
        with open(args.load_metadata) as f:
            metadata = json.load(f)
    else:
        generator = MetadataGenerator
        if args.malleability:
            generator = MalleabilityMetadataGenerator

        metadata = generator(args.trainfile, args.testfile, extend=not args.paired,
                             discourse=args.discourse, frames=args.frames,
                             sentiment=args.sentiment).data

    if args.save_metadata:
        with open(args.save_metadata, 'w') as f:
            json.dump(metadata, f)

    if args.indices:
        iterator = DataIterator
        if args.paired:
            iterator = PairedDataIterator

        generator = IndexGenerator(iterator, embeddings=args.embeddings,
                                   min_count=args.min_count, lower=args.lower,
                                   max_sentence_length=args.max_sentence_length,
                                   max_post_length=args.max_post_length)

        generator.save(args.indices)
