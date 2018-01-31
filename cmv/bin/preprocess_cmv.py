import json
import argparse

from cmv.preprocessing.metadataGenerator import MetadataGenerator
from cmv.preprocessing.malleabilityMetadataGenerator import MalleabilityMetadataGenerator
from cmv.preprocessing.embeddings import preprocess_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess CMV data for training')
    parser.add_argument('trainfile')
    parser.add_argument('valfile')
    parser.add_argument('--testfile')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--malleability', action='store_true')
    parser.add_argument('--save_metadata_file')

    parser.add_argument('--discourse', action='store_true')
    parser.add_argument('--frames', action='store_true')    
    parser.add_argument('--embeddings')

    parser.add_argument('--max_examples')
    
    args = parser.parse_args()

    generator = MetadataGenerator
    if args.malleability:
        generator = MalleabilityMetadataGenerator

    metadata = generator(args.trainfile, args.valfile, test_filename=args.testfile, extend=args.paired,
                         discourse=args.discourse, frames=args.frames, num_examples=args.max_examples).data

    if args.embeddings:
        embeddings = preprocess_embeddings(args.embeddings, metadata['train'])
        metadata.update(embeddings=embeddings)
    
    if args.save_metadata_file:
        with open(args.save_metadata_file, 'w') as f:
            json.dump(metadata, f)
