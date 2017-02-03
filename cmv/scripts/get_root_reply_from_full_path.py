import sys
import json

in_file_name = sys.argv[1]
out_file_name = sys.argv[2]

with open(in_file_name) as f:
    metadata = json.load(f)

new_metadata = {}    
for key in metadata:
    if key in ('train_neg', 'train_pos', 'val_neg', 'val_pos'):
        new_metadata[key] = []
        for post in metadata[key]:
            sentences = []
            for index,sentence in enumerate(post):
                if ' '.join(sentence['words']) == 'INTERMEDIATE_DISCUSSION':
                    break
                sentences.append(sentence)
            new_metadata[key].append(sentences)
    else:
        new_metadata[key] = metadata[key]

with open(out_file_name, 'w') as f:
    json.dump(new_metadata, f)
    
