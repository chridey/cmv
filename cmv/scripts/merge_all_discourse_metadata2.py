import json
import sys

in_filename = sys.argv[1]
out_filename = sys.argv[2]

with open(in_filename) as f:
    metadata = json.load(f)

for key in ('train_neg', 'train_pos', 'val_neg', 'val_pos'):
    if key in metadata and key +'1' not in metadata:
        metadata[key +'1'] = metadata[key]
    metadata[key] = []

new_metadata = {}
for key in ('train_neg', 'train_pos', 'val_neg', 'val_pos'):
    if key in metadata and key +'1' not in metadata:
        metadata[key +'1'] = metadata[key]
    new_metadata[key] = []

    metadata[key + '1_indices'] = list(range(len(metadata[key+'1'])))
    new_metadata[key + '_indices'] = []
    
for key in sorted(metadata):
    if key[-1].isdigit():
        print (key, len(metadata[key]), len(metadata[key + '_indices']))
        base = key[:-1]
        if base[-1].isdigit():
            base = base[:-1]
        print(base)
        new_metadata[base].extend(metadata[key])
        print(len(new_metadata[base]))

        print(base + '_indices')
        new_metadata[base + '_indices'].extend(metadata[key + '_indices'])
        print(len(new_metadata[base + '_indices']))

new_metadata['train_op'] = metadata['train_op']
new_metadata['val_op'] = metadata['val_op']
                
with open(out_filename, 'w') as f:
    json.dump(new_metadata, f)
