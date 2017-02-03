import sys
import json

discourse_metadata = sys.argv[1]
orig_metadata = sys.argv[2]
out_metadata = sys.argv[3]

print('loading discourse...')
with open(discourse_metadata) as f:
    discourse = json.load(f)

print('loading orig...')    
with open(orig_metadata) as f:
    orig = json.load(f)

print('combining...')        
out = discourse
for key in ('train_op', 'train_titles', 'val_op', 'val_titles'):
    out[key] = orig[key]

print('saving out...')    
with open(out_metadata, 'w') as f:
    json.dump(out, f)
    
