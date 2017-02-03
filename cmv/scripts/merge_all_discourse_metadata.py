import sys
import json
import os

import spacy

from metadata import getSentenceMetadata

nlp = spacy.en.English()

op_dir = sys.argv[1]
out_file = sys.argv[2]

assert( len(sys.argv) > 3 )

metadata = {}
for filename in sys.argv[3:]:
    print(filename)
    with open(filename) as f:
        j = json.load(f)

    for key in j:
        print(key)
        metadata[key] = j[key]

for name in ('train', 'val'):
    full_path_op_dir = os.path.join(op_dir, name)
    print(full_path_op_dir)

    metadata[name + '_op'] = []
    for filename in os.listdir(full_path_op_dir):
        print(filename)
        doc = []
        with open(os.path.join(full_path_op_dir, filename)) as f:
            data = f.read()
            
        parsed_data = nlp(unicode(data, encoding='utf-8'))
        for sentence in parsed_data.sents:
            doc.append(sentence)

        doc_metadata = getSentenceMetadata([doc])
        metadata[name + '_op'].append(doc_metadata[0])

with open(out_file, 'w') as f:
    json.dump(metadata, f)
