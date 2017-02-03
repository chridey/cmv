from __future__ import print_function

#read all bad files
#for each file
#write INTRA_SENTENCE
#write norel for each
#write newline
#write ADJACENT_SENTENCE
#write norel for each

import sys
import os

base_dir = sys.argv[1]

for bad_dir in ('bad/pos/train/1', 'bad/neg/train/1', 'bad/pos/val/1', 'bad/neg/val/1'):
    full_path_bad_dir = os.path.join(base_dir, bad_dir)
    full_path_parse_dir = full_path_bad_dir.replace('bad', '').replace('pos', 'pos.dp').replace('neg', 'neg.dp')
    print(full_path_bad_dir, full_path_parse_dir)
    for filename in os.listdir(full_path_bad_dir):
        print(filename)
        lines = []
        with open(os.path.join(full_path_bad_dir, filename)) as f:
            for line in f:
                if not len(line.strip()):
                    continue
                lines.append(line.strip())
            
        print(os.path.join(full_path_parse_dir, filename))
        
        with open(os.path.join(full_path_parse_dir, filename), 'w') as fw:        
            print('INTRA_SENTENCE', file=fw)
            print(file=fw)
            for line in lines:
                print('norel\t{0}\t{0}\t{0}\t{0}\t{0}\t{0}\t{1}'.format(-1, line), file=fw)
            print(file=fw)
            print('ADJACENT_SENTENCES', file=fw) #file=fw
            print(file=fw)        
            for i in range(1,len(lines)):
                print('norel\t{0}\t{0}\t{1}\t{0}\t{0}\t{2}'.format(-1,lines[i-1],lines[i]), file=fw)


