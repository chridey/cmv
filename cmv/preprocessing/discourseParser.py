import sys
import subprocess
import tempfile
import re

class DiscourseParser:
    tab_re = re.compile("\t+")
    punct_sub = re.compile('[^a-zA-Z]*([a-zA-Z]+)[^a-zA-Z]*')
    starting_punct_re = re.compile('^[^a-zA-z]+')
    MAX_CONNECTIVE_LENGTH = 6
    
    def __init__(self):
        self.command = 'java -Xmx8G -cp bin:lib/* discourse.tagger.app.DiscourseTaggingRunner '
        with open(os.path.join(os.path.split(__file__)[0], 'markers_big')) as f:
            self.valid_connectives = set(f.read().splitlines())

    def getConnective(self, start, arg1_start, arg1_end, arg2_start, arg2_end, sentence):
        if arg2_start < arg1_start:
            #try for lengths up to 5
            for i in range(self.MAX_CONNECTIVE_LENGTH):
                connective_start = 0            
                connective = sentence.split()[:i]
                connective = [self.punct_sub.sub('\\1', loc) for loc in connective]

                if not all(loc.isalpha() for loc in connective):
                    connective = sentence.split()[1:i+1]
                    connective = [self.punct_sub.sub('\\1', loc) for loc in connective]
                    connective_start += len(sentence.split()[0]) + 1
                else:
                    match = self.starting_punct_re.match(sentence)
                    if match is not None:
                        starting_punct = match.group()
                        connective_start += len(starting_punct)

                connective = ' '.join(connective).lower()
                if connective in self.valid_connectives:
                    break
        else:
            connective_start = arg1_end-start
            connective_end = arg2_start-start
            connective = unicode(sentence, encoding='utf-8')[connective_start:connective_end].lower()
            if connective not in self.valid_connectives:
                connective = unicode("\t" + sentence,
                                     encoding='utf-8')[connective_start:connective_end].lower()
                connective_start -= 1

        if connective not in self.valid_connectives:
            print(connective, connective_start, sentence)
            return sentence.split()[0].lower(), 0
            #raise Exception

        return connective, connective_start


    def processDiscourse(self, discourse_data, length):
        adj = False
        inter_sentence = []
        intra_sentence = {i:[] for i in range(length)}
        
        for line in discourse_data.split('\n'):
            if line.startswith('INTRA_SENTENCE') or not len(line.strip()):
                continue
            if line.startswith('ADJACENT_SENTENCES'):
                adj = True
                continue
            if adj:
                try:
                    relation, i1, i2, first, j1, j2, second = self.tab_re.split(line)
                except Exception:
                    print(line)
                    raise Exception
                assert((i1 == '-1' or i1.isdigit()) and (i2 == '-1' or i2.isdigit()) and (j1 == '-1' or j1.isdigit()) and (j2 == '-1' or j2.isdigit()))
                
                inter_sentence.append(relation)
            else:
                relation, start, end, arg1_start, arg1_end, arg2_start, arg2_end, sentence = self.tab_re.split(line)
                if int(arg1_start) != -1:
                    connective, connective_start = getConnective(int(start), int(arg1_start),
                                                                 int(arg1_end), int(arg2_start),
                                                                 int(arg2_end), sentence)
                    intra_sentence[start].append((relation, connective, connective_start))
                    
        return inter_sentence, [intra_sentence[i] for i in sorted(intra_sentence.keys())]
    
    def parse(self, document):
        infile = tempfile.NamedTemporaryFile('/tmp/discourse_input')
        infile.write(document)
        outfile = '/tmp/discourse_output'
        
        p = subprocess.Popen(self.command.split() + [infile.name, outfile])
        out, err = p.communicate()

        with open(outfile) as f:
            discourse_data = f.read()

        return self.processDiscourse(self, discourse_data, len(document.split('\n')))
