from __future__ import print_function

import sys
import json
import itertools
import collections
import copy
import re
import os
import bz2

from spacy.en import English

from cmv.preprocessing.preprocess import normalize_from_body

nlp = English()

cmv_pattern = re.compile('cmv:?', re.IGNORECASE)

def cleanup(text, op=False):

    cleaned_text = normalize_from_body(text, op=op, lower=False)
    cleaned_text = cmv_pattern.sub('', cleaned_text)
    
    parsed_text = [nlp(unicode(i)) for i in cleaned_text.split('\n')]
    return parsed_text

#go through each thread
#first determine all the root replies by building a list of children and checking against that
#build forest of replies
#also determine the deltas awarded by checking the user name DeltaBot and retracing steps up each tree, assign a delta to each tree (i dont think there can or should be more than one per tree)

def build_comment_forest(data):
    #the comment forest, for each node index, a list of children
    #initially the list contains the child IDs but after the second pass, they contain the node index
    comment_forest = collections.defaultdict(list)
    comment_id_lookup = {}
    
    #go through each comment, storing the lookup id
    for i,node in enumerate(data['comments']):
        comment_id_lookup[node['id']] = i
        
    #now go through, adding its children to the comment_forest
    for i,node in enumerate(data['comments']):
        if 'replies' in node and node['replies'] is not None and 'data' in node['replies']:
            for child in node['replies']['data']['children']:
                comment_forest[i].append(comment_id_lookup[child])

    #add the root replies to the tree (its a tree now)
    children = set(itertools.chain(*(comment_forest.values())))
    comment_forest[-1] = [i for i in range(len(data['comments'])) if i not in children]
    
    return comment_forest

def backtrack(stack):
    stack[-1].pop()
    #backtrack if we have exhausted the nodes at this level, until we get to any new nodes
    while len(stack) and not len(stack[-1]):
        stack.pop()
        #need to pop this too because it means we are done with this node
        if len(stack) and len(stack[-1]):
            stack[-1].pop()
            
        #print(3, stack)
        
    return stack


def delta_dfs(op, comment_forest, comments, root=-1):
    #do a DFS on the comment forest until we find a delta bot
    #once there, backtrack to the comment right before the OP gives a delta (and make sure its the OP giving the delta), keep everything from the OP and responder (only?) in this path

    if not len(comment_forest[root]):
        return []
    
    #print(comment_forest)
    stack = [list(comment_forest[root])]
    delta_paths = []
    
    while len(stack):
        #print(1, stack)
        #check if the top node on the stack is a deltabot
        #if not, add the next node
        node = stack[-1][-1]

        #print(node)        
        if 'author' in comments[node] and comments[node]['author'] == 'DeltaBot' and 'delta awarded' in comments[node]['body']:
            assert('delta awarded to' in comments[node]['body'])
            
            #backtrack by 1, check if the author of the post is the OP (delta given by op)
            #also check if the delta is given to the author of the root response

            '''
            start = comments[node]['body'].find('/u/')
            end = comments[node]['body'].find('.')
            if end != -1:
                recipient = comments[node]['body'][start+len('/u/'):end]
            else:
                recipient = comments[node]['body'][start+len('/u/'):].strip()
            '''

            start = comments[node]['body'].find('delta awarded to ') + len('delta awarded to ')
            end = comments[node]['body'][start:].find('.')
            if end == -1:
                end = comments[node]['body'][start:].find(' ')
            else:
                next_space = comments[node]['body'][start:].find(' ')
                if next_space != -1:
                    end = min(end, next_space)
                          
            if end != -1:
                recipient = comments[node]['body'][start:start+end]
            else:
                recipient = comments[node]['body'][start:]

            '''
            recipient = comments[node]['body'][start:start+end].replace('/u/', '').strip()
    
            if len(recipient) and recipient[-1] == '.':
                recipient = recipient[:-1]
            '''
            recipient = recipient.replace('/u/', '').strip()
             
            #check for the case where this node has no parent for some reason
            if len(stack) > 1:
                stack.pop()            
                node = stack[-1][-1]

                print(recipient, comments[stack[0][-1]]['author'], comments[node]['author'], op)            
                if comments[node]['author'] == op and comments[stack[0][-1]]['author'] == recipient:
                    delta_paths.append([i[-1] for i in stack])
                #we dont care about any other children of the post where OP gives the delta, so we can continue from here
        #check if the current node has any children, otherwise pop the stack
        elif node in comment_forest:
            stack.append(list(comment_forest[node]))
            continue

        stack = backtrack(stack)

    return delta_paths
            
def response_dfs(op, comment_forest, comments, deltas=None, root=-1):
    #do a DFS for negative responses, but only on the trees where the OP replies directly to the root response
    #if that holds, keep everything in the path, as long as the last comment is from the OP or RR poster
    #basically just continue DFS down the path if a comment is either from OP or RR, terminate otherwise

    '''
    root_replies = []
    for rr in comment_forest[-1]:
        if rr in deltas:
            continue
        for child in comment_forest[rr]:
            if 'author' in comments[child] and comments[child]['author'] == op:
                root_replies.append([rr])                
                break
            
    return root_replies
    '''
    
    #print(comment_forest)
    responses = [i for i in comment_forest[root] if i not in deltas and 'author' in comments[i]]
    if not len(responses):
        return []
    
    stack = [responses]
    root_replies = []
    
    while len(stack):
        
        #print(1, stack)
        #check if the top node on the stack is OP or RR
        #if not, add current path and backtrack
        node = stack[-1][-1]

        #print(node)
        author = comments[stack[0][-1]]['author']
        valid_children = []
        for child in comment_forest[node]:
            if 'author' in comments[child] and comments[child]['author'] in (op, author):
                valid_children.append(child)
        if len(valid_children):
            stack.append(valid_children)
            continue

        #otherwise weve reached the end of an OP-RR interaction
        #so add the whole path to the root_replies unless this RR is already in root_replies
        if stack[0][-1] not in {i[0] for i in root_replies}:
            #make sure at least one response is OP and one is RR
            path = [i[-1] for i in stack]
            authors = set(comments[i]['author'] for i in path)
            #print(path, authors)
            if len(authors) == 2 and authors == set((op, author)):
                #print(path)
                root_replies.append(path)
            
        stack = backtrack(stack)

    return root_replies


if __name__ == '__main__':            
    total_responses = 0
    total_responses_ge50 = 0
    total_responses_ge50_min10 = 0
    total_responses_ge50_delta = 0
    total_responses_delta = 0    
    total_deltas = 0
    total_deltas_rr = 0
    total_deltas_ge50_min10 = 0
    thread_count = 0
    total_threads_uniq_delta = 0
    total_threads_uniq_delta_rr = 0    
    total_pairs = 0
    neg_counts = collections.defaultdict(int)
    pos_counts = collections.defaultdict(int)

    infile = sys.argv[1]
    outfile = sys.argv[2]
    if len(sys.argv) > 3:
        start = int(sys.argv[3])
        end = int(sys.argv[4])
    else:
        start = 0
        end = float('inf')
        
    pairs = []
    xml = dict(positive=[], negative=[])
    
    with open(infile) as f:
        for line in f:
            thread_count += 1
            if thread_count-1 < start:
                continue
            if thread_count-1 > end:
                break
            
            data = json.loads(line)
            comment_forest = build_comment_forest(data)

            #deltas = [i for i,j in enumerate(data['comments']) if 'author' in j and j['author'] == 'DeltaBot' and 'delta awarded' in j['body']]
            print('old', len([i for i,j in enumerate(data['comments']) if 'author' in j and j['author'] == 'DeltaBot' and 'delta awarded' in j['body']]))
            deltas = delta_dfs(data['author'], comment_forest, data['comments'])

            root_replies = response_dfs(data['author'], comment_forest, data['comments'],
                                        set([i[0] for i in deltas]))
            #root_replies = comment_forest[-1]

            root_replies_min50 = [i[0] for i in root_replies if 'body' in data['comments'][i[0]] and len(data['comments'][i[0]]['body'].split()) >= 50]
            
            total_responses += len(root_replies)
            total_deltas += len(deltas)
            total_responses_ge50 += len(root_replies_min50)
            if len(deltas):
                total_responses_ge50_delta += len(root_replies_min50)
                total_responses_delta += len(root_replies)
                total_threads_uniq_delta += 1
                pos_counts[len(deltas)] += 1
                neg_counts[len(root_replies)] += 1
                total_pairs += len(deltas)*len(root_replies)
                if len(root_replies):
                    total_threads_uniq_delta_rr += 1
                    
                    pair = dict(op_text=data['selftext'],
                                op_title=data['title'],
                                op_author=data['author'],
                                op_name=data['name'])
                    
                    pair['positive'] = {}
                    pair['positive']['comments'] = []

                    #id, title, url, author, selftext
                    xml_base = '<?xml version="1.0"?>'
                    xml_base += '<thread ID="{}">'.format(data['id'])
                    xml_base += '<title>{}</title>'.format(data['title'].encode('utf-8'))
                    xml_base += '<source>{}</source>'.format(data['url'].encode('utf-8'))
                    xml_base += '<OP author="{}">\n{}\n</OP>'.format(data['author'].encode('utf-8'),
                                                                     data['selftext'].encode('utf-8'))
                    
                    for delta_path in deltas:
                        #body, author, id
                        xml_discussion = xml_base
                        assert(data['comments'][delta_path[0]]['parent_id'] == data['name'])
                        
                        #get posts with just the root reply poster
                        text = ''
                        for post_index in delta_path:
                            xml_discussion += '<reply id="{}" author="{}"{}'.format(data['comments'][post_index]['id'],
                                                                                    data['comments'][post_index]['author'],
                                                                                    data['comments'][post_index]['body'].encode('utf-8'))
                            if data['comments'][post_index]['author'] == data['comments'][delta_path[0]]['author']:
                                text += data['comments'][post_index]['body']
                                text += '\nINTERMEDIATE_DISCUSSION\n'

                        data_copy = copy.deepcopy(data)
                        data_copy['comments'][delta_path[0]]['body'] = text
                        
                        pair['positive']['comments'].append(data_copy['comments'][delta_path[0]])
                        xml['positive'].append(xml_discussion + '</reply>'*len(delta_path)+'</thread>')
                        
                    pair['negative'] = {}
                    pair['negative']['comments'] = []
                    for root_reply in root_replies:
                        #print(root_reply)
                        #print(comment_forest[-1])
                        #print(data['author'], data['comments'][root_reply[0]]['author'])
                        xml_discussion = xml_base
                        assert(data['comments'][root_reply[0]]['parent_id'] == data['name'])

                        #get posts with just the root reply poster
                        text = ''
                        for post_index in root_reply:
                            xml_discussion += '<reply id="{}" author="{}"{}'.format(data['comments'][post_index]['id'],
                                                                                    data['comments'][post_index]['author'],
                                                                                    data['comments'][post_index]['body'].encode('utf-8'))
                            if data['comments'][post_index]['author'] == data['comments'][root_reply[0]]['author']:
                                text += data['comments'][post_index]['body']
                                text += '\nINTERMEDIATE_DISCUSSION\n'

                        data_copy = copy.deepcopy(data)
                        data_copy['comments'][root_reply[0]]['body'] = text
                        
                        pair['negative']['comments'].append(data_copy['comments'][root_reply[0]])
                        xml['negative'].append(xml_discussion + '</reply>'*len(root_reply)+'</thread>')
                        
                    pairs.append(copy.deepcopy(pair))
                            
            #if len(root_replies_min50) >= 10:
            #    total_responses_ge50_min10 += len(root_replies_min50)
            #    total_deltas_ge50_min10 += len(deltas)
            if len(root_replies):
                total_deltas_rr += len(deltas)
                
            print(thread_count-1, total_responses, total_responses_delta, total_responses_ge50, total_responses_ge50_delta, total_deltas, total_deltas_rr, total_threads_uniq_delta, total_threads_uniq_delta_rr, total_pairs)

    with bz2.BZ2File(outfile, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair))
            f.write('\n')

    xml_base_dir = 'cmv_xml'
    if not os.path.exists(xml_base_dir):
        os.mkdir(xml_base_dir)
    for key in xml:
        xml_key_dir = os.path.join(xml_base_dir, key)
        if not os.path.exists(xml_key_dir):
            os.mkdir(xml_key_dir)
        for index,thread in enumerate(xml[key]):
            with open(os.path.join(xml_key_dir, str(index) + '.xml'), 'w') as f:
                f.write(thread)
                    
    print(pos_counts)
    print(neg_counts)
