from __future__ import print_function

import sys
import json
import itertools
import collections
import copy
import re
import os
import bz2


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


def delta_dfs(op, comment_forest, comments):
    #do a DFS on the comment forest until we find a delta bot
    #once there, backtrack to the comment right before the OP gives a delta (and make sure its the OP giving the delta), keep everything from the OP and responder (only?) in this path

    if not len(comment_forest[-1]):
        return [], []
    
    #print(comment_forest)
    stack = [list(comment_forest[-1])]
    delta_bots = []
    positives = []
    
    while len(stack):
        #print(1, stack)
        #check if the top node on the stack is a deltabot
        #if not, add the next node
        node = stack[-1][-1]

        if 'author' in comments[node] and comments[node]['author'] == 'DeltaBot':            
            if len(stack) > 1:
                stack.pop()            
                positives.append(stack[-1][-1])
            delta_bots.append(node)
        elif node in comment_forest:
            stack.append(list(comment_forest[node]))
            continue

        stack = backtrack(stack)

    negatives = list(set(range(len(comments))) - set(positives) - set(delta_bots))
    
    return positives, negatives            

if __name__ == '__main__':            
    infile = sys.argv[1]
    outdir = sys.argv[2]

    total_positives = 0
    total_negatives = 0
    with open(infile) as f:
        for index,line in enumerate(f):
            data = json.loads(line)
            comment_forest = build_comment_forest(data)

            positives, negatives = delta_dfs(data['author'], comment_forest, data['comments'])

            total_positives += len(positives)
            total_negatives += len(negatives)
            print(index, len(positives), len(negatives), total_positives, total_negatives)

            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir_pos = os.path.join(outdir, 'positives')
            if not os.path.exists(outdir_pos):
                os.mkdir(outdir_pos)
            outdir_neg = os.path.join(outdir, 'negatives')
            if not os.path.exists(outdir_neg):
                os.mkdir(outdir_neg)
            pos_file = os.path.join(outdir_pos, '{}.json'.format(index))
            neg_file = os.path.join(outdir_neg, '{}.json'.format(index))

            with open(pos_file, 'w') as f:
                for pos in positives:
                    f.write(json.dumps(data['comments'][pos]))
                    f.write('\n')

            with open(neg_file, 'w') as f:
                for neg in negatives:
                    f.write(json.dumps(data['comments'][neg]))
                    f.write('\n')

