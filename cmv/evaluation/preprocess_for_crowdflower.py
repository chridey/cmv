#load data
#only remove "change my view or cmv or edit"
#columns: title,op,response1,label1,response2,label2

import random
import csv
import re

from loadData import load_train_pairs,load_test_pairs

cmv_pattern = re.compile('cmv:?', re.IGNORECASE)

from preprocess import normalize_from_body, preprocess, remove_special_token

def cleanup(text, op=False):

    cleaned_text = remove_special_token(normalize_from_body(text, lower=False))
    cleaned_text = cmv_pattern.sub('', cleaned_text)
    cleaned_text = cleaned_text.replace('\t', ' ')
    
    return cleaned_text

def get_columns(data):
    ret_data = []
    for pair in data:
        title = cleanup(pair['op_title'])
        op_text = cleanup(pair['op_text'])

        neg_text = ''
        for index,comment in enumerate(pair['negative']['comments']):
            if index > 0:
                neg_text += '\n---------------\n'
            neg_text += cleanup(comment['body'])
            
        pos_text = ''
        for index,comment in enumerate(pair['positive']['comments']):
            if index > 0:
                pos_text += '\n---------------\n'
            pos_text += cleanup(comment['body'])

        if random.random() > 0.5:
            response1 = pos_text
            label1 = 1
            response2 = neg_text
            label2 = 0
        else:
            response2 = pos_text
            label2 = 1
            response1 = neg_text
            label1 = 0
            
        ret_data.append([title,op_text,response1,str(label1),response2,str(label2)])
        
    return ret_data

if __name__ == '__main__':
    test = load_test_pairs()

    test_formatted = get_columns(test)
    print(len(test_formatted))
    
    with open('test.csv', 'w') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['title', 'op_text', 'response1', 'label1', 'response2', 'label2'])
        for line in test_formatted:
            writer.writerow([i.encode('utf-8') for i in line])
