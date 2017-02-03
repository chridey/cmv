import csv
import sys

from sklearn.metrics import accuracy_score

NUM_JUDGMENTS = '_trusted_judgments'
A_PREDICTION = 'do_you_think_the_original_poster_would_find_argument_a_convincing'
B_PREDICTION = 'do_you_think_the_original_poster_would_find_argument_b_convincing'
RANK_PREDICTION = 'please_indicate_which_argument_you_found_more_convincing_regardless_of_whether_you_found_either_argument_to_be_convincing_or_not'
RANK_PREDICTION2 = 'please_indicate_which_argument_you_think_the_original_poster_found_more_convincing_regardless_of_whether_you_thought_they_would_have_found_either_argument_to_be_convincing_or_not'
A_LABEL = 'label1'
B_LABEL = 'label2'

filename = sys.argv[1]

predictions = []
pairwise_predictions = []
labels = []
pairwise_labels = []

with open(filename) as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        rank = RANK_PREDICTION
        if not row[rank]:
            rank = RANK_PREDICTION2
        print(row[NUM_JUDGMENTS], row[A_PREDICTION], row[B_PREDICTION],
              row[rank], row[A_LABEL], row[B_LABEL])
        if int(row[NUM_JUDGMENTS]) < 3:
            continue
        predictions.append(row[A_PREDICTION] == 'yes')
        labels.append(row[A_LABEL] == '1')
        predictions.append(row[B_PREDICTION] == 'yes')
        labels.append(row[B_LABEL] == '1')
        
        pairwise_predictions.append(row[rank] in ('equally_convincing', 'a'))
        pairwise_labels.append(row[A_LABEL] == '1')

print(len(predictions), sum(predictions))
print(len(pairwise_predictions), sum(pairwise_predictions))
print(len(labels), sum(labels))
print(len(pairwise_labels), sum(pairwise_labels))
        
print(accuracy_score(labels, predictions))
print(accuracy_score(pairwise_labels, pairwise_predictions))
                                    
