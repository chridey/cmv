import argparse

from cmv.preprocessing.loadData import load_train_pairs,load_test_pairs,handle_pairs_input

from cmv.rnn.preprocessing import build_indices
from cmv.rnn.argumentationRNN import ArgumentationRNN


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train an argument RNN')

    parser.add_argument('--lambda_w', type=float, default=0)
    parser.add_argument('--lambda_c', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_batches', type=int, default=100)
    parser.add_argument('--root_replies', type=int, default=1)
    parser.add_argument('--dimension', type=int, default=100)
    
    args = parser.parse_args()

    lambda_w = args.lambda_w
    num_epochs = args.num_epochs
    num_batches = args.num_batches
    root_replies = args.root_replies

    train_pairs = load_train_pairs()
    heldout_pairs = load_test_pairs()
    
    train_op, train_neg, train_pos = handle_pairs_input(train_pairs, root_replies)
    heldout_op, heldout_neg, heldout_pos = handle_pairs_input(heldout_pairs, root_replies)

    op_ret, resp_ret, gold_labels, op_mask, resp_mask, indices = build_indices(train_op, train_pos, train_neg, mask=True)

    op_ret_val, resp_ret_val, gold_labels_val, op_mask_val, resp_mask_val, indices = build_indices(heldout_op, heldout_pos, heldout_neg, indices=indices, mask=True)

    argRNN = ArgumentationRNN(len(indices), args.dimension)
    
    batch_size = len(op_ret) // num_batches
    for epoch in range(num_epochs):
        for batch in range(num_batches):
            s = batch_size * batch
            e = batch_size * (batch+1)
            cost = argRNN.train(op_ret[s:e],
                                resp_ret[s:e],
                                op_mask[s:e],
                                resp_mask[s:e],
                                gold_labels[s:e],
                                args.lambda_w,
                                args.lambda_c)
            print(epoch, batch, cost)
            test_cost, acc = argRNN.validate(op_ret_val,
                                        resp_ret_val,
                                        op_mask_val,
                                        resp_mask_val,
                                        gold_labels_val,
                                        args.lambda_w,
                                        args.lambda_c)
            print(test_cost, acc)
