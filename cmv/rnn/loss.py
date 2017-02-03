#custom loss functions
import theano.tensor as T

def margin_loss(weights, pos, neg, delta=1, normalize=True):
    if normalize:
        pos /= pos.norm(2, axis=1)[:, None]
        neg /= neg.norm(2, axis=1)[:, None]

    #now B-long vector
    correct = T.sum(weights[None, :] * pos, axis=1)
    incorrect = T.sum(weights[None, :] * neg, axis=1)
    
    return T.sum(T.maximum(0., 1-correct+incorrect)).mean()
