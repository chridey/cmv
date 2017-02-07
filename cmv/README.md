Code for training models on social media for influence prediction

Python Dependencies:
spacy
nltk
numpy
scipy
pandas
theano
lasagne

Other Dependencies
SEMAFOR - http://www.cs.cmu.edu/~ark/SEMAFOR/
Discourse Tagger - http://www.cs.columbia.edu/~orb/code_data/2Taggers.zip

To run experiments, first preprocess the data:
preprocess_cmv.py trainfile testfile --save_metadata metadata_file --indices indices_file

Then you can run experiments with linear models:
cmv_predict.py metadata_file

Or you can run experiments with the RNN:
cmv_predict_rnn.py indices_file output_file
