Code for training models on social media for influence prediction

## Dependencies:
### python
We recommend using conda with the included requirements.txt file:
conda create -n new_environment -f requirements.txt

### other (optional)
- SEMAFOR - http://www.cs.cmu.edu/~ark/SEMAFOR/
- Discourse Tagger - http://www.cs.columbia.edu/~orb/code_data/2Taggers.zip

## Running the code
To run experiments, first preprocess the data: </br>
preprocess_cmv.py trainfile testfile --embeddings embeddings_file

Then you can run experiments with linear models: </br>
train_cmv_lr.py metadata_file

Or you can run experiments with the RNN:  </br>
cmv_predict_rnn.py metadata_file output_file
