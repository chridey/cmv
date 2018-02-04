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
preprocess_cmv.py trainfile testfile --save_metadata metadata_file --indices indices_file

Then you can run experiments with linear models: </br>
cmv_predict.py metadata_file

Or you can run experiments with the RNN:  </br>
cmv_predict_rnn.py indices_file output_file
