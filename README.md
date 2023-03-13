# Natural-Language-Processing-with-Disaster-Tweets
Twitter_disaster
NLP, text classification

Dataset
Number of train samples: 7613
Number of test samples: 3263
The task is binary class classificatoin, and our goal is to train a model that can read the tweets and distinguish real disasters from false. Please find more information on Kaggle: https://www.kaggle.com/competitions/nlp-getting-started

EDA
The labels are fairly balanced.
Most of the tweets are between 20 to 30 words.
There are stopwords in our samples, and the most common ones are: the, a, to ...etc.
There are punctuation, emoji and URLs in our samples.
Preprocessing
I removed URLS, stopwords and puncuation from our samples.
I left emojis because they might be meaningful in certain ways.
I used GloVe 6B-50d for encoding our text data.
Model structure
I trained three models for comparison:

SimpleRNN
LSTM
GRU
Each model has similiar structure: embedding layer, RNN layer and finally dense layer for classification.

Results
LSTM and GRU had similar performance, and they outperformed the SimpleRNN model. Our models learned sufficiently after the first few epochs but encountered overfitting issues as the training continued. For the detailed informatoin, please check the graph below.

Discussions
We can possibly avoid or alleviate the overfitting issues by adding more layers and parameters into the models.
We can possibly avoid or alleviate the overfitting issues by adding dropouts, batch normalization or regulation layers.
Pre-trained models specifically trained for medical images should have much beter results than the naive model in this notebook. Unfortunately, under the scope of this project, I did not investigate those possibilities.
