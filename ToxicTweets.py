# Toxic Tweets Exploratory Data Analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')

# file name
data = "dataset_for_detection_of_cybertrolls.json"


# Reading the JSON file
def load_data(data):
    tweets = pd.read_json(data, lines=True)
    tweets['label'] = tweets.annotation.apply(lambda x: x.get('label'))
    tweets['label'] = tweets['label'].apply(lambda x: x[0])
    X = tweets['content'].values
    Y = tweets['label'].values
    return X, Y


# Cleaning tweets
def clean_tweets(tweets):
    tweets = nltk.word_tokenize(tweets)
    corpus = []
    for tweet in tweets:
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)
        tweet = tweet.lower()
        tweet = tweet.strip()
        lemmatizer = nltk.WordNetLemmatizer()
        tweet = lemmatizer.lemmatize(tweet)
        corpus.append(tweet)
    return corpus


vector = CountVectorizer(tokenizer=clean_tweets)
tfidf = TfidfTransformer()
rf = RandomForestClassifier()
# convert to a sparse matrix

# Train test split
X, Y = load_data(data)
x_train, x_test, y_train, y_test = train_test_split(X, Y)

# bag of words
x_train_vector = vector.fit_transform(x_train)
x_train_tfidf = tfidf.fit_transform(x_train_vector)
# Output is the tf-idf weights (higher the tf-idf score, rarer the term)
rf.fit(x_train_tfidf, y_train)

# predict on test data
x_test_vector = vector.transform(x_test)
x_test_tfidf = tfidf.transform(x_test_vector)
y_pred = rf.predict(x_test_tfidf)

# performance evaluation
confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred, pos_label='1'))
print(precision_score(y_test, y_pred, pos_label='1'))
print(f1_score(y_test, y_pred, pos_label='1'))
