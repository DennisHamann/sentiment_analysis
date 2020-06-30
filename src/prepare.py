import io
import sys
import xml.etree.ElementTree
import random
import re
import os
import errno
import zipfile
import logging
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup
import re  # For regular expressions
from gensim.models import word2vec
import nltk
# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.
import csv 
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english"))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def review_wordlist(review, remove_stopwords=False):
    # 1. Removing html tags
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 2. Removing non-letter.
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        words = [w for w in words if not w in stops]

    return (words)
    
def review_sentences(review, tokenizer, remove_stopwords=False):
    # 1. Using nltk tokenizer
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_wordlist(raw_sentence, \
                                             remove_stopwords))

    # This returns the list of lists
    return sentences


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython prepare.py data\n')
        sys.exit(1)

    # Maybe implement other ratios if necessary, Test data set split ratio
    split = 0.20
    random.seed(20170426)

    input = sys.argv[1]
    output = sys.argv[2]

    input_train = os.path.join(input, 'trainset.zip')
    input_test = os.path.join(input, 'testset.zip')
    print(input_train, input_test)
    
    zf = zipfile.ZipFile(input_train, 'r')
    zf.extractall(output)
    zf.close()
    zf = zipfile.ZipFile(input_test, 'r')
    zf.extractall(output)
    zf.close()
    
    path_train = os.path.join(output, 'Train.csv')
    path_test = os.path.join(output, 'Test.csv')
    train = pd.read_csv('data/prepared/Train.csv')
    test = pd.read_csv('data/prepared/Test.csv')
    
    sentences = []
    print("Parsing sentences from training set")
    for review in train["text"]:
        sentences += review_sentences(review, tokenizer)

   
    # Creating the model and setting values for the various parameters, To do: finetuning
    num_features = 80  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of parallel threads
    context = 10  # Context window size
    downsampling = 1e-3  # (0.001) Downsample setting for frequent words

    # Initializing the train model

    print("Training model....")
    model = word2vec.Word2Vec(sentences, \
                              workers=num_workers, \
                              size=num_features, \
                              min_count=min_word_count, \
                              window=context,
                              sample=downsampling)

    # To make the model memory efficient
    model.init_sims(replace=True)

    # Saving the model for later use. Can be loaded using Word2Vec.load()
    model_name = f"{num_features}features_model"
    model_path = os.path.join(output, model_name)
    model.save(model_path)




