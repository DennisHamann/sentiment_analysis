# Firstly, please note that the performance of google word2vec is better on big datasets.
# In this example we are considering only 25000 training examples from the imdb dataset.
# Therefore, the performance is similar to the "bag of words" model.

# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup
import re  # For regular expressions
from gensim.models import word2vec
import nltk
import logging
import pickle
# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
import os
import sys
import errno
# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.
import csv 
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english"))

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


# This function converts a text to a sequence of words.
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


# This function splits a review into sentences
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


# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0
    list = []
    # Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            list.append(model[word])

    return list


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    listoflist = [] # Maybe as np array np.zeros((len(reviews), num_features), dtype="float32")
    for review in list(reviews):
        # Printing a status message every 1000th review
        if counter % 10000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        listoflist.append(featureVecMethod(review, model, num_features))
        counter = counter + 1

    return listoflist


if __name__ == '__main__':
    # Read data from files

    input = sys.argv[1]
    output = sys.argv[2]
    mkdir_p(sys.argv[2])
    path_train = os.path.join(input, 'Train.csv')
    path_test = os.path.join(input, 'Test.csv')
    train = pd.read_csv('data/prepared/Train.csv')
    test = pd.read_csv('data/prepared/Test.csv')

    sentences = []
    print("Parsing sentences from training set")
    for review in train["text"]:
        sentences += review_sentences(review, tokenizer)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    model_name = f"{num_features}features_40minwords_10context"
    model_path = os.path.join(output, model_name)
    model.save(model_path)
    print(model.wv.most_similar("crime"))
    print(model.wv.most_similar("beer"))
    print(model.wv.most_similar("math"))
    # This will give the total number of words in the vocabolary created from this dataset
    model.wv.syn0.shape

    # Save Labels
    df=pd.read_csv('data/prepared/Train.csv')
    classes = list(df['label'])
    df_path = os.path.join(output, f'{num_features}trainDatalabel.pickle')
    with open(df_path, 'wb') as f:
        pickle.dump(classes, f)
        
    df=pd.read_csv('data/prepared/Test.csv')
    classes = list(df['label'])
    df_path = os.path.join(output, f'{num_features}testDatalabel.pickle')
    with open(df_path, 'wb') as f:
        pickle.dump(classes, f)
      
    # Calculating average feature vector for training set
    clean_train_reviews = []
    for review in train['text']:
        clean_train_reviews.append(review_wordlist(review, remove_stopwords=True))

    trainDataVecs=getAvgFeatureVecs(clean_train_reviews, model, num_features)
    train_path = os.path.join(output, f'{num_features}trainDataVec.csv')
    print(type(trainDataVecs))
    counter = 0
    with open(train_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in tqdm(trainDataVecs):
                          
            writer.writerow(line)
          

    # Calculating average feature vactors for test set
    clean_test_reviews = []
    for review in test["text"]:
        clean_test_reviews.append(review_wordlist(review, remove_stopwords=True))

    testDataVecs=getAvgFeatureVecs(clean_test_reviews, model, num_features)
    test_path = os.path.join(output, f'{num_features}testDataVec.csv')
    with open(test_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in testDataVecs:
            writer.writerow(line)
    



