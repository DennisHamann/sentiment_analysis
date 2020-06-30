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
from os import path
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


def data2memmap(file,mmap,no_of_reviews,maxlen,vecsize):
    mmap = os.path.join('data/features/',mmap)
    data = np.memmap(mmap, dtype='float', mode='w+', shape=(no_of_reviews, maxlen, vecsize))
    for (idx, row) in enumerate(file):
        review_length = len(row)
        review_length = min(review_length,maxlen)
        review = list()
        for i in range(review_length-1):
            wordtmp = row[i].replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ]', '').replace(']', '').replace('[ ', '').replace('[', '').replace('\n', '')
            wordtmp = wordtmp.split(' ')
            review.append(np.array(wordtmp).astype('float'))
        review = np.array(review)
        data[idx, (maxlen - review.shape[0]):maxlen, :] = review
    data.flush()


def get_shape(file):
    maxlen = 0
    for (idx, row) in enumerate(file):
        i = 0
        for word in row:
            i += 1
        if i > maxlen:
            maxlen = i
    no_of_reviews = idx+1
    return no_of_reviews, maxlen


if __name__ == '__main__':
    # Read data from files

    input = sys.argv[1]
    output = sys.argv[2]
    mkdir_p(sys.argv[2])
    path_train = os.path.join(input, 'Train.csv')
    path_test = os.path.join(input, 'Test.csv')
    train = pd.read_csv('data/prepared/Train.csv')
    test = pd.read_csv('data/prepared/Test.csv')
    # Creating the model and setting values for the various parameters, To do: finetuning
    num_features = 80  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of parallel threads
    context = 10  # Context window size
    downsampling = 1e-3  # (0.001) Downsample setting for frequent words
    model_path = os.path.join(input, f"{num_features}features_model")
    if path.exists(model_path):
        model = word2vec.Word2Vec.load(os.path.join(input, f"{num_features}features_model"))
    else:
        print('Error: model not found')
        '''
        sentences = []
        print("Parsing sentences from training set")
        for review in train["text"]:
            sentences += review_sentences(review, tokenizer)

             

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
        '''
    # This will give the total number of words in the vocabolary created from this dataset
    model.wv.syn0.shape

    # Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)

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
    train_path = os.path.join(output, f'{num_features}trainDataVec.csv')
    with open(train_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for review in tqdm(train['text']):
            clean_train_reviews=review_wordlist(review, remove_stopwords=True)
            trainDataVecs=featureVecMethod(clean_train_reviews, model, num_features)
            writer.writerow(trainDataVecs)
          
    # Calculating average feature vector for test set
    clean_test_reviews = []
    test_path = os.path.join(output, f'{num_features}testDataVec.csv')
    with open(test_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for review in tqdm(test['text']):
            clean_test_reviews=review_wordlist(review, remove_stopwords=True)
            testDataVecs=featureVecMethod(clean_test_reviews, model, num_features)
            writer.writerow(testDataVecs)

    # initializing the memory maps
    print('creating memory map...')
    trainXfile = csv.reader(open('data/features/80trainDataVec.csv', 'rt'))
    testXfile = csv.reader(open('data/features/80testDataVec.csv', 'rt'))
    no_of_reviews_train, maxlen_train = get_shape(trainXfile)
    no_of_reviews_test, maxlen_test = get_shape(testXfile)
    shape = np.array([maxlen_train,no_of_reviews_train,maxlen_test,no_of_reviews_test])
    np.save('data/features/shape', shape)
    data2memmap(trainXfile, 'trainmapX', no_of_reviews_train, maxlen_train,vecsize)
    data2memmap(testXfile, 'testmapX', no_of_reviews_test, maxlen_test,vecsize)

    



