import sys
import os
import errno

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.datasets import imdb
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pickle
import csv
import random
#import src/featurization.py
#import src/prepare.py



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def train_model(model,trainX,trainY,testX,testY):
    model.fit(generator(trainX,trainY,1), nb_epoch=10, sample_per_epoch=2, verbose=1, validation_data=generator(testX,testY,1))
    return model

def get_model(input_shape,output_shape):
    model = Sequential()
    #add LSTM layer
    model.add(LSTM(128,input_shape=input_shape))
    #prevent overfitting
    model.add(Dropout(0.9))
    #binary output - maybo more?
    model.add(Dense(output_shape, activation='sigmoid'))
    #define optimizer, metric
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def get_data(no_of_reviews,maxlen):
    trainX = np.memmap('features/trainmapX', dtype='float', mode='r',shape=(no_of_reviews,maxlen,10))
    testX = np.memmap('features/testmapX', dtype='float', mode='r',shape=(no_of_reviews,maxlen,10))

    #does not work with pickle and might have to be changed depending on the given data
    trainYfile = csv.reader(open('data/features/10trainDatalabel.pickle', 'rb'))
    testYfile = csv.reader(open('data/features/10testDatalabel.pickle', 'rb'))
    trainY = np.array(trainYfile)
    testY = np.array(testYfile)
    return trainX, testX, trainY, testY

def data2memmap(file,mmap,no_of_reviews,maxlen):
    mmap = os.path.join('features/',mmap)
    data = np.memmap(mmap, dtype='float', mode='w+', shape=(no_of_reviews, maxlen, 10))
    data[:] = 0.
    for (idx, row) in enumerate(file):
        review_length = len(row)
        review = list()
        for i in range(review_length):
            wordtmp = row[i].replace(']', '').replace('[', '')
            wordtmp = wordtmp.split(',')
            review.append(np.array(wordtmp).astype('float'))
        review = np.array(review)
        data[idx, (maxlen - review.shape[0]):maxlen, :] = review
    data.flush()

def generator(X,Y,batch_size):
    batchX = np.zeros((batch_size,X.shape[1],10))
    batchY = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
            idx = random.choice(X.shape[0],1)
            batchX[i,:,:] = X[idx,:,:]
            batchY[i] = Y[idx]
        yield batchX, batchY

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train.py features model\n')
        sys.exit(1)
    input = sys.argv[1]
    output = sys.argv[2]

    print('data path:', input)
    print('output path:', output)
    mkdir_p(sys.argv[2])
    writepath = os.path.join(sys.argv[2], 'model.h5')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write('Output')

    #needs to be changed
    no_of_reviews = 4
    maxlen = 42

    #initializing the memory maps
    #this might be transfered to featurization
    trainXfile = csv.reader(open('data/features/10trainDataVec.pickle', 'rb'))
    testXfile = csv.reader(open('data/features/10testDataVec.pickle', 'rb'))
    data2memmap(trainXfile, 'trainmapX', no_of_reviews, maxlen)
    data2memmap(testXfile, 'testmapX', no_of_reviews, maxlen)

    #get datasets
    print('loading data')
    trainX, testX, trainY, testY = get_data(no_of_reviews,maxlen)

    #get model
    print("defining model...")
    model = get_model((maxlen,10), 1)
    #needs to be modified for memmaps
    print('training model...')
    model = train_model(model, trainX, trainY, testX, testY)

    #save model to output folder
    model.save(writepath)


