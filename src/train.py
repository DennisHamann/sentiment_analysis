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
    model.fit(generator(trainX,trainY,25), epochs=5, steps_per_epoch=100, verbose=1, validation_data=generator(testX,testY,25), validation_steps=50)
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

def get_data(no_of_reviews_train,no_of_reviews_test,maxlen):
    trainX = np.memmap('data/features/trainmapX', dtype='float', mode='r',shape=(no_of_reviews_train,maxlen,80))
    testX = np.memmap('data/features/testmapX', dtype='float', mode='r',shape=(no_of_reviews_test,maxlen,80))

    #does not work with pickle and might have to be changed depending on the given data
    trainYfile = open('data/features/80trainDatalabel.pickle', 'rb')
    testYfile = open('data/features/80testDatalabel.pickle', 'rb')
    trainY = pickle.load(trainYfile)
    testY = pickle.load(testYfile)
    return trainX, testX, trainY, testY

#python src/train.py data/features data/models
def data2memmap(file,mmap,no_of_reviews,maxlen):
    mmap = os.path.join('data/features/',mmap)
    data = np.memmap(mmap, dtype='float', mode='w+', shape=(no_of_reviews, maxlen, 80))
    print('mmap=0')
    print('start iterator')
    for (idx, row) in enumerate(file):
        review_length = len(row)
        review_length = min(review_length,maxlen)
        review = list()
        print(idx)
        for i in range(review_length-1):
            wordtmp = row[i].replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ]', '').replace(']', '').replace('[ ', '').replace('[', '').replace('\n', '')
            wordtmp = wordtmp.split(' ')
            review.append(np.array(wordtmp).astype('float'))
        review = np.array(review)
        data[idx, (maxlen - review.shape[0]):maxlen, :] = review
    data.flush()

def generator(X,Y,batch_size):
    batchX = np.zeros((batch_size,X.shape[1],80))
    batchY = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
            idx = random.choice(range(X.shape[0]-1))
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
    no_of_reviews_train = 40000
    no_of_reviews_test = 5000
    maxlen = 1000

    #initializing the memory maps
    #this might be transfered to featurization
    #print('creating memory map...')
    #trainXfile = csv.reader(open('data/features/80trainDataVec.csv', 'rt'))
    #testXfile = csv.reader(open('data/features/80testDataVec.csv', 'rt'))
    #data2memmap(trainXfile, 'trainmapX', no_of_reviews_train, maxlen)
    #data2memmap(testXfile, 'testmapX', no_of_reviews_test, maxlen)

    #get datasets
    print('loading data...')
    trainX, testX, trainY, testY = get_data(no_of_reviews_train,no_of_reviews_test,maxlen)

    #get model
    print("defining model...")
    model = get_model((maxlen,80), 1)
    #needs to be modified for memmaps
    print('training model...')
    model = train_model(model, trainX, trainY, testX, testY)

    #save model to output folder
    model.save(writepath)


