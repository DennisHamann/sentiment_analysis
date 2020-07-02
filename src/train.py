import sys
import os
import errno

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing import sequence
import pickle
import csv
import random



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def train_model(model,trainX,trainY,testX,testY,batch_size,no_of_reviews_train,no_of_reviews_test,maxlen,vecsize):
    model.fit(generator(trainX,trainY,batch_size,vecsize,no_of_reviews_train,maxlen), epochs=5, steps_per_epoch=int(no_of_reviews_train/batch_size), verbose=1, validation_data=generator(testX,testY,batch_size,vecsize,no_of_reviews_test,maxlen), validation_steps=int(no_of_reviews_train/batch_size))
    return model

def get_model(input_shape,output_shape):
    model = Sequential()
    #add LSTM layer
    model.add(LSTM(256,input_shape=input_shape))
    #prevent overfitting
    model.add(Dropout(0.1))
    #binary output - maybo more?
    model.add(Dense(output_shape, activation='sigmoid'))
    #define optimizer, metric
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize):
    trainX = np.memmap('data/features/trainmapX', dtype='float', mode='r',shape=(no_of_reviews_train,maxlen,vecsize))
    testX = np.memmap('data/features/testmapX', dtype='float', mode='r',shape=(no_of_reviews_test,maxlen,vecsize))
    trainYfile = open('data/features/80trainDatalabel.pickle', 'rb')
    testYfile = open('data/features/80testDatalabel.pickle', 'rb')
    trainY = pickle.load(trainYfile)
    testY = pickle.load(testYfile)
    return trainX, testX, trainY, testY

def data2memmap(file,mmap,no_of_reviews,maxlen,vecsize):
    mmap = os.path.join('data/features/',mmap)
    data = np.memmap(mmap, dtype='float', mode='w+', shape=(no_of_reviews, maxlen, vecsize))
    print('mmap=0')
    print('start iterator')
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


def generator(X,Y,batch_size,vecsize,no_of_reviews,maxlen):
    batchX = np.zeros((batch_size,maxlen,vecsize))
    batchY = np.zeros((batch_size))
    index=np.array(range(no_of_reviews-1))
    random.shuffle(index)
    Y=np.array(Y)
    while True:
        for i in range(int((no_of_reviews-1)/batch_size)):
            batchX[:,:,:] = X[index[i*batch_size:(i+1)*batch_size],:,:]
            batchY[:] = Y[index[i*batch_size:(i+1)*batch_size]]
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
    batch_size = 400

    [maxlen,no_of_reviews_train,no_of_reviews_test,vecsize] = np.load('data/features/shape.npy')


    #get datasets
    print('loading data...')
    trainX, testX, trainY, testY = get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize)

    #get model
    print("defining model...")
    model = get_model((maxlen,vecsize), 1)
    print('training model...')
    model = train_model(model, trainX, trainY, testX, testY,batch_size,no_of_reviews_train,no_of_reviews_test,maxlen,vecsize)

    #save model to output folder
    model.save(writepath)


# python3 src/prepare.py data/dataset data/prepared
# python3 src/featurization.py data/prepared data/features
#python src/train.py data/features data/models
#
#
