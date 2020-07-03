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

def get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,input):
    test_vec_path = os.path.join(input, f'{vecsize}testmapX')
    train_vec_path = os.path.join(input, f'{vecsize}trainmapX')
    test_label_path = os.path.join(input, f'{vecsize}testDatalabel.pickle')
    train_label_path = os.path.join(input, f'{vecsize}trainDatalabel.pickle')
    trainX = np.memmap(train_vec_path, dtype='float', mode='r',shape=(no_of_reviews_train,maxlen,vecsize))
    testX = np.memmap(test_vec_path, dtype='float', mode='r',shape=(no_of_reviews_test,maxlen,vecsize))
    trainYfile = open(train_label_path, 'rb')
    testYfile = open(test_label_path, 'rb')
    trainY = pickle.load(trainYfile)
    testY = pickle.load(testYfile)
    return trainX, testX, trainY, testY

def generator(X,Y,batch_size,vecsize,no_of_reviews,maxlen):
    batchX = np.zeros((batch_size,maxlen,vecsize))
    batchY = np.zeros((batch_size))
    index = np.array(range(no_of_reviews - 1))
    Y=np.array(Y)
    while True:
        random.shuffle(index)
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
    writepath = os.path.join(output, 'model.h5')
    batch_size = 400
    shape_path = os.path.join(input,'shape.npy')
    [maxlen,no_of_reviews_train,no_of_reviews_test,vecsize] = np.load(shape_path)

    #get datasets
    print('loading data...')
    trainX, testX, trainY, testY = get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,input)

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
