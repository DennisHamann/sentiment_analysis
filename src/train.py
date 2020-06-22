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
    model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=1, validation_data=(testX,testY))
    return model

def get_model(input_shape,output_shape):
    #set variables --- these may change when switching to the new dataset
    ##embedding_vector_length=32
    #max_review_length=500
    #top_words=5000
    model = Sequential()
    #add embedding layer for word to vec
    #model.add(Embedding(top_words, embedding_vector_length,input_length=max_review_length))
    #add LSTM layer
    model.add(LSTM(128,input_shape=input_shape))
    #prevent overfitting
    model.add(Dropout(0.9))
    #binary output - maybo more?
    model.add(Dense(output_shape, activation='sigmoid'))
    #define optimizer, metric
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    #train model with given dataset
    return model

#with open(output, 'w') as f:
#    f.write('This is a trained model')


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

    #get datasets
    trainXfile = open('data/features/10trainDataVec.pickle','rb')
    trainYfile = open('data/features/10trainDatalabel.pickle','rb')
    testXfile  = open('data/features/10testDataVec.pickle','rb')
    testYfile  = open('data/features/10testDatalabel.pickle','rb')

    trainX = pickle.load(trainXfile)
    trainY = pickle.load(trainYfile)
    testX  = pickle.load(testXfile)
    testY  = pickle.load(testYfile)

    #this has do be removed once the data is complete

    trainY=np.array(trainY[0:50])
    testY=np.array(testY[0:50])

    train_max_len = len(max(trainX,key=len))
    test_max_len = len(max(testX,key=len))

    max_len = max(train_max_len,test_max_len)

    trainX=sequence.pad_sequences(trainX,maxlen=max_len,padding='pre',dtype="f")
    testX=sequence.pad_sequences(testX,maxlen=max_len,padding='pre',dtype="f")


    #top_words=5000
    #(trainX,trainY),(testX,testY) =  imdb.load_data(num_words=top_words)



    #bring datasets to the same lenght

    #review_length=500
    ##trainX=sequence.pad_sequences(trainX,maxlen=review_length)
    #testX=sequence.pad_sequences(testX,maxlen=review_length)

    #get model
    print("definiing model...")
    model = get_model(trainX.shape[1:3],1)

    #train model
    print("training model...")
    model = train_model(model,trainX,trainY,testX,testY)

    #save model to output folder

    model.save(writepath)


