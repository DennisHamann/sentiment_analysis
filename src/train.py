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
    model.fit(trainX, trainY, epochs=1, batch_size=128, verbose=1, validation_data=(testX,testY))
    return model

def get_model(input_shape,output_shape):
    #set variables --- these may change when switching to the new dataset
    embedding_vector_length=32
    max_review_length=500
    top_words=5000
    model = Sequential()
    #add embedding layer for word to vec
    model.add(Embedding(top_words, embedding_vector_length,input_length=max_review_length))
    #add LSTM layer
    model.add(LSTM(64,input_shape=input_shape))
    #prevent overfitting
    model.add(Dropout(0.5))
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

    top_words=5000
    (trainX,trainY),(testX,testY) =  imdb.load_data(num_words=top_words)

    #bring datasets to the same lenght

    review_length=500
    trainX=sequence.pad_sequences(trainX,maxlen=review_length)
    testX=sequence.pad_sequences(testX,maxlen=review_length)

    #get model

    model = get_model(trainX.shape,1)

    #train model

    model = train_model(model,trainX,trainY,testX,testY)

    #save model to output folder

    model.save('writepath')


