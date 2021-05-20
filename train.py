# Import Library
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKerasTF

import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import f1_score
from utils.data import read_data
from utils.sentences import GetSentence
from utils.token_train import text_to_token
from utils.split_data import train_split
from model import model_lstm

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Process some intergers")
    parser.add_argument('--train_path', type=str, default="./VnDT/POS-tags-train.conll")
    parser.add_argument('--test_path', type=str, default="./VnDT/POS-tags-test.conll")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    data, words, n_words = read_data(args.train_path)
    sentences = GetSentence.sentences(GetSentence, data)

    X, Y, max_len, pos = text_to_token(data,words)
    x_train, y_train, x_test, y_test = train_split(X,Y)
    history, model = model_lstm(x_train, y_train, max_len, n_words, len(pos), args.batch_size, args.epochs)

    print("Model evaluate: ",str(model.evaluate(x_test,np.array(y_test)))) 

    #Predict
    i = np.random.randint(0,x_test.shape[0])
    p = model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)

    y_true=np.argmax(np.array(y_test),axis=-1)[i]
    print("{:15} ({:5}): {}".format("Word", "True", "Predicted by our Model"))

    print("--"*20)

    for w,true,pred in zip(x_test[i],y_true,p[0]):
        print("{:15}{}\t {}".format(words[w-1],pos[true],pos[pred]))


    # True
    i = np.random.randint(0,x_test.shape[0])
    p = model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)

    y_true=np.argmax(np.array(y_test),axis=-1)[i]
    print("{:15} ({:5}): {}".format("Word", "True", "Predicted by our Model"))

    print("--"*10)

    for w,true,pred in zip(x_test[i],y_true,p[0]):
        print("{:15}{}\t {}".format(words[w-1],pos[true],pos[pred]))


    # F1-score
    print("F1-score: ",str(f1_score(y_true,p[0],average='micro')))

    model.save_weights('best_model.h5')

if __name__ == '__main__':
    main()