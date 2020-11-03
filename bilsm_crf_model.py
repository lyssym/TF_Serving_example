# _*_ coding: utf-8 _*_

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras.initializers import glorot_normal
from keras_contrib.layers import CRF
import process_data
import pickle


EMBED_DIM = 40
BiRNN_UNITS = 160


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data.load_data()
    else:
        with open('model/config.pkl', 'rb') as f:
            (vocab, chunk_tags) = pickle.load(f)

    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, embeddings_initializer=glorot_normal(2020)))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    model.add(Dropout(0.4))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)
