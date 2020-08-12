import os
import sys
import re
import gc
import csv
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(2019)
os.environ["OMP_NUM_THREADS"] = "8"
warnings.filterwarnings('ignore')

from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from keras.preprocessing import text, sequence
from keras import optimizers
from keras.optimizers import Adam, RMSprop
from keras import initializers, regularizers, constraints, callbacks, optimizers, layers, callbacks
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, CuDNNLSTM, CuDNNGRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.layers import PReLU, BatchNormalization
from keras.models import Model, load_model
from keras.engine import InputSpec, Layer
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU


# -------------------------------------
# --- utility functions and classes ---
# -------------------------------------
class AccuracyEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_label = np.argmax(y_pred, axis=1)
            y_val_label = np.argmax(self.y_val, axis=1)
            score = accuracy_score(y_val_label, y_pred_label)

            print("Accuracy - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')			

def read_data():
    with open('data_train.pkl', 'rb') as f:
        train = pickle.load(f)

    with open('data_test.pkl', 'rb') as f:
        test = pickle.load(f)

    X_train_all = np.array(train[0])
    y_train_all = np.array(train[1])
    X_test = np.array(test)
    X_train_all_df = pd.DataFrame([X_train_all]).T
    X_test_all_df = pd.DataFrame([X_test]).T
    X_train_all_df.columns = ['text']
    X_test_all_df.columns = ['text']

    y_train_all_binary_df = pd.get_dummies(y_train_all)

    return X_train_all_df, X_test_all_df, y_train_all_binary_df


embedding_file_dct = {
    'glove'   : './glove.840B.300d/glove.840B.300d.txt',
    'wiki'    : './wiki-news-300d-1M/wiki-news-300d-1M.vec',
    'paragram': './paragram_300_sl999/paragram_300_sl999.txt'
}
embed_size   = 300
max_features = 67236
max_len      = 150

# -----------------
# --- read data ---
# -----------------
print('Read data . . .')
train, test, target = read_data()


# --------------------------
# --- text preprocessing ---
# --------------------------

list_classes = target.columns.tolist()
y = target.values

print('Preprocessing . . .')
train.loc[:, 'text'] = train.loc[:, 'text'].str.lower()
test.loc[:, 'text'] = test.loc[:, 'text'].str.lower()

train.loc[:, 'text'] = train.loc[:, 'text'].apply(lambda x: clean_url(x))
test.loc[:, 'text'] = test.loc[:, 'text'].apply(lambda x: clean_url(x))

train.loc[:, 'text'] = train.loc[:, 'text'].apply(lambda x: clean_text(x))
test.loc[:, 'text'] = test.loc[:, 'text'].apply(lambda x: clean_text(x))

train.loc[:, 'text'] = train.loc[:, 'text'].apply(lambda x: clean_symbol(x))
test.loc[:, 'text'] = test.loc[:, 'text'].apply(lambda x: clean_symbol(x))

X_train_all = train

raw_text_train = X_train_all["text"]
raw_text_test = test["text"]

# tokenization
tk = Tokenizer(num_words=max_features, lower=True)
tk.fit_on_texts(raw_text_train)
X_train_all.loc[:, "text_seq"] = tk.texts_to_sequences(raw_text_train)
test.loc[:, "text_seq"] = tk.texts_to_sequences(raw_text_test)

# padding to fixed length
X_train_all = pad_sequences(X_train_all.text_seq, maxlen=max_len)
test = pad_sequences(test.text_seq, maxlen=max_len)


# -------------------------------------
# --- apply word-embeddings on text ---
# -------------------------------------
print('Transforming word embeddings on text . . .')

def load_glove(word_index, max_features):
    embedding_file = embedding_file_dct['glove']
    embeddings_index = dict(get_coefs(*o.split(' ')) for o in open(embedding_file))
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index))
    print('Number of word_index = {}'.format(len(word_index)))
    print('Number of words = {}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
def load_para(word_index, max_features):
    embedding_file = embedding_file_dct['paragram']
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore') if len(o) > 100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.0053247833, 0.49346462
    embed_size = all_embs.shape[1]
    nb_words = min(max_features, len(word_index))
    print('Number of word_index = {}'.format(len(word_index)))
    print('Number of words = {}'.format(nb_words))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

word_index = tk.word_index
nb_words = min(max_features, len(word_index))

embedding_matrix_glove = load_glove(word_index, max_features=max_features)
embedding_matrix_para = load_para(word_index, max_features=max_features)
embedding_matrix = np.hstack([embedding_matrix_glove, embedding_matrix_para])

	
# --------------------
# --- define model ---
# --------------------
embed_size = 600

def build_model(X_train, Y_train, X_valid, Y_valid, lr=0.0, lr_d=0.0, units=0, dr=0.0):
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x, x_h, x_c = Bidirectional(GRU(128, return_sequences=True, return_state=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, x_h, max_pool])
    x = Dense(192, activation="relu")(x)
    x = Dense(20, activation="softmax")(x)
    
    model = Model(inputs=inp, outputs=x)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr_d, clipvalue=1.0), metrics=["accuracy"])
    
    for bs in [64, 64, 128, 256, 512]:
        history = model.fit(X_train, Y_train, batch_size=bs, epochs=1, validation_data=(X_valid, Y_valid), 
                           verbose=1, callbacks=[acc_val])
    return model

	
# -----------------
# --- run model ---
# -----------------
print('Run models . . .')
n_folds = 10
folds = KFold(n_splits=n_folds, shuffle=True, random_state=2019)
preds = []

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train_all)):
    
    X_train, X_valid = X_train_all[trn_idx], X_train_all[val_idx]
    Y_train, Y_valid = y[trn_idx], y[val_idx]
	acc_val = AccuracyEvaluation(validation_data=(X_valid, Y_valid), interval=1)
    model = build_model(X_train, Y_train, X_valid, Y_valid, lr=1e-3, lr_d=1.5e-5, units=128, dr=0.3)
    preds.append(model.predict(test, batch_size=1024, verbose=1))

# --------------------------
# --- submit predictions ---
# --------------------------	
submission = np.zeros((test.shape[0], 20))
submission_df = pd.DataFrame(submission)
submission_df.columns = target.columns.tolist()

pred = sum(preds) / n_folds
submission_df[list_classes] = pred

submission_df_final = pd.DataFrame(submission_df.idxmax(axis=1))
submission_df_final.columns = ['Category']
submission_df_final.index.name = 'Id'
submission_df_final.to_csv('submission_2nd_milestone_v15.csv')
