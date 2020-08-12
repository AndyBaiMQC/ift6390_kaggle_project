import os
import re
import gc
import time
start_time = time.time()

import pickle
import string
import numpy as np
import pandas as pd

from utils import *

from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from utils import *

english_stemmer = SnowballStemmer('english')


def read_data():
    with open('data_train.pkl', 'rb') as f:
        train = pickle.load(f)

    with open('data_test.pkl', 'rb') as f:
        test = pickle.load(f)

    # Convert to numpy array
    X_train_all = np.array(train[0])
    y_train_all = np.array(train[1])
    X_test = np.array(test)

    return X_train_all, y_train_all, X_test

def extract_text_numerical_features(df, colname='text'):

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


if __name__ == '__main__':
    # Load train and test data
    print('Loading data . . .')
    X_train_all, y_train_all, X_test = read_data()
    train_size = X_train_all.shape[0]
    print('Training size = {}'.format(train_size))

    # Convert to pandas dataframe
    X_train_all_df = pd.DataFrame([X_train_all]).T
    X_test_all_df = pd.DataFrame([X_test]).T
    X_train_all_df.columns = ['text']
    X_test_all_df.columns = ['text']
    X_all_df = pd.concat([X_train_all_df, X_test_all_df])

    # Extract meta numerical features
    X_text_num_capitals = X_all_df['text'].apply(lambda x: np.sum(1 for c in x if c.isupper())).values.reshape(-1, 1)
    X_text_num_title = X_all_df['text'].apply(lambda x: np.sum(1 for c in x.split() if c.isupper())).values.reshape(-1, 1)

    X_all_df['text'] = X_all_df['text'].apply(lambda x: clean_text(x))
    X_all_df['text'] = X_all_df['text'].apply(lambda x: clean_url(x))

    X_all_df.loc[:, 'text_len'] = X_all_df['text'].apply(lambda x: len(x))

    X_text_num_exclamation = X_all_df['text'].apply(lambda x: x.count('!')).values.reshape(-1, 1)
    X_text_num_punct = X_all_df['text'].apply(lambda x: np.sum(x.count(p) for p in string.punctuation)).values.reshape(-1, 1)
    X_text_len = X_all_df['text_len'].values.reshape(-1, 1)
    X_text_len_inv = 1 / (X_all_df['text_len'] + 1).values.reshape(-1, 1)
    X_text_len_log_inv = 1 / np.log1p(X_all_df['text_len'] + 1).values.reshape(-1, 1)
    X_text_num_words = X_all_df['text'].apply(lambda x: len(x.split())).values.reshape(-1, 1)
    X_text_words_len_ratio = X_text_num_words * X_text_len_inv
    X_text_words_len_log_ratio = X_text_num_words * X_text_len_log_inv
    X_text_len_inv_pw = X_text_len_inv ** 1.5
    X_text_capital_len_ratio = X_text_num_capitals * X_text_len_inv
    X_text_uniq_words = X_all_df['text'].apply(lambda x: len(set(w for w in x.split()))).values.reshape(-1, 1)
    X_text_uniq_words_ratio = (X_text_uniq_words + 1) / (X_text_num_words + 1)
    X_text_exclamation_ratio = X_text_num_exclamation / (X_text_num_punct + 1)
    X_text_uniq_words_ratio2 = X_text_uniq_words * X_text_len_inv
    X_text_num_sent = X_all_df['text'].apply(lambda x: len(re.split("\\.{1,}|\\!{1,}|\\?{1,}|\n{1,}", x))).values.reshape(-1, 1)
    X_text_avg_sent_len = X_text_num_sent * X_text_len_inv
    X_text_uniq_words_sent_ratio = X_text_uniq_words / X_text_num_sent
    X_text_title_words_ratio = (X_text_num_title + 1) / (X_text_num_words + 1)

    X_num_features = np.hstack((X_text_num_capitals,
                                X_text_num_title,
                                X_text_num_exclamation,
                                X_text_num_punct,
                                X_text_len_inv, 
                                X_text_len_log_inv,
                                X_text_num_words,
                                X_text_words_len_ratio,
                                X_text_words_len_log_ratio,
                                X_text_len_inv_pw,
                                X_text_capital_len_ratio,
                                X_text_uniq_words,
                                X_text_uniq_words_ratio,
                                X_text_exclamation_ratio,
                                X_text_uniq_words_ratio2,
                                X_text_num_sent, 
                                X_text_avg_sent_len,
                                X_text_uniq_words_sent_ratio,
                                X_text_title_words_ratio))

    X_train_num = X_num_features[:train_size]
    X_test_num = X_num_features[train_size:]
    print('Numerical features = {}'.format(X_num_features.shape[1]))

    # Tfidf 
    tfidf_vectorizer = StemmedTfidfVectorizer(ngram_range=(1, 1),
        min_df=1, 
        strip_accents='unicode',
        sublinear_tf=True,
        token_pattern=r'\w{1,}|\!{2,}|\.{2,}|\?{2,}|\n{2,}')

    tfidf_vectorizer_char = TfidfVectorizer(ngram_range=(1, 6),
        sublinear_tf=True,
        min_df=1,
        strip_accents='unicode',
        analyzer='char',
        max_features=300000)

    print('Running tfidf on unigram . . .')
    X_text_all = tfidf_vectorizer.fit_transform(X_all_df['text'].values)
    print('Running tfidf (char level) . . .')
    X_text_char_all = tfidf_vectorizer_char.fit_transform(X_all_df['text'].values)
    X_sparse = hstack((X_text_all, X_text_char_all)).tocsr()
    print('Combined tfidf features = {}'.format(X_sparse.shape[1]))


    # Cross validation and data settings
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)
    cv_scores = []
    class_pred = np.zeros(train_size)

    X_text_train = X_sparse[:train_size]
    X_text_test = X_sparse[train_size:]

    X_text_train = hstack((X_text_train, X_train_num)).tocsr()
    X_text_test = hstack((X_text_test, X_test_num)).tocsr()

    # Processing target label and submission file
    y_train_all_binary_df = pd.get_dummies(y_train_all)
    class_names = y_train_all_binary_df.columns.tolist()
    y_train_all_binary_df_pred = y_train_all_binary_df.copy()
    y_train_all_binary_df_pred.loc[:, :] = 0
    submission = np.zeros((X_test.shape[0], 20))
    submission_df = pd.DataFrame(submission)
    submission_df.columns = y_train_all_binary_df_pred.columns.tolist()

    for i_c, class_name in enumerate(class_names):
        print('Class {}: {}'.format(i_c, class_name))
        X_text_train_, X_text_val_, y_train_train_binary_df, y_train_val_binary_df = train_test_split(X_text_train, y_train_all_binary_df, 
            test_size=0.2, random_state=2019)
        y_train = y_train_train_binary_df[class_name]
        y_val = y_train_val_binary_df[class_name]
        clf = NbSvmClassifier(C=100, n_jobs=8)
        clf.fit(X_text_train_, y_train)
        y_val_pred_proba = clf.predict_proba(X_text_val_)[:, 1]
        y_val_pred = (y_val_pred_proba > .5).astype(int)

        print('\tAccuracy = {:.5f}'.format(accuracy_score(y_val, y_val_pred)))
        submission_df[class_name] = clf.predict_proba(X_text_test)[:, 1]

        # for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_text_train)):
        #     clf = NbSvmClassifier(C=100, n_jobs=8)
        #     clf.fit(X_text_train[trn_idx.reshape(-1)], train_target[trn_idx.reshape(-1)])
        #     y_train_all_binary_df_pred[class_name][val_idx.reshape(-1)] = clf.predict_proba(X_text_train[val_idx.reshape(-1)])[:, 1]
        #     y_preds = clf.predict(X_text_train[val_idx.reshape(-1)])
        #     print('\tFold {}: accuracy = {:.5f}'.format(n_fold + 1, accuracy_score(y_preds, y_train_all_binary_df[class_name][val_idx.reshape(-1)])))
        #     submission_df[class_name] = clf.predict_proba(X_text_test)[:, 1] / folds.n_splits

    submission_df_final = pd.DataFrame(submission_df.idxmax(axis=1))
    submission_df_final.columns = ['Category']
    submission_df_final.index.name = 'Id'
    submission_df_final.to_csv('submission_2nd_milestone_v1.csv')

    y_train_val_pred = y_train_all_binary_df.loc[y_train_val_binary_df.index].idxmax(axis=1)
    print('Validation accuracy = {:.5f}'.format(accuracy_score(y_train_all[y_train_val_binary_df.index], y_train_val_pred.values)))

    print('Time used = {:.2f}'.format(time.time() - start_time))
