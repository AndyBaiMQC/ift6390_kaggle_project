import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, preprocessing
from sklearn.feature_selection import SelectKBest
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

def preprocess(X_text, X_val_text,X_test, y_train):
    vectorizer = TfidfVectorizer(ngram_range = (1,3),
                decode_error = 'replace',
                strip_accents= 'unicode',
                analyzer = 'word',
                stop_words = 'english',
                min_df = 2)
    X = vectorizer.fit_transform(X_text)
    X_test = vectorizer.transform(X_test)
    X_val = vectorizer.transform(X_val_text)

    selector = SelectKBest(k=min(25000, X.shape[1]))
    selector.fit(X, y_train)
    X = selector.transform(X)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)

    print(X.shape, X_val.shape, X_test.shape)
    return X, X_val, X_test

# Source: https://developers.google.com/machine-learning/guides/text-classification/step-4
def create_model(hidden_layers, units, dropout_rate, input_shape, num_classes):
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(hidden_layers):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=num_classes, activation='softmax'))
    return model


def trained_model(data,
                learning_rate=1e-3,
                epochs=100,
                batch_size=128,
                hidden_layers=1,
                units=32,
                dropout_rate=0.0):

    (x_train, train_labels), (x_val, val_labels) = data

    num_classes = max(train_labels) + 1

    model = create_model(hidden_layers=hidden_layers,
                          units=units,
                          dropout_rate=dropout_rate,
                          input_shape=x_train.shape[1:],
                          num_classes=num_classes)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True)]

    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,
            batch_size=batch_size)

    history = history.history
    print('Best validation accuracy:', max(history['val_acc']), ' loss:', history['val_loss'][np.argmax(history['val_acc'])])
    return model

def majority_predictions(dataset, stacked_models):
    # method recycled from bagging
    # majority predictions of models
    votes = np.zeros((dataset.shape[0], 20)) #(nb samples, nb classes) where votes[i,j] is nb of vote that sample i is from class j
    for sub_model in stacked_models:
        class_preds = np.argmax(sub_model.predict(dataset), axis=1)  # for each samples, int representing the class with highest prob
        votes = votes + np.eye(20)[class_preds] #transform class_preds into onehot were the one is the vote for the predicted class
    return np.argmax(votes, axis=1) #for each samples, return int representing the class with most votes


def create_submission(prediction):
    index = np.arange(0,len(prediction))
    result = np.column_stack((index, prediction))
    header = np.array(['Id','Category'])
    result = np.vstack((header,result))
    result = result.astype(str)
    print(result)
    np.savetxt("submission_nn.csv", result, delimiter=",",  fmt='%s')



if __name__== "__main__":


    X, y = np.load("reddit-comments/data_train.pkl", allow_pickle=True)

    x_train, x_val, y_train, y_valid = model_selection.train_test_split(X, y)

    X_test = np.load("reddit-comments/data_test.pkl", allow_pickle=True)

    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_valid = encoder.transform(y_valid)

    x_train, x_val, X_test = preprocess(x_train, x_val, X_test, y_train)

    model = trained_model(((x_train, y_train), (x_val, y_valid)),
                    learning_rate=1e-3,
                    epochs=50,
                    batch_size=128,
                    hidden_layers=1,
                    units=32,
                    dropout_rate=0.5)

    preds = majority_predictions(X_test, [model])
    print('\n', preds)
    print('\n', encoder.inverse_transform(preds))
    create_submission(encoder.inverse_transform(preds))
