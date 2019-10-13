#naive bayes classifier usnig bag of words
#phase 1: can only uses standard python package + numpy + nltk

#Source: the idea of upweighting title words comes from Ko_2002.pdf (Improving text categorization using the importance of sentences)

import numpy as np
import nltk
import copy
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class BayesClassifier:

    def __init__(self, X, y, laplace_cst=0, stem_method=None, upweight_train=0, upweight_test=0):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.laplace_cst = laplace_cst
        self.stem_method = stem_method
        self.upweight_train = upweight_train
        self.upweight_test = upweight_test
        self.m = len(set(y)) #number of classes
        self.categories = list(set(y))
        self.dict_category_index = {k: v for v, k in enumerate(self.categories)} #dict_category_index['cat'] == self.categories.index('cat') but quicker

    def preprocess(self, X):
        # Replace every text for a list of preprocessed words

        stop_words = set(stopwords.words('english'))
        for i in range(len(X)):
            if self.stem_method is not None:
                X[i] = [self.stem_method(word) for word in nltk.word_tokenize(X[i]) if
                        word not in stop_words]
            else:
                #X[i] = re.sub(r'http\S+', '', X[i]) #remove links
                X[i] = re.sub('\W+',' ', X[i] ) #remove special caracters
                #X[i] = ' '.join([word for word in X[i].split() if len(word) > 2]) # remove words that are 1 or 2 characters

                #upweighting title words
                for word in X[i].split(' '):
                    cats = [c.lower() for c in self.categories]
                    if word.lower() == self.y_train[i].lower() and X == self.X_train:
                        X[i] += (' ' + word)*self.upweight_train
                    elif word.lower() in cats and X != self.X_train:
                        X[i] += (' ' + word)*self.upweight_test

                X[i] = [SnowballStemmer("english").stem(WordNetLemmatizer().lemmatize(word)) for word in nltk.word_tokenize(X[i]) if
                        word not in stop_words]


    def fit(self):
        X = self.X_train
        self.preprocess(X)
        all_words = [word for i in range(len(X)) for word in X[i]]
        self.nb_words = len(all_words)
        unique_words = list(set(all_words))
        self.dict_word_index = {k: v for v, k in enumerate(unique_words)} #word_index['word'] == unique_words.index('word') but quicker

        ## Create bag of words
        # each columns is a categories/label/subreddit. label_j = self.categories[j]
        # each row is a word. word_i = unique_words[i]
        # bag[i,j] = number of appearance of word i in category j
        bag = np.zeros((len(unique_words), self.m))

        self.nb_words_per_category = [0 for i in range(self.m)]

        for i in range(len(X)):
            for word in X[i]:
                cat_index = self.dict_category_index[self.y_train[i]]
                word_index = self.dict_word_index[word]
                bag[word_index, cat_index] += 1
                self.nb_words_per_category[cat_index] += 1

        #P(word_i|y_j) = (count(w_i & y_j) + cst) / (count(y_j) + cst*nb_words)
        self.log_probability_of_word_given_label = np.log((bag + self.laplace_cst) / (np.array(self.nb_words_per_category) + self.laplace_cst*self.nb_words))
        self.priors = np.array(self.nb_words_per_category) / self.nb_words

    def predict(self, X_test):
        # y_predicted for x is the y that maximizes P(x|y)*P(y)
        # P(x|y) = P(word_1,word_2,...,wosrd_n|y) = P(word_1|y)*P(word_2|y)*...*P(word_n|y)
        # P(word_i|y_j) = self.probability_of_word_given_label[i,j]
        # P(y) = self.priors
        predictions = []
        X_test = copy.deepcopy(X_test)
        self.preprocess(X_test)

        for sample in X_test:

            # for each sample, calculate argmax[P(sample|y_0)P(y_0), ..,P(sample|y_m-1)P(y_m-1)]
            words_prob = np.array([0 for i in range(self.m)]) # placeholder
            for word in sample:
                # for each word, get [P(word|y_0), ..., P(word|y_m-1)]
                ind = self.dict_word_index.get(word, None)
                if ind is not None:
                    words_prob = np.vstack([words_prob, self.log_probability_of_word_given_label[ind]])

                else:
                    # This word was never seen before
                    never_seen_prob = np.log((np.zeros((1, self.m)) + self.laplace_cst) / (np.array(self.nb_words_per_category) + self.laplace_cst * self.nb_words))
                    words_prob = np.vstack([words_prob, never_seen_prob])

            words_prob = words_prob[1:] #remove placeholder

            #log(P(sample|y_j)) = log(P(y_j)) + sum_i log(P(word|y_j))
            log_prob_sample_given_label = np.sum(words_prob, axis=0) + np.log(self.priors)
            pred_index = np.argmax(log_prob_sample_given_label)
            predictions.append(self.categories[pred_index])

        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        correct = 0
        for i in range(len(y)):
            if y_pred[i] == y[i]:
                correct += 1
        return correct / len(y)

    def create_submission(self,prediction):
        index = np.arange(0,len(prediction))
        result = np.column_stack((index, prediction))
        header = np.array(['Id','Category'])
        result = np.vstack((header,result))
        result = result.astype(str)
        print(result)
        np.savetxt("submission.csv", result, delimiter=",",  fmt='%s')




if __name__ == "__main__":
    X, y = np.load("reddit-comments/data_train.pkl", allow_pickle=True)

    idx = np.random.permutation(len(y))
    X = list(np.array(X)[idx])
    y = np.array(y)[idx]

    X_test = np.load("reddit-comments/data_test.pkl", allow_pickle=True)

    model = BayesClassifier(X, y, laplace_cst=0.02, stem_method=None, upweight_train=0, upweight_test=4)
    model.fit()
    prediction = model.predict(X_test)
    model.create_submission(prediction)

