import logging
import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import time

from classification import conf
from classification import utils

start_time = time.time()
LOG = logging.getLogger(__name__)


class Classifier(object):
    """Classifier class"""

    def __init__(self, algorithm, text_extract_type):
        """
        :param str algorithm: choosen algorithm will be used to classify.
        :param str text_extract_type: text extraction (Bag of Words & TF_IDF).
        """
        self.algorithm = algorithm
        self.text_extract_type = text_extract_type
        self.estimator = None
        self.train()

    def load_train_set(self):
        """Load train set from database."""
        return utils.get_data_from_db('training')

    def word_to_vector(self):
        """Convert from text to vector."""
        contents, labels = self.load_train_set()

        if self.text_extract_type == 'BOW':
            LOG.info('Convert contents to BOW.')
            self.vectorizer = utils.bag_of_words()
        else:
            LOG.info('Convert contents to TF-IDF.')
            self.vectorizer = utils.td_idf()

        self.X = self.vectorizer.fit_transform(contents)
        self.y = np.array(labels)

    def _tuning_parameters(self):
        """Tuning the hyper-parameters of an estimator."""

        if self.algorithm == 'SVM':
            tmp_estimator = SVC(C=1)
            pre_tune_params = conf.SVM_PARAMS
        else:
            tmp_estimator = KNeighborsClassifier()
            pre_tune_params = [conf.KNN_PARAMS]

        # Tuning estimator, modify n_jobs and pre_dispatch depend on
        # your machine, which run this. More details, plz read sklearn
        # documentation about GridSearchCV.
        grid_search = GridSearchCV(tmp_estimator, pre_tune_params,
                                   cv=5, n_jobs=-1, pre_dispatch='0.5*n_jobs')
        grid_search.fit(self.X, self.y)
        LOG.info('Tuning the hyper-parameters of an {} estimator' .
                 format(self.algorithm))
        self.estimator = grid_search.best_estimator_


    def train(self):
        LOG.info('Training...')
        self.word_to_vector()
        if not self.estimator:
            self._tuning_parameters()

    def predict(self, content):
        content = self.vectorizer.transform(content)
        return self.estimator.predict(content)



def excute(alogrithm, text_extraction, content):
    if alogrithm == 'KNN':
        if text_extraction == 'Bag of word':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/KNN_BOW.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/KNN_TF-IDF.pickle"
    else:
        if text_extraction == 'Bag of word':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_BOW.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF-IDF.pickle"

    a = pickle.load(open(path, 'rb'))
    result = a.predict(content)
    return result


if __name__ == '__main__':

    content = ''' sport a game

    '''


    # classifier = Classifier('SVM', 'TF-IDF')
    # utils.save(classifier, conf.SAVED_DIR + 'SVM_TF-IDF.pickle')
    # print(excute('SVM', 'TF-IDF', [content]))


    # classifier = Classifier('KNN', 'TF-IDF')
    # utils.save(classifier, conf.SAVED_DIR + 'KNN_TF-IDF.pickle')
    # print(excute('KNN', 'BOW', ['sport a game','politics']))

    print("--- %s seconds ---" % (time.time() - start_time))
