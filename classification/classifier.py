import logging
import os
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from classification import conf
from classification import utils


LOG = logging.getLogger(__name__)


class Classifier(object):

    """Classifier class"""

    def __init__(self, algorithm, text_extract_type, is_eng=True):
        """
        :param str algorithm: choosen algorithm will be used to classify.
        :param str text_extract_type: text extraction (Bag of Words & TF_IDF).
        :param boolean is_eng: language can be eng or vi.
        """
        self.algorithm = algorithm
        self.text_extract_type = text_extract_type
        self.is_eng = is_eng
        self.estimator = None
        self.train()

    def load_train_set(self):
        """Load train set from database."""
        return utils.get_data_from_db()

    def word_to_vector(self):
        """Convert from text to vector."""
        contents, labels = self.load_train_set()

        if self.text_extract_type == 'Bag Of Words':
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
            tmp_estimator = SVC(C=1, probability=True)
            pre_tune_params = conf.SVM_PARAMS
        else:
            tmp_estimator = KNeighborsClassifier()
            pre_tune_params = conf.KNN_PARAMS

        # Tuning estimator, modify n_jobs and pre_dispatch depend on
        # your machine, which run this. More details, plz read sklearn
        # documentation about GridSearchCV.
        grid_search = GridSearchCV(tmp_estimator, pre_tune_params,
                                   cv=2, n_jobs=-1, pre_dispatch='0.5*n_jobs')
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
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/K-NN_Bag Of Words_eng.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/K-NN_TF IDF_eng.pickle"
    else:
        if text_extraction == 'Bag of word':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_Bag Of Words_eng.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF IDF_eng.pickle"
    a = pickle.load(open(path, 'rb'))
    result = a.predict([content])
    return result[0]

if __name__ == '__main__':

    # classifier = Classifier('SVM','TF-IDF')
    # utils.save(classifier,conf.SAVED_DIR+'SVM_TF IDF_eng.pickle')
    # content = []
    # # content.append("Investigators looking for 5-year-old Aramazd Andressian Jr., who has been missing for more than a month, searched the Montebello home of his grandmother on Thursday. During the two-hour search of the paternal grandmother’s home, which began about 9 a.m., investigators seized “various items of evidence,” according to the Los Angeles County Sheriff’s Department.“We’re just searching for additional evidence to help us locate the child,” said Los Angeles Sheriff’s Det. Louie Aguilera. He declined to say what items authorities recovered but said the search was related to “new investigative leads that we have developed.”The boy was reported missing on April 22, two days after he went to Disneyland with his father and his relatives.Aramazd Andressian Sr., 35, has been named a person of interest in the case, and detectives said that he has given “inconsistent” and “misleading” statements. Andressian was found unconscious in a South Pasadena park at 6:30 a.m. April 22, the day he was supposed to return the boy to his  estranged wife. .Andressian was briefly held on charges of child endangerment and child abduction, then released because of insufficient evidence. He has hired a lawyer and refused to speak with investigators, giving only a written statement provided by his attorney.Aramazd was last seen alive April 21 about 1 a.m. after leaving Disneyland with his father and other relatives. Investigators also think the elder Andressian visited the Cachuma Lake Recreation Area in Santa Barbara County with the boy before he was reported missing, but there were no confirmed sightings of the child there.Earlier this month, authorities increased a reward for information on Aramazd’s disappearance to $20,000. The boy’s mother, Ana Estevez, made a tearful plea to the public at a news conference.“My son’s disappearance is my worst nightmare,” she said, adding that she thinks Aramazd is still alive. She told her son to be brave and that she loves him.The next day deputies returned to the park where Andressian was found to conduct a large-scale search with cadaver dogs, horses and a drone. They found nothing. The CBO, a nonpartisan office, analyzed the health bill passed by the House. A housing project built with shipping containers just opened in Orange County. Why are donut boxes pink? Joe Rohde is the Imagineer behind Disneys Pandora and Guardians of the Galaxy rides at their parks.Credits: Myung J. Chun / Jay L. Clendenin / Gary Coronado / KTLA / Allen J. SchabenThe CBO, a nonpartisan office, analyzed the health bill passed by the House. A housing project built with shipping containers just opened in Orange County. Why are donut boxes pink? Joe Rohde is the Imagineer behind Disneys Pandora and Guardians of the Galaxy rides at their parks.Credits: Myung J. Chun / Jay L. Clendenin / Gary Coronado / KTLA / Allen J. SchabenThe CBO, a nonpartisan office, analyzed the health bill passed by the House. A housing project built with shipping containers just opened in Orange County. Why are donut boxes pink? Joe Rohde is the Imagineer behind Disneys Pandora and Guardians of the Galaxy rides at their parks.Credits: Myung J. Chun / Jay L. Clendenin / Gary Coronado / KTLA / Allen J. SchabenThe CBO, a nonpartisan office, analyzed the health bill passed by the House. A housing project built with shipping containers just opened in Orange County. Why are donut boxes pink? Joe Rohde is the Imagineer behind Disneys Pandora and Guardians of the Galaxy rides at their parks.Credits: Myung J. Chun / Jay L. Clendenin / Gary Coronado / KTLA / Allen J. SchabenChapman University has awarded an honorary degree to the mother of a quadriplegic student after she attended every class with him and took his notes while he pursued his master’s degree in business administration.Chapman University has awarded an honorary degree to the mother of a quadriplegic student after she attended every class with him and took his notes while he pursued his master’s degree in business administration.Britain warned of another possible terrorist attack on Tuesday. States are moving fast to protect their pot industries from threats by the White House. The Marciano Art Museum is opening a landmark L.A. building to the public for the first time. Ready for an endless winter?Credits: Getty / Glenn Koeing / KTLA / Yoshihiro MakinoBritain warned of another possible terrorist attack on Tuesday. States are moving fast to protect their pot industries from threats by the White House. The Marciano Art Museum is opening a landmark L.A. building to the public for the first time. Ready for an endless winter?Credits: Getty / Glenn Koeing / KTLA / Yoshihiro MakinoSan Clemente beaches were reopened Tuesday with no shark sightings through the morning. The were closed on Monday after a fisherman hooked a 12-foot shark off the pier. (Mark Boster / Los Angeles Times)San Clemente beaches were reopened Tuesday with no shark sightings through the morning. The were closed on Monday after a fisherman hooked a 12-foot shark off the pier. (Mark Boster / Los Angeles Times)Former CIA Director John Brennan was asked why Americans should care about the investigation into Russian interference in the 2016 U.S. presidential election.Former CIA Director John Brennan was asked why Americans should care about the investigation into Russian interference in the 2016 U.S. presidential election.nicole.santacruz@latimes.comFor more crime news, follow @nicolesantacruz on Twitter.")
    # content.append('sport game boxing')
    # result = classifier.predict(content)
    # print(result)


    print (excute('SVM', 'TFIDF', 'sports boxing'))































