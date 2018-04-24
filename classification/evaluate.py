from classification.classifier import Classifier
from classification import conf
from classification import utils
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle
import itertools
import numpy as np
from sklearn_evaluation import ClassifierEvaluator
from sklearn_evaluation import plot

def load_pickle(alogrithm, text_extraction):
    if alogrithm == 'KNN':
        if text_extraction == 'TF-IDF':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/KNN_TF-IDF.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/KNN_BOW.pickle"
    else:
        if text_extraction == 'TF-IDF':
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF-IDF.pickle"
        else:
            path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_BOW.pickle"

    return pickle.load(open(path, 'rb'))


def split_train_test(alogrithm, text_extraction):
    X_test, y_test = utils.get_data_from_db('test')

    classifer = load_pickle(alogrithm, text_extraction)
    y_pred = classifer.predict(X_test)

    result = {
        'y_test': y_test,
        'y_pred': y_pred,
    }
    return result



if __name__ == '__main__':
    result = split_train_test('SVM','TF-IDF')
    target_names = ['business', 'entertainment', 'health','sports', 'politics']
    # print(metrics.accuracy_score(result['y_test'], result['y_pred']))
    # print(metrics.confusion_matrix(result['y_test'], result['y_pred']))
    print(metrics.classification_report(result['y_test'], result['y_pred'], target_names))

    '''visualize_confusion_matric()'''
    # plt.figure(figsize=(8, 6))
    # result1 = split_train_test('SVM', 'TF-IDF')
    # plot.confusion_matrix(result1['y_test'], result1['y_pred'])
    # plt.savefig('svm-confusion-matrix.png')
    # plt.close()
    #
    # plt.figure(figsize=(8, 6))
    # plot.precision_recall(result1['y_test'], result1['y_score'])
    # plt.savefig('svm-precision-recall.png')
    # plt.close()
    #
    # plt.figure(figsize=(8, 6))
    # result2 = split_train_test('KNN', 'TF-IDF')
    # plot.confusion_matrix(result2['y_test'], result2['y_pred'])
    # plt.savefig('knn-confusion-matrix.png')
    # plt.close()
    #
    # plt.figure(figsize=(8, 6))
    # plot.precision_recall(result2['y_test'], result2['y_score'])
    # plt.savefig('knn-precision-recall.png')
    # plt.close()
