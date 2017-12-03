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

def load_pickle(text_extraction):
    if text_extraction == 'BOW':
        path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_BOW.pickle"
    else:
        path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF-IDF.pickle"

    return pickle.load(open(path, 'rb'))


def split_train_test(text_extraction):
    X_test, y_test = utils.get_data_from_db('test')

    classifer = load_pickle(text_extraction)
    y_pred = classifer.predict(X_test)
    y_score = classifer.estimator.predict_proba(classifer.vectorizer.transform(X_test))

    result = {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_score': y_score,
    }
    return result



if __name__ == '__main__':
    # print(metrics.accuracy_score(y_test, y_pred))
    # print(metrics.confusion_matrix(y_test, y_pred))
    # print(metrics.classification_report(y_test, y_pred, target_names))

    # visualize_confusion_matric()
    plt.figure(figsize=(8, 6))
    result1 = split_train_test('BOW')
    plot.confusion_matrix(result1['y_test'], result1['y_pred'])
    # plot.precision_recall(result1['y_test'], result1['y_score'])
    plt.savefig('bow-1.png')
    plt.close()

    # plt.figure(figsize=(8, 6))
    # plot.roc(result1['y_test'], result1['y_score'])
    # plt.savefig('bow-2.png')
    # plt.close()
    #
    # plt.figure(figsize=(8, 6))
    # result2 = split_train_test('TF-IDF')
    # # plot.confusion_matrix(result['y_test'], result['y_pred'])
    # plot.precision_recall(result2['y_test'], result2['y_score'])
    # plt.savefig('tf-1.png')
    # plt.close()
    #
    # plt.figure(figsize=(8, 6))
    # plot.roc(result2['y_test'], result2['y_score'])
    # plt.savefig('tf-2.png')
    # plt.close()
