# from classification.classifier import Classifier
# from classification import conf
# from classification import utils
# from sklearn import metrics
#
# import os
# import pickle
#
#
# def excute(text_extraction, content):
#     if text_extraction == 'BOW':
#         path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_BOW.pickle"
#     else:
#         path = os.path.dirname(os.path.realpath(__file__)) + "/saved/SVM_TF-IDF.pickle"
#
#     a = pickle.load(open(path, 'rb'))
#     result = a.predict(content)
#     return result
#
#
# if __name__ == '__main__':
#     X_test, y_test = utils.get_data_from_db('test')
#
#     y_pred = excute('TF-IDF', X_test)
#
#     print(metrics.accuracy_score(y_test, y_pred))
#
#     print(metrics.confusion_matrix(y_test, y_pred))
#
#     target_names = ['business', 'entertainment', 'health', 'sports', 'politics']
#
#     print(metrics.classification_report(y_test, y_pred, target_names))
