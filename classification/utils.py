import sqlite3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from classification import conf
import logging
import numpy as np
import os
import pickle

LOG = logging.getLogger(__name__)


def get_data_from_db(type_data):
    '''
    Get 'content' and 'type' from database
    :rtype: list
    '''

    # Initialize `connection` var to None. In case we could not
    # create a connection to the database(for ex the disk is full)
    # we would not have a connection var defined.

    if type_data == 'training':
        db = conf.SQLITE_DB_TRAINING
    elif type_data == 'test':
        db = conf.SQLITE_DB_TEST

    connection = None
    try:
        connection = sqlite3.connect(db)

        cursor = connection.cursor()
        cursor.execute(conf.QUERY)
        # result = cursor.fetchmany(rows)
        # LOG.info('Get random data from database.')

        contents = []
        labels = []

        for r in cursor:
            contents.append(r[0].strip())
            labels.append(r[1])
        return (contents, labels)

    except Exception as e:
        LOG.exception('Failed when connecting to database: {}. '.format(e))

    finally:
        if connection:
            LOG.info('Close database connection.')
            connection.close()


def bag_of_words():
    """Bag of words vectorize.
    """
    return CountVectorizer(stop_words='english', token_pattern=r'\b[^\W\d_]+\b', max_features=None)


def td_idf(is_eng=True):
    """Tf Idf vectorize.
    """
    return TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\W\d_]+\b', max_features=None)


def save(obj, path):
    """Save Classifier object to pickle file."""
    if os.path.isfile(path):
        LOG.info('File existed! Use load() method.')
    else:
        pickle.dump(obj, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load(path):
    """Load Classifier object from pickle file"""
    if not os.path.isfile(path):
        LOG.info('File doesnt existed!')
        raise IOError()
    else:
        return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    X, a = get_data_from_db()
    y = np.array(a)
    print(X)
