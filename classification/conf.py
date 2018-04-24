import os

SQLITE_DB_TRAINING = '/home/vanduc/p4.sqlite'
SQLITE_DB_TEST = '/home/vanduc/p4.sqlite'

QUERY = """SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('sports')
         LIMIT 50) AS tab1
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('entertainment')
         LIMIT 50) AS tab2
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('business')
         LIMIT 50) AS tab3
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('politics')
         LIMIT 50) AS tab4
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('technology')
         LIMIT 50) AS tab5"""


KNN_PARAMS = {

    'n_neighbors': [100],
    'metric': ['euclidean', 'minkowski', 'manhattan']
}

SVM_PARAMS = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100]
    },
    {
        'kernel': ['linear'],
        'C': [1, 10, 100]
    }
]




# Saved DIR path

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_DIR = CURRENT_DIR + '/saved/'


