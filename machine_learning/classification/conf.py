import os

# SQLITE_DB_TRAINING = '/home/vanduc/lagtime.sqlite'
SQLITE_DB_TRAINING = '/home/vanduc/p4.sqlite'
SQLITE_DB_TEST = '/home/vanduc/p4.sqlite'

QUERY1 = """SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('sports')
         LIMIT 2000) AS tab1
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('entertainment')
         LIMIT 2000) AS tab2
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('business')
         LIMIT 2000) AS tab3
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('politics')
         LIMIT 2000) AS tab4
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('technology')
         LIMIT 2000) AS tab5"""


QUERY2 = """SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('sports')
         LIMIT 3000, 2000) AS tab1
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('entertainment')
         LIMIT 3000, 2000) AS tab2
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('business')
         LIMIT 3000, 2000) AS tab3
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('politics')
         LIMIT 3000, 2000) AS tab4
UNION ALL
SELECT content, type
  FROM (SELECT * FROM news
         WHERE type IN ('technology')
         LIMIT 2000, 2000) AS tab5"""

QUERY = QUERY1

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


