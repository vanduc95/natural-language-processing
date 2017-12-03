import os

# SQLITE_DB_TRAINING = '/home/vanduc/lagtime.sqlite'
SQLITE_DB_TRAINING = '/home/vanduc/p4.sqlite'
SQLITE_DB_TEST = '/home/vanduc/p4.sqlite'

QUERY = """SELECT `content`, `type`
      FROM `news`
      WHERE `type` IN ('business', 'entertainment', 'health',
                    'sports', 'politics')
      ORDER BY RANDOM() LIMIT 200"""

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
print(CURRENT_DIR)
SAVED_DIR = CURRENT_DIR + '/saved/'