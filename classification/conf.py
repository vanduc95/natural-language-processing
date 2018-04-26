import os

SQLITE_DB = '/home/vanduc/lagtime.sqlite'

QUERY = """SELECT `content`, `type`
        FROM `news`
        ORDER BY RANDOM()
        LIMIT 10000"""

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