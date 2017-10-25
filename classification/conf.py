import os

SQLITE_DB = '/home/vanduc/lagtime.sqlite'

QUERY = """SELECT `content`, `type`
        FROM `news`
        ORDER BY RANDOM()
        LIMIT 10000"""

