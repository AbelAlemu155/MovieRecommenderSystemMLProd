

import sqlite3, hashlib
from constants import *
def pyhash(x):
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)


def define_hash(con):
    con.create_function("pyhash", 1, pyhash)

def map_user_to_group(user_id): 
    if(pyhash(user_id) % 100) < 10:
        return TREATMENT_GROUP
    return CONTROL_GROUP

