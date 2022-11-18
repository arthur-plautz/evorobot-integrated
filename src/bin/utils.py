import os 
import numpy as np

# Sorting functions

# Descendent sorting
def descendent_sort(vect):
    # Copy of the vector
    tmpv = np.copy(vect)
    n = len(tmpv)
    # Index list
    index = np.arange(n, dtype=np.int32)
    i = 0
    while i < n:
        # Look for maximum
        maxv = tmpv[0]
        maxi = 0
        j = 1
        while j < n:
            if tmpv[j] > maxv:
                maxv = tmpv[j]
                maxi = j
            j += 1
        vect[i] = tmpv[maxi]
        index[i] = maxi
        i += 1
        # Set invalid value
        tmpv[maxi] = -999999999999.0
    return vect, index

# Ascendent sorting
def ascendent_sort(vect):
    # Copy of the vector
    tmpv = np.copy(vect)
    n = len(tmpv)
    # Index list
    index = np.arange(n, dtype=np.int32)
    i = 0
    while i < n:
        # Look for maximum
        minv = tmpv[0]
        mini = 0
        j = 1
        while j < n:
            if tmpv[j] < minv:
                minv = tmpv[j]
                mini = j
            j += 1
        vect[i] = tmpv[mini]
        index[i] = mini
        i += 1
        # Set invalid value
        tmpv[mini] = 999999999999.0
    return vect, index

def get_root_dir():
    exe_path = str(os.getcwd())
    split_str = exe_path.split('evorobot-integrated')
    return split_str[0] + 'evorobot-integrated'

def create_dir(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)

def create_dirs(base_path, directories):
    incremental_path = base_path
    for directory in directories.split('/'):
        incremental_path += '/'+directory
        if not os.path.isdir(incremental_path):
            os.mkdir(incremental_path)

def remove_dir(directory):
    if os.path.isdir(directory):
        os.rmdir(directory)

def remove_file(path):
    if os.path.isfile(path):
        os.remove(path)

def verify_file(path):
    if os.path.isfile(path):
        return True
