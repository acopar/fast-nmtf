#!/usr/bin/env python

import os
import sys
import csv
import time
import pickle
import numpy as np
from scipy.sparse import csr_matrix

def ensure_dir(f):
    d = os.path.dirname(f)
    if d:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except OSError as e:
                print(f)
                raise e

### Load ###

def load_coo(filename, verbose=False):
    fp = open(filename, 'r')
    reader = csv.reader(fp, delimiter=',')
    header = reader.next()
    if len(header) != 2:
        raise Exception("Wrong coo header format")
    
    n = int(header[0])
    m = int(header[1])
    X = np.zeros((n,m), dtype=np.float32)
    it = 0
    t0 = time.time()
    for line in reader:
        if verbose == True:
            t1 = time.time()
            if t1 - t0 > 10:
                print("Progress: %.2fM lines processed" % (float(it) / 1000000))
            t0 = t1
        i = int(line[0])
        j = int(line[1])
        value = float(line[2])
        if value < 0:
            print('Warning: setting negative value to zero, position (%d,%d)' % (i,j))
            value = 0.0
        X[i,j] = value
        it += 1
    
    fp.close()
    return X

def load_csv(filename, delimiter=','):
    # loads csv file into array of rows (arrays)
    csv.field_size_limit(sys.maxsize)
    fp = open(filename)
    if not fp:
        print("Error: Cannot open file: %s" % filename)
        return None
    
    reader = csv.reader(fp, delimiter=delimiter)
    data = []
    for row in reader:
        data.append(row)
    fp.close()
    return data

def load_numpy(filename):
    if os.path.isfile(filename) == False:
        print("Error: Cannot open file: %s" % filename)
        return None
    d = np.load(filename)
    if 'indices' in d:
        # sparse
        X = csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])
        return X
    elif 'data' in d:
        return d['data']
    else:
        print("Error: No numpy data in file %s" % filename)
        return None

### Store ###

def write_coo(filename, X):
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    n = X.shape[0]
    m = X.shape[1]
    writer.writerow([n,m])
    for i in range(n):
        for j in range(m):
            value = X[i,j]
            if value > 0:
                writer.writerow([i,j,value])
    fp.close()

def save_csv(filename, data, delimiter=',', append=False):
    # write csv file (arrays)
    ensure_dir(filename)
    if append:
        fp = open(filename, 'a')
    else:
        fp = open(filename, 'w')
    
    if not fp:
        print("Error: Cannot open file: %s" % filename)
        return None
    
    writer = csv.writer(fp, delimiter=delimiter)
    for line in data:
        writer.writerow(line)
    fp.close()

def save_numpy(filename, data):
    ensure_dir(filename)
    if type(data) == csr_matrix:
        np.savez(filename, data=data.data, indices=data.indices, indptr=data.indptr, shape=data.shape)
    else:
        np.savez(filename, data=data)

def load_data(filename):
    base, ext = os.path.splitext(filename)
    if ext == '.coo':
        X = load_coo(filename)
    elif ext == '.csv':
        X = load_csv(filename)
    elif ext == '.npz':
        X = load_numpy(filename)
    else:
        raise Exception("Format not recognized: %s" % ext)
    return X

def save_data(filename, X):
    base, ext = os.path.splitext(filename)
    if ext == '.coo':
        write_coo(filename, X)
    elif ext == '.csv':
        save_csv(filename, X)
    elif ext == '.npz':
        save_numpy(filename, X)
    else:
        raise Exception("Format not recognized: %s" % ext)


def load_file(filename):
    # load pickle file
    if os.path.isfile(filename) == False:
        print("Error: Cannot open file: %s" % filename)
        return None
    fp = open(filename, 'rb')
    d = pickle.load(fp)
    fp.close()
    return d

def dump_file(filename, data):
    # dump pickle file
    ensure_dir(filename)
    fp = open(filename, 'wb')
    pickle.dump(data, fp, 2)
    fp.close()