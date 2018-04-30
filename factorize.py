#!/usr/bin/env python

import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.linalg as la
import threading

from scipy.sparse import csr_matrix, csc_matrix
from parser import *
from worker import *
from common import *
from sklearn.metrics.pairwise import cosine_distances


def normalize_data(X):
    return X / np.max(X)

def main():
    parser = argparse.ArgumentParser(description='nmtf-cod')
    parser.add_argument('-m', '--method', default='', help="Factorization model (either nmf or nmtf)")
    parser.add_argument('-s', '--sparse', action="store_true", help="Use sparse matrices")
    parser.add_argument('-t', '--technique', default='', help="Optimization technique (mu, cod, als, pg)")
    parser.add_argument('-k', '--k', default='20', help="Factorization rank")
    parser.add_argument('-p', '--parallel', type=int, default=6, help="Number of parallel workers")
    parser.add_argument('-S', '--seed', default='42', help="Random seed")
    
    parser.add_argument('-V', '--verbose', action="store_true", help="Print error function in each iteration")
    parser.add_argument('data', nargs='*', help='Other args')
    
    args = parser.parse_args()
    filename = args.data[0]
    data = load_data(filename)
    if data is None:
        raise Exception("Unable to open file: %s" % filename)
    #data = data.todense()
    #print type(data)
    X = data
    if type(data) != csr_matrix:
        X = np.array(X, dtype=dtype, order='C')
        #X = normalize_data(X)
    
    if args.sparse:
        X = csr_matrix(X)
    
    np.random.seed(42)
    max_iter = 100
    
    function_dict = {
        ('nmtf', 'mu'): nmtf,
        ('nmtf', 'cod'): nmtf_cod,
        ('nmtf', 'als'): nmtf_als,
        ('nmtf', 'pg'): nmtf_pg,
    }
    
    basedata = os.path.splitext(os.path.basename(filename))[0]
    
    method_list = [args.method]
    if args.method == '':
        method_list = ['nmf', 'nmtf']
    
    technique_list = args.technique.split(',')
    if args.technique == '':
        technique_list = ['mu', 'cod', 'als', 'pg']
        
    k_list = [int(s) for s in args.k.split(',')]
    seed_list = [int(s) for s in args.seed.split(',')]
    
    if len(k_list) > 1 and len(seed_list) > 1:
        print("Multiple k and seed parameters supplied")
        print("Use either K or seed as a list")
        raise Exception("Too many parameters exception")
    
    tasks = []
    conv_trace = {}
    for key in [(m,t) for m in method_list for t in technique_list]:
        if key not in function_dict:
            print("Model %s with technique %s is not available" % key)
            continue
        
        conv_trace[key] = []
        for k in k_list:
            for seed in seed_list:
                params = {'X': X, 'k': k, 'seed': seed, 'method': key[0], 'technique': key[1], 
                    'max_iter': max_iter, 'verbose': args.verbose, 'store_results': True, 'basename': basedata}
                tasks.append(Task(function_dict[key], params))
    
    model = Model()
    mt = MainThread(model, n_workers=args.parallel)
    mt.start()
    for task in tasks:
        model.q.put(task)
    model.q.put(0)
    
    try:
        while mt.isAlive():
            time.sleep(1)
    except KeyboardInterrupt:
        print "KeyboardInterrupt"
        mt.stop()
    
    mt.join()


if __name__ == '__main__':
    main()