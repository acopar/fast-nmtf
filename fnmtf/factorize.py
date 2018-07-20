#!/usr/bin/env python

import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.linalg as la

from scipy.sparse import csr_matrix, csc_matrix
from loader import *
from worker import *
from common import *
from engine import Engine

from mu import nmtf_mu
from pg import nmtf_pg
from cod import nmtf_cod
from als import nmtf_als

def normalize_data(X):
    return X / np.max(X)

def pprint(X):
    for i in range(X.shape[0]):
        s = ''
        for j in range(X.shape[1]):
            s += "%.1f" % X[i,j] + ' '
        print(s)

def main():
    parser = argparse.ArgumentParser(description='fast-nmtf')
    parser.add_argument('-i', '--iterations', type=int, default=10, help="Maximum number of iterations.")
    parser.add_argument('-t', '--technique', default='', help="Optimization technique (mu, cod, als, pg)")
    parser.add_argument('-k', '--k', default='20', help="Factorization rank")
    parser.add_argument('-p', '--parallel', type=int, default=6, help="Number of workers")
    parser.add_argument('-S', '--seed', default='42', help="Random seed")
    
    parser.add_argument('-V', '--verbose', action="store_true", help="Print error function in each iteration")
    parser.add_argument('data', nargs='*', help='Other args')
    
    args = parser.parse_args()
    
    filename = args.data[0]
    data = load_data(filename)
    if data is None:
        raise Exception("Unable to open file: %s" % filename)

    print(type(data))
    X = data
    #X = np.array([[5,0,4,3], [1,3,3,1], [3,3,0,2], [2,3,0,2], [4,3,0,2]])
    #basedata='test'
    
    basedata = os.path.splitext(os.path.basename(filename))[0]
    
    #data = data.todense()
    #print type(data)
    #X = data
    print("Shape", X.shape)
    sparse = False
    if type(data) == csr_matrix:
        sparse = True
    #    X = np.array(X, dtype=dtype, order='C')
        #X = normalize_data(X)
    
    # double
    X = X.astype(np.float64)
    #if args.sparse:
    #    X = csr_matrix(X)
    
    np.random.seed(42)
    max_iter = args.iterations
    
    function_dict = {
        'mu': nmtf_mu,
        'cod': nmtf_cod,
        'als': nmtf_als,
        'pg': nmtf_pg,
    }
    
    method_list = ['nmtf']
    technique_list = args.technique.split(',')
    if args.technique == '':
        technique_list = ['mu', 'cod', 'als', 'pg']
        
    k_list = [int(s) for s in args.k.split(',')]
    seed_list = [int(s) for s in args.seed.split(',')]
    
    if len(k_list) > 1 and len(seed_list) > 1:
        print("Cannot use both multiple K and seed parameters")
        print("Use either K or seed as a list")
        raise Exception("Too many parameters exception")
    
    XX = None
    if type(X) == csr_matrix:
        XX = X.power(2)
        TrX = XX.sum()
    else:
        XX = np.multiply(X, X)
        TrX = np.sum(XX)
    engine = Engine()
    tasks = []
    conv_trace = {}
    for t in technique_list:
        if t not in function_dict:
            print("Technique %s is not available" % t)
            continue
        
        conv_trace[t] = []
        for k in k_list:
            for seed in seed_list:
                params = {'engine': engine, 'X': X, 'k': k, 'k2': k, 'seed': seed, 'method': 'nmtf', 'technique': t, 
                    'max_iter': max_iter, 'verbose': args.verbose, 'store_results': True, 'basename': basedata, 
                    'label': "%s-vanilla" % basedata}
#                factors, hist = function_dict[t](params)
                tasks.append(Task(function_dict[t], params))
    
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
        print("KeyboardInterrupt")
        mt.stop()
    
    mt.join()


if __name__ == '__main__':
    main()
