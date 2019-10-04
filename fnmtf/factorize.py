#!/usr/bin/env python

import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.linalg as la

from scipy.sparse import csr_matrix, csc_matrix
from fnmtf.loader import *
from fnmtf.common import *
from fnmtf.engine import Engine

from fnmtf.mu import nmtf_mu
from fnmtf.pg import nmtf_pg
from fnmtf.cod import nmtf_cod
from fnmtf.als import nmtf_als

def normalize_data(X):
    return X / np.max(X)

def pprint(X):
    for i in range(X.shape[0]):
        s = ''
        for j in range(X.shape[1]):
            s += "%.1f" % X[i,j] + ' '
        print(s)

def main():
    #### Command line arguments

    # -t [arg]: Optimization technique [mu, als, pg, cod]
    # -s: Use sparse matrices
    # -k [arg]: factorization rank, positive integer
    # -p [arg]: number of parallel workers
    # -S [arg]: random seed
    # -e [arg]: stopping criteria threshould (higher means more iterations), default=6
    # -m [arg]: minimum number of iterations
    # data: last argument is path to the dataset (required)
    
    parser = argparse.ArgumentParser(description='fast-nmtf')
    parser.add_argument('-i', '--iterations', type=int, default=20, help="Maximum number of iterations.")
    parser.add_argument('-m', '--min-iter', type=int, default=-1, help="Specify minimum number of iterations.")
    parser.add_argument('-t', '--technique', default='', help="Optimization technique (mu, cod, als, pg)")
    parser.add_argument('-k', '--k', default='20', help="Factorization rank")
    parser.add_argument('-k2', '--k2', default='', help="Factorization rank (column dimension)")
    parser.add_argument('-p', '--parallel', type=int, default=-1, help="Number of MKL threads, max=nproc/2")
    parser.add_argument('-S', '--seed', default='0', help="Random seed")
    parser.add_argument('-e', '--epsilon', default=6, type=int,
        help="Convergence criteria: relative difference in function is less than 10^(-epsilon)")
    
    parser.add_argument('-V', '--verbose', action="store_true", help="Print error function in each iteration")
    parser.add_argument('data', nargs='*', help='Other args')
    
    args = parser.parse_args()
    
    # read and parse data
    filename = args.data[0]
    data = load_data(filename)
    if data is None:
        raise Exception("Unable to open file: %s" % filename)

    X = data
    basedata = os.path.splitext(os.path.basename(filename))[0]
    
    print("Loaded %s, shape" % filename, X.shape)
    sparse = False
    if type(data) == csr_matrix:
        sparse = True

    # double
    X = X.astype(np.float64)
    
    # set global seed
    # additional seed parameter will be used just before the initialization
    np.random.seed(0)
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
    k2_list = None
    if args.k2:
        k2_list = [int(s) for s in args.k2.split(',')]
    seed_list = [int(s) for s in args.seed.split(',')]
    
    # Initialize standard CPU engine
    engine = Engine(epsilon=args.epsilon, parallel=args.parallel)
    
    for t in technique_list:
        if t not in function_dict:
            print("Technique %s is not available" % t)
            continue
        
        # prevent multiplicative updates from stopping too early
        min_iter = args.min_iter
        if min_iter == -1 and t == 'mu':
            min_iter = 100
        elif min_iter == -1:
            min_iter = 1
        
        for ik, k in enumerate(k_list):
            if k2_list == None:
                k2 = k
            else:
                k2 = k2_list[ik]
            for seed in seed_list:
                params = {'engine': engine, 'X': X, 'k': k, 'k2': k2, 
                    'seed': seed, 'method': 'nmtf', 'technique': t, 
                    'max_iter': max_iter, 'min_iter': min_iter, 'epsilon': args.epsilon,
                    'verbose': args.verbose, 'store_history': True, 'store_results': True,
                    'basename': basedata, 'label': "%s" % basedata}
                    
                # Run the factiorization
                # depending on the input, mu, cod, als or pg is used
                function_dict[t](params)


if __name__ == '__main__':
    main()
