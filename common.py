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


MAXILON = 10**(9)
EPSILON = 10**(-9)
dtype = np.float64

def nprand(x, y, seed=42):
    # Use floats for speed
    np.random.seed(seed)
    X = np.random.rand(x, y)
    X = np.array(X, dtype=dtype, order='C')
    print X.shape, X.sum()
    return X

def vsum(A):
    return np.sum(A, axis=0).reshape(1,-1)

def project(A):
    A[np.where(A < 0)] = 0
    A[np.where(A > MAXILON)] = MAXILON
    return A

def inverse(A):
    A = np.nan_to_num(A)
    try:
        X = la.inv(A)
        X = np.nan_to_num(X)
        return X
    except la.LinAlgError:
        print("Warning: singular matrix")
        X = la.pinv(A)
        X = np.nan_to_num(X)
        return X

def divide(A, B):
    if np.isscalar(B):
        if B < EPSILON:
            B = EPSILON
    else:
        B[np.where(B < EPSILON)] = EPSILON
    return np.divide(A, B)


def multiply(A, B):
    if type(A) == np.ndarray:
        D = np.multiply(A, B)
    else:
        D = A.multiply(B)
    return D


def square(A):
    return np.multiply(A, A)

def bigdot(X, Y):
    if type(X) == csr_matrix or type(X) == csc_matrix:
        return X.dot(Y)
    else:
        return np.dot(X, Y)

def dump_history(params, err_history):
    filename = '../results/%s/%s_%s/%d.csv' % (params['basename'], params['method'], params['technique'], params['k'])
    ensure_dir(filename)
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    for i, h in enumerate(err_history):
        writer.writerow([params['method'], params['technique'], i, h, params['k']])
    fp.close()


def tri_factorization(func):
    #def new_f(X, k=20, seed=42, max_iter=10, verbose=False):
    def new_f(params):
        X = params['X']
        seed = params['seed']
        k = params['k']
        max_iter = params['max_iter']
        verbose = params['verbose']
        method = params['method']
        technique = params['technique']
        
        print("Task started: (%s, %s)" % (method, technique))
        np.random.seed(seed)
        n, m = X.shape
        U = nprand(n, k)
        S = nprand(k, k)
        V = nprand(m, k)
        if type(X) == csr_matrix:
            XX = X.power(2)
            TrX = XX.sum()
        else:
            XX = np.multiply(X, X)
            TrX = np.sum(XX)
        t0 = time.time()
        factors, err_history = func(X, U, S, V, TrX, k=k, max_iter=max_iter, verbose=verbose)
        print("Task (%s, %s) finished in:", (method, technique, k, time.time()-t0))
        validate_factors(factors)
        
        if params['store_results']:
            dump_history(params, err_history[1:])
            dump_file('../results/%s/%s_%s.pkl' % (params['basename'], method, technique), factors)
        
        return factors, err_history
    return new_f

