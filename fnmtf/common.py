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

def nprand(x, y, dtype=np.float64, seed=42):
    X = np.random.rand(x, y)
    X = np.array(X, dtype=dtype, order='C')
    return X


def dump_history(params, err_history):
    filename = 'results/%s/%s/%d_%d.csv' % (params['label'], params['technique'], params['k'], params['seed'])
    ensure_dir(filename)
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    for i, h in enumerate(err_history):
        writer.writerow([params['method'], params['technique'], i, h, params['k'], params['seed']])
    fp.close()

def dump_runtime(params, runtime):
    filename = 'results/%s/%s/%d_%d.time' % (params['label'], params['technique'], params['k'], params['seed'])
    ensure_dir(filename)
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    writer.writerow([params['method'], params['technique'], runtime, params['k'], params['seed'], params['max_iter'], params['min_iter'], params['epsilon']])
    fp.close()

def validate_factors(factors):
    for f in factors:
        if np.any(f < 0):
            raise Exception("Assert exception: factor contains negative values")


def tri_factorization(func):
    def new_f(params):
        X = params['X']
        seed = params['seed']
        k = params['k']
        k2 =params['k2']
        max_iter = params['max_iter']
        min_iter = params['min_iter']
        verbose = params['verbose']
        method = params['method']
        engine = params['engine']
        technique = params['technique']
        
        print("Task started: (technique=%s, k=%d, seed=%d)" % (technique, k, seed))
        
        # random initialization
        np.random.seed(seed)
        n, m = X.shape
        U = nprand(n, k, dtype=X.dtype)
        S = nprand(k, k2, dtype=X.dtype)
        V = nprand(m, k2, dtype=X.dtype)
        
        # Calculate norm of input matrix X
        Xt = None
        if type(X) == csr_matrix:
            XX = X.power(2)
            TrX = XX.sum()
            Xt = csr_matrix(X.T)
        else:
            XX = np.multiply(X, X)
            TrX = np.sum(XX)
            Xt = np.array(X.T, order='C')
        
        
        t0 = time.time()
        engine.clean()
        
        # Run factorization
        factors, err_history = func(engine, X, Xt, U, S, V, TrX, k=k, k2=k2, max_iter=max_iter, min_iter=min_iter, verbose=verbose)
        runtime = time.time()-t0
        print("Task (%s, k=%dx%d) finished in:" % (technique, k, k2), runtime)
        
        # Check if any values are below zero
        # If not, non-negativity was not properly enforced
        validate_factors(factors)
        
        # store history and runtime
        if params['store_history']:
            dump_history(params, err_history[1:])
            dump_runtime(params, runtime)
        
        # store factors to use in further processing
        if params['store_results']:
            dump_file('results/%s/%s.pkl' % (params['label'], technique), factors)
            U, S, V = factors
            save_csv('results/%s/%s/U.csv' % (params['label'], technique), U)
            save_csv('results/%s/%s/S.csv' % (params['label'], technique), S)
            save_csv('results/%s/%s/V.csv' % (params['label'], technique), V)
        
        return factors, err_history
    return new_f


class Timer():
    # Timer class for benchmarking purposes
    def __init__(self, system=True):
        self.t = {}
        self.c = {}
        self.last = None
        self.system = system
    
    def time(self):
        if self.system == False:
            return time.time()
        else:
            return os.times()[4]
    
    def get(self, label=None):
        if label not in self.t:
            return None
        else:
            return self.t[label]
    
    def check(self, label=None):
        if label not in self.t or self.t[label] == None:
            self.t[label] = 0.0
    
    def labelize(self, label):
        if label == None:
            if self.last:
                label = self.last
        return label
    
    def clear(self):
        self.t = {}
        self.c = {}
        self.last = None
    
    def reset(self, label=None):
        label = self.labelize(label)
        self.t[label] = 0.0
        self.c[label] = self.time()
    
    def start(self, label=None):
        label = self.labelize(label)
        if label == None:
            return
        self.check(label=label)
        self.c[label] = self.time()
        self.last = label
    
    def pause(self, label=None):
        label = self.labelize(label)
        if label == None:
            return
        if label in self.c or self.c[label] != None:
            self.t[label] += self.time() - self.c[label]
    
    def stop(self, label=None):
        self.pause(label=label)
        if self.last:
            self.last = None
    
    def split(self, label=None):
        label = self.labelize(label)
        if label == None or label == self.last:
            return
        if self.last:
            self.stop(label=self.last)
        self.start(label=label)
        
    def __str__(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        portions = [(key, 100*value/total) for key, value in elements]
        portions = [(key, '%.2f' % value) for key, value in portions]
        return str(portions)

    def add(self, other):
        for key in other.t:
            if key in self.t:
                self.t[key] = self.t[key] + other.t[key]
            else:
                self.t[key] = other.t[key]
    
    def asdict(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        portions = [(key, value/total) for key, value in elements]
        portions = {key: value for key, value in portions}
        return portions

    def elapsed(self):
        return self.t
    
    def total_elapsed(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        return total
