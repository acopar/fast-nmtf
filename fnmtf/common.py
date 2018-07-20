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
from loader import *
from worker import *

def nprand(x, y, dtype=np.float64, seed=42):
    # Use floats for speed
    np.random.seed(seed)
    X = np.random.rand(x, y)
    X = np.array(X, dtype=dtype, order='C')
    #print X.shape, X.sum()
    return X


def dump_history(params, err_history):
    filename = '../results/%s/%s_%s/%d.csv' % (params['label'], params['method'], params['technique'], params['k'])
    ensure_dir(filename)
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    for i, h in enumerate(err_history):
        writer.writerow([params['method'], params['technique'], i, h, params['k']])
    fp.close()

def validate_factors(factors):
    for f in factors:
        if np.any(f < 0):
            raise Exception("Assert exception: factor contains negative values")


def tri_factorization(func):
    #def new_f(X, k=20, seed=42, max_iter=10, verbose=False):
    def new_f(params):
        X = params['X']
        seed = params['seed']
        k = params['k']
        k2 =params['k2']
        max_iter = params['max_iter']
        verbose = params['verbose']
        method = params['method']
        engine = params['engine']
        technique = params['technique']
        
        print("Task started: (%s, %s)" % (method, technique))
        np.random.seed(seed)
        n, m = X.shape
        U = nprand(n, k, dtype=X.dtype)
        S = nprand(k, k2, dtype=X.dtype)
        V = nprand(m, k2, dtype=X.dtype)
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
        #locals().update(engine.methods())
        #globals().update(engine.methods())
        engine.clean()
        
        factors, err_history = func(engine, X, Xt, U, S, V, TrX, k=k, k2=k2, max_iter=max_iter, verbose=verbose)
        print("Task (%s) finished in:", (technique, k, time.time()-t0))
        print("Engine Mflops/iteration:", float(engine.operations/max_iter)/1000000, float(engine.soperations/max_iter)/1000000)
        print("Engine timer:", str(engine.timer))
        validate_factors(factors)
        
        if params['store_results']:
            dump_history(params, err_history[1:])
            dump_file('../results/%s/%s.pkl' % (params['label'], technique), factors)
        
        return factors, err_history
    return new_f


class Timer():
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