#!/usr/bin/env python

import time
import numpy as np

from scipy.sparse import csr_matrix, csc_matrix
from common import *

@tri_factorization
def nmtf(X, U, S, V, TrX, k=20, max_iter=10, verbose=False):
    err_history = []
    
    for it in range(max_iter):
        # error
        XV = bigdot(X, V)
        T2 = np.dot(XV.T, U)
        T3 = np.dot(S, T2)
        tr2 = np.trace(T3)
        UtU = np.dot(U.T, U)
        VtV = np.dot(V.T, V)
        T4 = np.dot(np.dot(np.dot(UtU, S), VtV), S.T)
        tr3 = np.trace(T4)
        E = 1 - (2*tr2 - tr3)/TrX
        err_history.append(E)
        
        if verbose:
            print("Error", E)
        
        NK13 = np.dot(XV, S.T)
        KM5 = np.dot(S, V.T)
        KK14 = np.dot(KM5, KM5.T)
        NK15 = np.dot(U, KK14)
        NK16 = divide(NK13, NK15)
        NK17 = np.sqrt(NK16)
        U = np.multiply(U, NK17)
        
        MK19 = bigdot(X.T, U)
        ML20 = np.dot(MK19, S)
        KK21 = np.dot(U.T, U)
        KL22 = np.dot(KK21, S)
        ML23 = np.dot(KM5.T, KL22)
        ML24 = divide(ML20, ML23)
        ML25 = np.sqrt(ML24)
        V = np.multiply(V, ML25)
        
        KL27 = np.dot(MK19.T, V)
        LL28 = np.dot(V.T, V)
        KL29 = np.dot(KL22, LL28)
        KL30 = divide(KL27, KL29)
        KL31 = np.sqrt(KL30)
        S = np.multiply(S, KL31)
        
    factors = U, S, V
    return factors, err_history