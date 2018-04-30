#!/usr/bin/env python

import time
import numpy as np
import scipy.linalg as la

from common import *

@tri_factorization
def nmtf_als(X, U, S, V, TrX, k=20, max_iter=10, verbose=False):
    err_history = []
    try:
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
            
            KM5 = np.dot(S, V.T)
            NK13 = np.dot(XV, S.T)
            KK14 = np.dot(KM5, KM5.T)
            KK15 = inverse(KK14)
            NK16 = np.dot(NK13, KK15)
            U = project(NK16)
            #U = np.nan_to_num(U)
            
            MK18 = bigdot(X.T, U)
            ML19 = np.dot(MK18, S)
            KK20 = np.dot(U.T, U)
            LK21 = np.dot(S.T, KK20)
            LL22 = np.dot(LK21, S)
            LL23 = inverse(LL22)
            ML24 = np.dot(ML19, LL23)
            V = project(ML24)
            #V = np.nan_to_num(V)
            
            KK26 = inverse(KK20)
            KL27 = np.dot(MK18.T, V)
            LL28 = np.dot(V.T, V)
            LL29 = inverse(LL28)
            KL30 = np.dot(KK26, KL27)
            KL31 = np.dot(KL30, LL29)
            S = project(KL31)
            #S = np.nan_to_num(S)
            #print U.sum(), V.sum(), S.sum()
    
    except ValueError:
        print("Warning: ValueError caught")
    
    factors = U, S, V
    return factors, err_history
