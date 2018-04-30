#!/usr/bin/env python

import time
import numpy as np
import scipy.linalg as la
from common import *


@tri_factorization
def nmtf_cod(X, U, S, V, TrX, k=20, max_iter=10, verbose=False):
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
        
        KM5 = np.dot(S, V.T)
        NK13 = np.dot(XV, S.T)
        KK14 = np.dot(KM5, KM5.T)
        MK15 = np.multiply(KM5.T, KM5.T)
        AK16 = vsum(MK15)
        
        for i in range(k):
            NA30 = np.dot(U, KK14[:,i])
            NA31 = np.subtract(NK13[:,i], NA30)
            NA32 = divide(NA31, AK16[:,i])
            NA33 = np.add(U[:,i], NA32)
            NA34 = project(NA33)
            U[:,i] = NA34.ravel()
        
        MK17 = bigdot(X.T, U)
        ML18 = np.dot(MK17, S)
        NL19 = np.dot(U, S)
        LL20 = np.dot(NL19.T, NL19)
        NL21 = np.multiply(NL19, NL19)
        AL22 = vsum(NL21)
        
        for i in range(k):
            MA30 = np.dot(V, LL20[:,i])
            MA31 = np.subtract(ML18[:,i], MA30)
            MA32 = divide(MA31, AL22[:,i])
            MA33 = np.add(V[:,i], MA32)
            MA34 = project(MA33)
            V[:,i] = MA34.ravel()
        
        KL23 = np.dot(MK17.T, V)
        KK24 = np.dot(U.T, U)
        LL25 = np.dot(V.T, V)
        NK26 = np.multiply(U, U)
        AK27 = vsum(NK26)
        ML28 = np.multiply(V, V)
        AL29 = vsum(ML28)
        
        for i in range(k):
            for j in range(k):
                AL30 = np.dot(KK24[i,:], S)
                AA31 = np.dot(AL30, LL25[:,j])
                AA32 = np.subtract(KL23[i,j], AA31)
                AA33 = np.multiply(AK27[:,i], AL29[:,j])
                AA34 = divide(AA32, AA33)
                AA35 = np.add(S[i,j], AA34)
                AA36 = project(AA35)
                S[i,j] = AA36
    
    factors = U, S, V
    validate_factors(factors)
    return factors, err_history
