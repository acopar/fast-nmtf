#!/usr/bin/env python

import time

from common import *

@tri_factorization
def nmtf_mu(engine, X, Xt, U, S, V, TrX, k=20, k2=20, max_iter=10, verbose=False):
    err_history = []
    globals().update(engine.methods())
    timer = Timer()
    
    for it in range(max_iter):
        # error
        timer.split('XV')
        XV = bigdot(X, V)
        timer.split('error')
        T2 = dot(XV.T, U)
        T3 = dot(S, T2)
        tr2 = trace(T3)
        UtU = dot(U.T, U)
        VtV = dot(V.T, V)
        UtUS = dot(UtU, S)
        T4 = dot(dot(UtUS, VtV), S.T)
        tr3 = trace(T4)
        E = 1 - (2*tr2 - tr3)/TrX
        err_history.append(E)
        
        if verbose:
            print("Error", E)
        
        timer.split('U')
        NK13 = dot(XV, S.T)
        KM5 = dot(S, V.T)
        KK14 = dot(KM5, KM5.T)
        NK15 = dot(U, KK14)
        NK16 = divide(NK13, NK15)
        NK17 = sqrt(NK16)
        U = multiply(U, NK17)
        
        timer.split('Xt')
        MK19 = bigdot(Xt, U)
        timer.split('V')
        ML20 = dot(MK19, S)
        KK21 = dot(U.T, U)
        KL22 = dot(KK21, S)
        ML23 = dot(KM5.T, KL22)
        ML24 = divide(ML20, ML23)
        ML25 = sqrt(ML24)
        V = multiply(V, ML25)
        
        timer.split('S')
        KL27 = dot(MK19.T, V)
        LL28 = dot(V.T, V)
        KL29 = dot(KL22, LL28)
        KL30 = divide(KL27, KL29)
        KL31 = sqrt(KL30)
        S = multiply(S, KL31)
    
    print("Timer", str(timer))
    factors = U, S, V
    return factors, err_history