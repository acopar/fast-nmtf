#!/usr/bin/env python

from fnmtf.common import *

@tri_factorization
def nmtf_mu(engine, X, Xt, U, S, V, TrX, k=20, k2=20, max_iter=10, min_iter=1, verbose=False):
    # This function calculates NMTF with Multiplicative update rules
    # return value:
    # factors: list of numpy arrays in order: [U, S, V]
    # err_history: list of floats
    
    err_history = []
    globals().update(engine.methods())

    for it in range(max_iter):
        # error
        XV = bigdot(X, V)
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
        
        # Update U factor matrix
        NK13 = dot(XV, S.T)
        KM5 = dot(S, V.T)
        KK14 = dot(KM5, KM5.T)
        NK15 = dot(U, KK14)
        NK16 = divide(NK13, NK15)
        U = multiply(U, NK16)
        
        # Update V factor matrix
        MK19 = bigdot(Xt, U)
        ML20 = dot(MK19, S)
        KK21 = dot(U.T, U)
        KL22 = dot(KK21, S)
        ML23 = dot(KM5.T, KL22)
        ML24 = divide(ML20, ML23)
        V = multiply(V, ML24)
        
        # Update S factor matrix
        KL27 = dot(MK19.T, V)
        LL28 = dot(V.T, V)
        KL29 = dot(KL22, LL28)
        KL30 = divide(KL27, KL29)
        S = multiply(S, KL30)
        
        # check convergence
        # multiplicative updates minimum number of iterations parameter
        if it > min_iter and check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    factors = U, S, V
    return factors, err_history