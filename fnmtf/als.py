#!/usr/bin/env python

from fnmtf.common import *


@tri_factorization
def nmtf_als(engine, X, Xt, U, S, V, TrX, k=20, k2=20, max_iter=10, min_iter=1, verbose=False):
    # This function calculates NMTF with Alternating least squares optimization technique
    # return value:
    # factors: list of numpy arrays in order: [U, S, V]
    # err_history: list of floats
    
    err_history = []
    globals().update(engine.methods())
    try:
        for it in range(max_iter):
            # error
            XV = bigdot(X, V)
            
            # Calculate objective function
            T2 = dot(XV.T, U)
            T3 = dot(S, T2)
            tr2 = trace(T3)
            UtU = dot(U.T, U)
            VtV = dot(V.T, V)
            T4 = dot(dot(dot(UtU, S), VtV), S.T)
            tr3 = trace(T4)
            E = 1 - (2*tr2 - tr3)/TrX
            err_history.append(E)
            if verbose:
                print("Error", E)
            
            # Update U factor matrix
            KM5 = dot(S, V.T)
            NK13 = dot(XV, S.T)
            KK14 = dot(KM5, KM5.T)
            KK15 = inverse(KK14)
            NK16 = dot(NK13, KK15)
            U = project(NK16)
            
            # Update V factor matrix
            MK18 = bigdot(Xt, U)
            ML19 = dot(MK18, S)
            KK20 = dot(U.T, U)
            LK21 = dot(S.T, KK20)
            LL22 = dot(LK21, S)
            LL23 = inverse(LL22)
            ML24 = dot(ML19, LL23)
            V = project(ML24)
        
            # Update S factor matrix    
            KK26 = inverse(KK20)
            KL27 = dot(MK18.T, V)
            LL28 = dot(V.T, V)
            LL29 = inverse(LL28)
            KL30 = dot(KK26, KL27)
            KL31 = dot(KL30, LL29)
            S = project(KL31)
            
            if check_stop(err_history) > 0:
                print("Stopping after %d iterations" % it)
                break
        
    except ValueError:
        print("Warning: ValueError caught")
    
    factors = U, S, V
    return factors, err_history
