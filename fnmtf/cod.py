#!/usr/bin/env python

from fnmtf.common import *

@tri_factorization
def nmtf_cod(engine, X, Xt, U, S, V, TrX, k=20, k2=20, max_iter=10, min_iter=1, verbose=False):
    # This function calculates NMTF with Coordinate descent optimization technique
    # return value:
    # factors: list of numpy arrays in order: [U, S, V]
    # err_history: list of floats
    
    n = X.shape[0]
    m = X.shape[1]
    err_history = []
    globals().update(engine.methods())
    for it in range(max_iter):
        # error
        XV = bigdot(X, V)
        T2 = dot(XV.T, U)
        T3 = dot(S, T2)
        tr2 = np.trace(T3)
        UtU = dot(U.T, U)
        VtV = dot(V.T, V)
        T4 = dot(dot(dot(UtU, S), VtV), S.T)
        tr3 = np.trace(T4)
        E = 1 - (2*tr2 - tr3)/TrX
        err_history.append(E)
        if verbose:
            print("Error", E)
        
        # Update U factor matrix    
        KM5 = dot(S, V.T)
        NK13 = dot(XV, S.T)
        KK14 = dot(KM5, KM5.T)
        MK15 = multiply(KM5.T, KM5.T)
        AK16 = vsum(MK15)
        
        for i in range(k):
            NA30 = dot(U, KK14[:,i])
            NA31 = sub(NK13[:,i], NA30)
            NA32 = divide(NA31, AK16[:,i])
            NA33 = add(U[:,i], NA32)
            project_to(NA33, U, i)
        
        # Update V factor matrix
        MK17 = bigdot(Xt, U)
        ML18 = dot(MK17, S)
        NL19 = dot(U, S)
        LL20 = dot(NL19.T, NL19)
        NL21 = multiply(NL19, NL19)
        AL22 = vsum(NL21)
        
        for i in range(k2):
            MA30 = dot(V, LL20[:,i])
            MA31 = sub(ML18[:,i], MA30)
            MA32 = divide(MA31, AL22[:,i])
            MA33 = add(V[:,i], MA32)
            project_to(MA33, V, i)
        
        # Update S factor matrix
        KL23 = dot(MK17.T, V)
        KK24 = dot(U.T, U)
        LL25 = dot(V.T, V)
        NK26 = multiply(U, U)
        AK27 = vsum(NK26)
        ML28 = multiply(V, V)
        AL29 = vsum(ML28)
        
        cod_s(S, KK24, LL25, KL23, AK27, AL29)
        
        # cod_s is a faster cython implementation of to the following python code
        """
        for i in range(k):
            for j in range(k2):
                AL30 = dot(KK24[i,:].reshape(1,-1), S)
                AA31 = dot(AL30, LL25[:,j])
                AA32 = sub(KL23[i,j], AA31)
                AA33 = multiply(AK27[:,i], AL29[:,j])
                AA34 = divide(AA32, AA33)
                AA35 = add(S[i,j], AA34)
                AA36 = project(AA35)
                S[i,j] = AA36
        """
        
        # Check convergence
        if check_stop(err_history) > 0:
            print("Stopping after %d iterations" % it)
            break
    
    factors = U, S, V
    return factors, err_history
