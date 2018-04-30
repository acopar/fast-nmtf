#!/usr/bin/env python

import time
import numpy as np
import scipy.linalg as la

from parser import *
from common import *

@tri_factorization
def nmtf_pg(X, U, S, V, TrX, k=20, max_iter=10, verbose=False):
    err_history = []
    
    print X.sum(), U.sum(), S.sum(), V.sum()
    
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
        
        VtV = np.dot(V.T, V)
        SvvS = np.dot(np.dot(S, VtV), S.T)
        NK16 = np.dot(U, SvvS)
        NK17 = divide(U, NK16)
        NK19 = np.dot(XV, S.T)
        NK5 = np.subtract(U, np.multiply(NK17, NK19))
        
        SUB = np.subtract(NK16, NK19)
        a_top = np.sum(np.multiply(NK5, SUB))
        PtP = np.dot(NK5.T, NK5)
        VtV = np.dot(V.T, V)
        a_bot = np.trace(np.dot(np.dot(S, VtV), np.dot(S.T, PtP)))
        
        #print U.sum(), SvvS.sum(), NK16.sum(), NK19.sum(), SUB.sum()
        
        nu = a_top / a_bot
        NK21 = nu * NK5
        NK22 = np.subtract(U, NK21)
        U = project(NK22)
        
        UtU = np.dot(U.T, U)
        SuuS = np.dot(np.dot(S.T, UtU), S)
        ML22 = np.dot(V, SuuS)
        
        XtU = bigdot(X.T, U)
        XtUS = np.dot(XtU, S)
        
        NL25 = np.dot(U, S)
        ML27 = divide(V, ML22)
        ML6 = np.subtract(V, np.multiply(ML27, XtUS))
        
        SUB = np.subtract(ML22, XtUS)
        a_top = np.sum(np.multiply(ML6, SUB))
        
        PtP = np.dot(ML6.T, ML6)
        UtU = np.dot(U.T, U)
        a_bot = np.trace(np.dot(np.dot(S, PtP), np.dot(S.T, UtU)))
        
        nv = a_top / a_bot
        ML31 = nv * ML6
        ML32 = np.subtract(V, ML31)
        V = project(ML32)


        UtU = np.dot(U.T, U)
        VtV = np.dot(V.T, V)
        
        KL33 = np.dot(np.dot(UtU, S), VtV)
        KL34 = np.dot(XtU.T, V)
        KL35 = divide(S, KL33)
        KL36 = np.multiply(KL35, KL34)
        KL7 = np.subtract(S, KL36)
        
        SUB = np.subtract(KL33, KL34)
        a_top = np.sum(np.multiply(KL7, SUB))
        a_bot = np.trace(np.dot(np.dot(UtU, KL7), np.dot(VtV, KL7.T)))
        
        ns = a_top / a_bot
        KL41 = ns * KL7
        KL42 = np.subtract(S, KL41)
        
        #print XtU.sum(), V.sum(), KL34.sum()
        #print S.sum(), KL33.sum(), KL35.sum()
        #print S.sum(), KL7.sum(), KL36.sum(),  a_top, a_bot
        S = project(KL42)
        #print nv, ML6.sum(), ML31.sum()
        
        #print U.sum(), V.sum(), S.sum()
        
    factors = U, S, V
    return factors, err_history

