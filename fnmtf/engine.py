import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
from types import FunctionType
from ctypes import *
import ctypes

from loops import Sloop, cproject, cproject64, cproject_to64
from fnmtf.common import Timer
from fnmtf.stop import *

EPSILON = np.finfo(np.float64).eps
MAXILON = 10**(9)

class Engine():
    # This is a wrapper class for numpy and matrix operations
    
    def __init__(self, stop='p10', epsilon=6, parallel=-1):
        self.stop = stop
        self.epsilon = epsilon
        self.profile = False
    
    def clean(self):
        pass

    def check_stop(self, history):
        return score_history(history, stop=self.stop, epsilon=self.epsilon)

    def bigdot(self, X, Y):
        if type(X) == csr_matrix or type(X) == csc_matrix:
            return X.dot(Y)
        else:
            return np.dot(X, Y)

    def dot(self, X, Y):
        return np.dot(X, Y)
    
    def add(self, X, Y):
        return np.add(X, Y)
    
    def sub(self, X, Y):
        return np.subtract(X, Y)
    
    def multiply(self, X, Y):
        if type(X) == np.ndarray:
            D = np.multiply(X, Y)
        else:
            D = X.multiply(Y)
        return D
    
    def divide(self, X, Y):
        if np.isscalar(Y):
            if Y < EPSILON:
                Y = EPSILON
        else:
            Y[np.where(Y < EPSILON)] = EPSILON
        return np.divide(X, Y)

    def trace(self, X):
        return np.trace(X)
    
    def inverse(self, A):
        A = np.nan_to_num(A)
        return la.pinv(A)

    def vsum(self, A):
        return np.sum(A, axis=0).reshape(1,-1)
    
    def project(self, A):
        if A.dtype == np.float64:
            cproject64(A, A)
        else:
            cproject(A, A)
        return A
    
    def project_to(self, X, Y, i):
        cproject_to64(i, X, Y)
    
    def square(self, A):
        return self.multiply(A, A)
    
    def sqrt(self, X):
        return np.sqrt(X)
    
    def norm1(self, X):
        return np.sum(X)
    
    def cod_u(self, U, KK14, NK13, AK16):
        Uloop(U, KK14, NK13, AK16)
        k = U.shape[1]
        
    def cod_v(self, V, LL20, ML18, AL22):
        m = V.shape[0]
        k = V.shape[1]
        Vloop(m, k, V, LL20, ML18, AL22, EPSILON)
    
    def cod_s(self, S, KK24, LL25, KL23, AK27, AL29):
        Sloop(S, KK24, LL25, KL23, AK27, AL29)

    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}
