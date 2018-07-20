import sys
import time
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.linalg as la
from types import FunctionType
from ctypes import *
import ctypes

from loops import Sloop, cproject, cproject64, cproject_to64
from common import Timer

EPSILON = np.finfo(np.float64).eps
MAXILON = 10**(9)

class Engine():
    def __init__(self):
        self.x = 0
        self.operations = 0
        self.soperations = 0
        self.timer = Timer()
        #mkl = cdll.LoadLibrary("libmkl_rt.so")
        #mkl.mkl_set_num_threads(ctypes.byref(ctypes.c_int(12)))
        #print("Threads", mkl.mkl_get_max_threads())
    
    def clean(self):
        self.operations = 0
        self.soperations = 0

    def bigdot(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if type(X) == csr_matrix or type(X) == csc_matrix:
            self.operations += X.nnz * Y.shape[1]
            return X.dot(Y)
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
            return np.dot(X, Y)

    def dot(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(Y.shape) == 1:
            self.operations += X.shape[0] * X.shape[1]
        else:
            self.operations += X.shape[0] * X.shape[1] * Y.shape[1]
        return np.dot(X, Y)
    
    def add(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return np.add(X, Y)
    
    def sub(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        return np.subtract(X, Y)
    
    def multiply(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        
        if type(X) == np.ndarray:
            D = np.multiply(X, Y)
        else:
            D = X.multiply(Y)
        return D
    
    def divide(self, X, Y):
        self.timer.split(sys._getframe().f_code.co_name)
        if len(X.shape) == 0:
            self.soperations += 1
        elif len(X.shape) == 1:
            self.soperations += X.shape[0]
        else:
            self.soperations += X.shape[0] * X.shape[1]
        if np.isscalar(Y):
            if Y < EPSILON:
                Y = EPSILON
        else:
            Y[np.where(Y < EPSILON)] = EPSILON
        return np.divide(X, Y)

    def trace(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return np.trace(X)
    
    def inverse(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        A = np.nan_to_num(A)
        try:
            X = la.inv(A)
            X = np.nan_to_num(X)
            return X
        except la.LinAlgError:
            #print("Warning: singular matrix")
            X = la.pinv(A)
            X = np.nan_to_num(X)
            return X

    def vsum(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += A.shape[0] * A.shape[1]
        return np.sum(A, axis=0).reshape(1,-1)
    
    def project(self, A):
        self.timer.split(sys._getframe().f_code.co_name)
        if A.dtype == np.float64:
            cproject64(A, A)
        else:
            cproject(A, A)
        return A
    
    def project_to(self, X, Y, i):
        self.timer.split(sys._getframe().f_code.co_name)
        cproject_to64(i, X, Y)
    
    def square(self, A):
        return self.multiply(A, A)
    
    def sqrt(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        self.soperations += X.shape[0] * X.shape[1]
        return np.sqrt(X)
    
    def norm1(self, X):
        self.timer.split(sys._getframe().f_code.co_name)
        return np.sum(X)
    
    def cod_u(self, U, KK14, NK13, AK16):
        self.timer.split(sys._getframe().f_code.co_name)
        Uloop(U, KK14, NK13, AK16)
        k = U.shape[1]
        #self.operations += U.shape[0] * k * k
        
    def cod_v(self, V, LL20, ML18, AL22):
        self.timer.split(sys._getframe().f_code.co_name)
        m = V.shape[0]
        k = V.shape[1]
        Vloop(m, k, V, LL20, ML18, AL22, EPSILON)
        #self.operations += V.shape[0] * k * k
    
    def cod_s(self, S, KK24, LL25, KL23, AK27, AL29):
        self.timer.split(sys._getframe().f_code.co_name)
        Sloop(S, KK24, LL25, KL23, AK27, AL29)
        k = S.shape[0]
        l = S.shape[1]
        self.operations += k * l * k + k * l * (l + 3)
    
    def methods(self):
        return {func: getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")}