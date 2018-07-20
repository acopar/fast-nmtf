import numpy as np
cimport numpy as np
np.import_array()
cdef double EPSILON = np.finfo(np.float64).eps
cdef double MAX = np.finfo(np.float64).max
from cython.parallel import prange, parallel
from cython import boundscheck, wraparound, nonecheck, cdivision, profile


def Sloop(np.ndarray[np.float64_t, ndim=2] S, 
    np.ndarray[np.float64_t, ndim=2] KK24, 
    np.ndarray[np.float64_t, ndim=2] LL25, 
    np.ndarray[np.float64_t, ndim=2] KL23,
    np.ndarray[np.float64_t, ndim=2] AK27,
    np.ndarray[np.float64_t, ndim=2] AL29):
    
    cdef Py_ssize_t i, j, ii, jj
    cdef Py_ssize_t k = S.shape[0]
    cdef Py_ssize_t l = S.shape[1]
    cdef np.float64_t AA31, AA32, AA33, AA34, AA35, AA36
    cdef np.ndarray[np.float64_t, ndim=2] AL30 = np.zeros([1, l], dtype=np.float64)
    for i in range(k):
        for jj in range(l):
            AL30[0,jj] = 0
            for ii in range(k):
                AL30[0,jj] += KK24[i,ii] * S[ii,jj]
        for j in range(l):
            AA31 = 0
            for jj in range(l):
                AA31 += AL30[0,jj] * LL25[jj,j]
            AA32 = KL23[i,j] - AA31
            AA33 = AK27[0,i] * AL29[0,j]
            if AA33 < EPSILON:
                AA33 = EPSILON
            AA34 = AA32 / AA33
            AA35 = S[i,j] + AA34
            if AA35 < 0:
                AA36 = 0
            else:
                AA36 = AA35
            
            AL30[0,j] -= KK24[i,i] * S[i,j]
            S[i,j] = AA36
            AL30[0,j] += KK24[i,i] * S[i,j]
    return S

def cdivide64(np.ndarray[np.float64_t, ndim=2] A,
    np.ndarray[np.float64_t, ndim=2] B,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < 0.000000001:
                C[i,j] = 0.000000001
            if B[i,j] < 0.000000001:
                C[i,j] = A[i,j] / 0.000000001
            else:
                C[i,j] = A[i,j] / B[i,j]
            if C[i,j] > 1000000000:
                C[i,j] = 1000000000
    return C

def cdivide(np.ndarray[np.float32_t, ndim=2] A,
    np.ndarray[np.float32_t, ndim=2] B,
    np.ndarray[np.float32_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < 0.000000001:
                C[i,j] = 0.000000001
            if B[i,j] < 0.000000001:
                C[i,j] = A[i,j] / 0.000000001
            else:
                C[i,j] = A[i,j] / B[i,j]
            if C[i,j] > 1000000000:
                C[i,j] = 1000000000
    return C

def cdivide_scalar64(np.ndarray[np.float64_t, ndim=2] A,
    np.float64_t B,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < 0.000000001:
                C[i,j] = 0.000000001
            elif B < 0.000000001:
                C[i,j] = A[i,j] / 0.000000001
            else:
                C[i,j] = A[i,j] / B
            
            if C[i,j] > 1000000000:
                C[i,j] = 1000000000
    return C

def cdivide_scalar(np.ndarray[np.float32_t, ndim=2] A,
    np.float32_t B,
    np.ndarray[np.float32_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < 0.000000001:
                C[i,j] = 0.000000001
            elif B < 0.000000001:
                C[i,j] = A[i,j] / 0.000000001
            else:
                C[i,j] = A[i,j] / B
            
            if C[i,j] > 1000000000:
                C[i,j] = 1000000000
    return C

def cmultiply64(np.ndarray[np.float64_t, ndim=2] A,
    np.ndarray[np.float64_t, ndim=2] B,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,j] = A[i,j] * B[i,j]
    return C

def cmultiply(np.ndarray[np.float32_t, ndim=2] A,
    np.ndarray[np.float32_t, ndim=2] B,
    np.ndarray[np.float32_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i,j] = A[i,j] * B[i,j]
    return C

def cmultiply_scalar64(np.ndarray[np.float64_t, ndim=2] A,
    np.float64_t B,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if B < 0.000000001:
                C[i,j] = A[i,j] * 0.000000001
            else:
                C[i,j] = A[i,j] * B
    return C

def cmultiply_scalar(np.ndarray[np.float32_t, ndim=2] A,
    np.float32_t B,
    np.ndarray[np.float32_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if B < 0.000000001:
                C[i,j] = A[i,j] * 0.000000001
            else:
                C[i,j] = A[i,j] * B
    return C
    
def cproject64(np.ndarray[np.float64_t, ndim=2] A,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    #cdef np.float64_t EPSILON = 10**(-9)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < EPSILON:
                C[i,j] = EPSILON
            elif A[i,j] > MAX:
                C[i,j] = MAX
            else:
                C[i,j] = A[i,j]
    return C

def cproject64_vector(np.ndarray[np.float64_t, ndim=1] A,
    np.ndarray[np.float64_t, ndim=1] C):
    cdef Py_ssize_t i, j
    #cdef np.float64_t EPSILON = 10**(-9)
    for i in range(A.shape[0]):
        break
        if A[i] < EPSILON:
            C[i] = EPSILON
        elif A[i] > MAX:
            C[i] = MAX
        else:
            C[i] = A[i]
    return C

def cproject(np.ndarray[np.float32_t, ndim=2] A,
    np.ndarray[np.float32_t, ndim=2] C):
    cdef Py_ssize_t i, j
    #cdef np.float32_t EPSILON = 10**(-9)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] < EPSILON:
                C[i,j] = EPSILON
            elif A[i,j] > MAX:
                C[i,j] = MAX
            else:
                C[i,j] = A[i,j]
    return C
    
    
def cproject_to64(int k, np.ndarray[np.float64_t, ndim=1] A,
    np.ndarray[np.float64_t, ndim=2] C):
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        if A[i] < EPSILON:
            C[i,k] = EPSILON
        else:
            C[i,k] = A[i]
    return C

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cpdef int cdot(double [:,:] A,
    double [:,:] B,
    double [:,:] C) nogil:
    cdef Py_ssize_t i, j
    cdef double T = 0
    
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            T = 0
            for k in range(A.shape[1]):
                T += A[i,k] * B[k,j]
            C[i,j] = T
    
