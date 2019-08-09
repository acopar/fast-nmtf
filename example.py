#!/usr/bin/env python

import numpy as np

from fnmtf.loader import load_numpy, save_numpy
from fnmtf.engine import Engine
from fnmtf.common import *
from fnmtf.cod import nmtf_cod

X = load_numpy('data/aldigs.npz')
if X is None:
    raise Exception("Unable to open file")
X = X.astype(np.float64)

epsilon = 6
engine = Engine(epsilon=epsilon, parallel=1)
params = {'engine': engine, 'X': X, 'k': 20, 'k2': 20, 'seed': 0, 'method': 'nmtf',
	'technique': 'cod', 'max_iter': 100, 'min_iter': 1, 'epsilon': epsilon,
	'verbose': False, 'store_history': True, 'store_results': False,
	'basename': 'aldigs', 'label': 'aldigs'}

factors, err = nmtf_cod(params)
U, S, V = factors
save_csv('U.csv', U)
save_csv('V.csv', V)
save_csv('S.csv', S)
print("Reconstruction error:", err[-1])

