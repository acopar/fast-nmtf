#!/bin/bash
# convergence visualization; only test for k=20
# this test takes many hours
for d in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 20 -S 0,1,2,3,4,5,6,7,8,9 -i 50000 data/$d
done
