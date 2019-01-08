#!/bin/bash
# convergence visualization; only test for k=20
# lower the convergence threshold, lower number of iterations
# this test will take a few hours
for d in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 20 -S 0,1,2,3,4,5,6,7,8,9 -e 5 -i 2000 data/$d
done
