#!/bin/bash
# full test, all parameters. May take days to compute
for d in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 10,20,30,40,50,60,70,80,90,100 -S 0,1,2,3,4,5,6,7,8,9 -i 50000 data/$d
done
