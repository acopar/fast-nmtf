#!/bin/bash
# full test, all parameters. May take days to compute
for d in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 10,20,30,40,50,60,70,80,90,100 -S 0 -e 5 -i 2000 data/$d
done
