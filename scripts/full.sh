#!/bin/bash
for dataset in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 10,20,30,40,50,60,70,80,90,100 data/$f
done
