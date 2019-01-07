#!/bin/bash
for dataset in aldigs.npz coil20.npz string.npz movielens.npz mutations.npz newsgroups.npz; do
    python fnmtf/factorize.py -t mu,als,pg,cod -k 20 -e 5 -i 2000 data/$f
done
