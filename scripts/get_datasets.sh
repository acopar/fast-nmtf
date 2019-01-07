#!/bin/bash

if [ ! -d data ]; then
    mkdir data
fi

for file in aldigs.npz mutations.npz newsgroups.npz movielens.npz coil20.npz string.npz; do
    if [ -f data/$file ]; then
        echo "File $file has already been downloaded"
        continue
    fi
    wget http://file.biolab.si/fast-nmtf/$file -O data/$file
done
