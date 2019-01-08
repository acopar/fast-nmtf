#!/bin/bash
ANACONDA=Anaconda3-5.3.1-Linux-x86_64.sh
ENVDIR=~/anaconda3
if [ -d $ENVDIR ]; then
    ENVDIR=~/anaconda$RANDOM
fi
wget https://repo.continuum.io/archive/$ANACONDA
bash $ANACONDA -b -p $ENVDIR
echo "source $ENVDIR/etc/profile.d/conda.sh" >> ~/.bashrc
source $ENVDIR/etc/profile.d/conda.sh
