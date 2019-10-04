# fast-nmtf
Fast optimization of non-negative matrix tri-factorization. 


## Installation ###

This project relies on numpy and scipy libraries. For best results, we recommend installing it inside the [Anaconda environment](https://www.anaconda.com/download/#linux). Anaconda simplifies the environment setup by providing optimized libraries for matrix operations (such as Intel MKL). 

```sh
   git clone https://github.com/acopar/fast-nmtf
   cd fast-nmtf
   conda env create -f environment.yml
   conda activate fast-nmtf
   pip install -e .
```

### Data ###

To download preprocessed benchmark datasets, use the provided ``get_datasets.sh`` script.
```
    scripts/get_datasets.sh
```

This script downloads datasets that have already been preprocessed and converted into npz (numpy compressed) format:
- [aldigs](http://file.biolab.si/fast-nmtf/aldigs.npz)
- [mutations](http://file.biolab.si/fast-nmtf/mutations.npz)
- [newsgroups](http://file.biolab.si/fast-nmtf/newsgroups.npz)
- [movielens](http://file.biolab.si/fast-nmtf/ml-10m.npz):
- [coil20](http://file.biolab.si/fast-nmtf/coil20.npz)
- [string](http://file.biolab.si/fast-nmtf/string.npz)

### Example ###

```sh
    python fnmtf/factorize.py -t cod -k 20 data/aldigs.npz
```

The following optimization techniques can be set with option ``-t``.:
- mu: multiplicative updates
- als: alternating least squares
- pg: projected gradient
- cod: coordinate descent

### Reproduce results ###

To exactly reproduce the experiments, where each dataset is run ten times on each of the optimization techniques, run the following command. This may take days depending on your configuration.

```sh
    bash scripts/full.sh
```

Long test will evaluate convergence (using the same factorization rank=20). This will take hours to complete (less than 10 times faster compared to full test).

```sh
    bash scripts/long.sh
```

There is a shorter version of the experiments, which has a lower threshould for convergence (epsilon=10^-5), max iterations set to 2000. This test will complete in a few hours.

```sh
    bash scripts/short.sh
```

After the experiments are done, you can visualize the output using the following two commands:

```sh
    python fnmtf/visualize.py
```

#### Command line arguments

- -t [arg]: Optimization technique [mu, als, pg, cod]
- -s: Use sparse matrices
- -k [arg]: factorization rank, positive integer
- -p [arg]: number of parallel workers
- -S [arg]: random seed
- -e [arg]: stopping criteria threshould (higher means more iterations), default=6
- -m [arg]: minimum number of iterations
- data: last argument is path to the dataset (required)

### Retrieve factors ###

After the factorization is finished, U, S, and V factors are stored in ``results/<dataset>/<technique>/<factor>.csv``. For example, if you selected ``cod`` technique, the results can be viewed using the following commands, where U is left factor, S is middle factor and V is right factor.

```sh
    cat results/aldigs/cod/U.csv
    cat results/aldigs/cod/S.csv
    cat results/aldigs/cod/V.csv
```

For convenience, all three factors are also saved in ``results/aldigs/cod.pkl`` as a tuple of numpy matrices and can be loaded with ``load_file`` function provided in ``loader.py``.
