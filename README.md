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

To run the experiment, where each dataset is run on each of the optimization techniques, run the following commands. This may take hours or even days depending on your configuration.

```sh
    bash scripts/full.sh
```

There is a shorter version of the experiments, which has lower threshould for convergence (epsilon=10^-5), max iterations set to 2000.

```sh
   bash scripts/short.sh
```

After the experiments are done, you can visualize the output using the following two commands:

```sh
    python fnmtf/analyze.py
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

