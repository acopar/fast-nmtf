# fast-nmtf
Fast methods for non-negative matrix tri-factorization


## Quick Setup ###

```sh
   git clone https://github.com/acopar/fast-nmtf
   cd fast-nmtf
   pip install -r requirements.txt
```


### Datasets

To download preprocessed benchmark datasets, use the provided ``get_datasets.sh`` script.
```
    scripts/get_datasets.sh
```

This script downloads datasets that have already been preprocessed and converted into coordinate list or npz format:
- [aldigs](http://file.biolab.si/fast-nmtf/aldigs.npz)
- [mutations](http://file.biolab.si/fast-nmtf/mutations.npz)
- [newsgroups](http://file.biolab.si/fast-nmtf/newsgroups.npz)
- [movielens](http://file.biolab.si/fast-nmtf/ml-10m.npz):
- [coil20](http://file.biolab.si/fast-nmtf/coil20.npz)
- [Retina](http://file.biolab.si/fast-nmtf/retina.npz)

### Example

```sh
    python factorize.py -t cod -k 20 aldigs.npz
```

The following optimization techniques can be set with option ``-t``.:
- mu: multiplicative updates
- als: alternating least squares
- pg: projected gradient
- cod: coordinate descent

#### Command line arguments

- -t: Optimization technique
- -s: Use sparse matrices
- -k: factorization rank
- -p: parallelization degree
- -S: seed
- data: last argument is path to dataset.

