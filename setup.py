#!/usr/bin/env python

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

MAJOR               = 0
MINOR               = 1
MICRO               = 0
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

reqs = []
if os.path.isfile('requirements.txt'):
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    reqs.extend(required)

params = {}

extensions = [Extension("loops", ["c/loops.pyx"], 
              #extra_compile_args=['-ffast-math'],
              #extra_link_args=['-O3'],
              include_dirs=[np.get_include()])]

setup(name='fast-nmtf',
    version=VERSION,
    description='Fast methods for non-negative matrix tri-factorization.',
    url='http://github.com/acopar/fast-nmtf',
    author='Andrej Copar',
    license='LGPL',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    zip_safe=False,
    install_requires=reqs,
    **params
)
