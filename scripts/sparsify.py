#!/usr/bin/env python

import os
import sys
import csv

from loader import *

def main(args):
    src = args[0]
    dst = args[1]
    
    data = load_data(src)
    X = csr_matrix(data)
    save_data(dst, X)    

if __name__ == '__main__':
    main(sys.argv[1:])
