#!/usr/bin/env python

import os
import csv
import subprocess
import argparse
import numpy as np
from collections import defaultdict

from loader import *
from stop import score_history, score_history2

IMG_DIR = 'img'

def data_from_csv(filename):
    fp = open(filename, 'r')
    reader = csv.reader(fp, delimiter=',')
    data = []
    for line in reader:
        it = int(line[2])
        value = float(line[3])
        data.append(value)
        
    return data

def load_hist(dataset, technique, k, seed):
    filename = "results/%s/%s/%d_%d.csv" % (dataset, technique_map[technique], k, seed)
    if not os.path.isfile(filename):
        print("File missing", filename)
        return None
    hist = data_from_csv(filename)
    return hist

max_iter = 50000
EPSILON=6
MIN_ITER=100

titles = {"aldigs": "AlphaDigit", "mutations": "Mutations", "retina": "Retina", "coil20": "Coil20", 
        "newsgroups": "Newsgroups", "movielens": "MovieLens", "string": "STRING"}


technique_map = {'MUR': 'mu', 'ALS': 'als', 'PG': 'pg', 
    'COD': 'cod'}
    
datasets = ['aldigs', 'coil20', 'string', 'movielens', 'mutations', 'newsgroups']

mmap2 = {'mu': 'MUR', 'als': 'ALS', 'pg': 'PG', 'cod': 'COD'}
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# to include averages over 10 random initializations use the following line:
# seed_list = c[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def stat_convergence(k=20):
    frames = {}
    lines = {}
    best_seed = {}
    best_lines = {}
    
    for dataset in datasets:
        fname = titles[dataset]
        frames[fname] = {}
        lines = {}
        best_seed = {}
        for technique in ['MUR', 'PG', 'COD', 'ALS']:
            technique_name = technique_map[technique]
            
            datapoint = {}
            best = 1.0
            best_seed[technique] = None
            lines[technique] = {}
            score_list = []
            score_list2 = []
            for seed in seed_list:
                hist = load_hist(dataset, technique, k, seed)
                if hist is None:
                    continue
                
                sco10 = score_history2(hist, stop='p10', epsilon=EPSILON)
                if sco10 == -1:
                    print("Experiment did not converge in the specified number of iterations", 
                        (technique, dataset, 'k=%d' % k, 'seed=%d' % seed))
                    print("If this is not an isolated case try increasing max_iter parameter")
                    continue
                #hist = hist[:sco10]
                score_list.append(sco10)
                
                if hist[-1] < best:
                    best = hist[-1]
                    best_seed[technique] = seed
                
                for i, h in enumerate(hist):
                    if i > 0 and i <= max_iter:
                        if i not in datapoint:
                            datapoint[i] = []
                        datapoint[i].append(h)

                lines[technique][seed] = hist[1:]
            frames[fname][technique] = datapoint
            print(dataset, technique, score_list, score_list2)
        
        best_lines[fname] = {}
        for t in best_seed:
            if len(lines[t]) == 0:
                continue
            bs = best_seed[t]
            if bs is None:
                print("Warning best seed is none for technique %s" % t)
                continue
            print(dataset, lines[t].keys())
            best_lines[fname][t] = lines[t][bs]
        
    dnames = [titles[d] for d in datasets]
    
    dump_file('results/visdata/convergence%d.pkl' % k, (dnames, frames, best_lines))

def stat_rank():
    frames = {}

    score = {}
    lines = {}
    klist = range(10, 110, 10)
    for dataset in datasets:
        lines[titles[dataset]] = {}
        score = {}
        seed_list2 = {}
        print(dataset)
        for technique in ['MUR', 'PG', 'COD', 'ALS']:
            technique_name = technique_map[technique]
            print(technique,)
            lines[titles[dataset]][technique] = []
            for k in klist:
                score[technique] = []
                seed_list2[technique] = []
                for seed in seed_list:
                    hist = load_hist(dataset, technique, k, seed)
                    if hist is None:
                        continue
                    if technique not in score:
                        score[technique] = []
                    wait = 0
                    if dataset in ['newsgroups', 'string', 'coil20'] and technique_name == 'mu':
                        wait = MIN_ITER
                    sco10 = score_history2(hist, stop='p10', epsilon=EPSILON, wait=MIN_ITER)
                    if sco10 == -1:
                        print("Experiment did not converge in the specified number of iterations", 
                            (technique, dataset, 'k=%d' % k, 'seed=%d' % seed))
                        print("If this is not an isolated case try increasing max_iter parameter")
                        continue
                    
                    score[technique].append(sco10)
                    seed_list2[technique].append(seed)
                    frames[(technique, k)] = hist

                if technique not in score:
                    print("Missing", dataset, technique)
                avg = np.round(np.mean(score[technique]))
                lines[titles[dataset]][technique].append(avg)
                print('&', avg, k, score[technique], seed_list2[technique])
            print('\\\\')

    dnames = [titles[d] for d in datasets]

    dump_file('results/visdata/rank.pkl', (lines, dnames, klist))


if __name__ == '__main__':
    stat_convergence(k=20)
    stat_rank()
