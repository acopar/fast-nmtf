#!/usr/bin/env python

import os
import csv
import subprocess
import argparse
import numpy as np
from collections import defaultdict

from loader import *
from stop import score_history, score_history2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import ticker


### STATIC VARIABLES ###

IMG_DIR = 'img'
USE_LATEX = False


max_iter = 50000
EPSILON=6
MIN_ITER=100

titles = {"aldigs": "AlphaDigit", "mutations": "Mutations", "coil20": "Coil20", 
        "newsgroups": "Newsgroups", "movielens": "MovieLens", "string": "STRING"}


technique_map = {'MUR': 'mu', 'ALS': 'als', 'PG': 'pg', 'COD': 'cod'}
    
DATASETS = ['aldigs', 'coil20', 'string', 'movielens', 'mutations', 'newsgroups']

mmap2 = {'mu': 'MUR', 'als': 'ALS', 'pg': 'PG', 'cod': 'COD'}

# to include averages over 10 random initializations use the following line:
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    
# list of optimization techniques
# keep ALS at the end to prevent influence of ALS on y limits
TECHNIQUES = ['MUR', 'PG', 'COD', 'ALS']


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
    if hist is None:
        return None, -1
    # set minimum number of iterations for multiplicative updates n_start parameter
    wait = 0
    if dataset in ['newsgroups', 'string', 'coil20'] and technique == 'MUR':
        wait = MIN_ITER
    sco10 = score_history(hist, stop='p10', epsilon=EPSILON, wait=MIN_ITER)
    return hist, sco10


def stat_convergence(k=20):
    frames = defaultdict(dict)
    best_lines = defaultdict(dict)
    
    for dataset in DATASETS:
        fname = titles[dataset]
        lines = defaultdict(dict)
        best_seed = defaultdict(dict)
        
        for technique in TECHNIQUES:
            datapoint = defaultdict(list)
            best = 1.0
            best_seed[technique] = None
            score_list = []
            for seed in seed_list:
                hist, sco10 = load_hist(dataset, technique, k, seed)
                if hist is None:
                    continue
                
                if sco10 == -1:
                    print("Experiment did not converge in the specified number of iterations", 
                        (technique, dataset, 'k=%d' % k, 'seed=%d' % seed))
                    print("If this is not an isolated case try increasing max_iter parameter")
                    continue
                
                score_list.append(sco10)
                
                if hist[-1] < best:
                    best = hist[-1]
                    best_seed[technique] = seed
                
                for i, h in enumerate(hist):
                    if i > 0 and i <= max_iter:
                        datapoint[i].append(h)

                lines[technique][seed] = hist[1:]
            frames[fname][technique] = datapoint
            print(dataset, technique, score_list)
        
        best_lines[fname] = {}
        for t in best_seed:
            if len(lines[t]) == 0:
                continue
            bs = best_seed[t]
            if bs is None:
                print("Warning best seed is none for technique %s" % t)
                continue
            best_lines[fname][t] = lines[t][bs]

    dump_file('results/visdata/convergence%d.pkl' % k, (frames, best_lines))

def stat_rank(printlatex=True):
    all_lines = {}
    KLIST = range(10, 110, 10)
    
    for dataset in DATASETS:
        lines = defaultdict(list)
        for technique in TECHNIQUES:
            for k in KLIST:
                score = []
                for seed in seed_list:
                    hist, sco10 = load_hist(dataset, technique, k, seed)
                    if hist is None:
                        continue
                    
                    if sco10 == -1:
                        print("Experiment did not converge in the specified number of iterations", 
                            (technique, dataset, 'k=%d' % k, 'seed=%d' % seed))
                        print("If this is not an isolated case try increasing max_iter parameter")
                        continue
                    
                    score.append(sco10)

                avg = np.round(np.mean(score))
                lines[technique].append(avg)

        all_lines[titles[dataset]] = lines
    dump_file('results/visdata/rank.pkl', (all_lines, KLIST))


def get_span(dataframe):
    ylim0 = 1.0
    ylim1 = 0.0
    for technique in TECHNIQUES:
        _maxX = 0
        
        lasts = []
        if technique not in dataframe:
            continue
        
        data = dataframe[technique]
        if len(data) == 0:
            continue
        
        x = sorted(data.keys())
        mx = max(x)
        if mx > _maxX:
            _maxX = mx
            
        if technique != 'ALS':
            for i in x:
                dat = data[i]
                _min = min(dat)
                _max = max(dat)
                if _min < ylim0:
                    ylim0 = _min 
                if _max > ylim1:
                    ylim1 = _max
        else:
            for i in x:
                dat = data[i]
                data[i] = [tt if tt < 1.0 else 1.0 for tt in data[i] ]
    return ylim0, ylim1

def plot_convergence(frames, best_lines, axes, filename=None, 
    xlabel='Iteration', ylabel='Objective value'):
    datasets = [titles[d] for d in DATASETS]
    # legend handles
    handles = []
    
    axind = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    colors = {'MUR': 'b', 'PG': 'g', 'ALS': 'y', 'COD': 'r'}
    
    for i, dataname in enumerate(datasets):
        I, J = axind[i]
        ax = axes[I,J]

        #if dataname in ['Mutations', 'MovieLens']:
        #    handles = []
        
        # calculate the range of objective function
        ylim0, ylim1 = get_span(frames[dataname])
            
        # set the limits so that lines are in the span 10%-90% of the height
        ax.set_ylim([ylim0-(ylim1-ylim0)*0.1, ylim1+(ylim1-ylim0)*0.1])
        
        for technique in TECHNIQUES:
            _maxX = 0
            
            lasts = []
            if technique not in frames[dataname]:
                continue
            data = frames[dataname][technique]
            print(dataname, technique, len(data))
            if len(data) == 0:
                continue
            
            x = sorted(data.keys())
            mx = max(x)
            if mx > _maxX:
                _maxX = mx
                
            last = x[-1]
            lasts.append(np.max(data[last]))
            CIS0 = []
            CIS1 = []
            EST = best_lines[dataname][technique]
            for j in x:
                est = np.mean(data[j])
                sd = np.std(data[j])
                mins = np.min(data[j])
                maxs = np.max(data[j])
                CIS0.append(mins)
                CIS1.append(maxs)

            ax.fill_between(x,CIS0,CIS1,alpha=0.2, color=colors[technique])
            hand, = ax.plot(range(1,len(EST)+1),EST,label=technique, color=colors[technique])
            if i == 0:
                handles.append(hand)

            lhigh = np.max(lasts)
        
        if J == 0:
            ax.set_ylabel(ylabel)
        if I == 2:
            ax.set_xlabel(xlabel)
        
        ax.set_xscale('log')

        plt.grid(False)
        sns.despine(left=False)
        
        ax.set_title(dataname)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f$'))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
    print(handles)
    return handles
    

def plot_rank(frames, klist, axes, filename=None, 
    xlabel='Factorization rank', ylabel='Number of iterations'):
    datasets = [titles[d] for d in DATASETS]
    handles = []

    axind = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    colors = {'MUR': 'b', 'PG': 'g', 'ALS': 'y', 'COD': 'r'}

    # Draw plots
    for i, dataname in enumerate(datasets):
        I, J = axind[i]
        ax = axes[I,J]

        if J == 0:
            # set y axis only for left plots
            ax.set_ylabel(ylabel)
        
        if I == 2:
            # set x axis only for bottom plots
            ax.set_xlabel(xlabel)
        
        # clean grid and some borders
        plt.grid(False)
        sns.despine(left=False)
        
        ax.set_title(dataname)
        
        for technique in TECHNIQUES:
            if technique not in frames[dataname]:
                continue
            data = frames[dataname][technique]
            hand, = ax.plot(klist, data, label=technique, color=colors[technique])    
            if i == 0:
                handles.append(hand)

    return handles


def figure_plotter(frames, klist, filename, xlabel='X', ylabel='Y', mode='convergence'):
    # set paper design
    sns.set_context("paper", font_scale=1)
    sns.set(style="whitegrid")
    
    matplotlib.rcParams['text.usetex'] = USE_LATEX
    matplotlib.rcParams['lines.linewidth'] = 2.0

    sns.set_style('ticks', {'xtick.major.size': 5.0, 'xtick.minor.size': 5.0, 
                'ytick.major.size': 5.0, 'ytick.minor.size': 5.0})
    
    plt.clf()
    # create the grid of 6 subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8))
    axind = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    axsub = [321, 322, 323, 324, 325, 326]
    
    # Specify colors for each method
    colors = {'MUR': 'b', 'PG': 'g', 'ALS': 'y', 'COD': 'r'}
    
    # static order of optimization techniques
    # put ALS at the end to prevent influence of ALS on y limits
    TECHNIQUES = ['MUR', 'PG', 'COD', 'ALS']
    if mode == 'convergence':
        handles = plot_convergence(frames, klist, axes)
    elif mode == 'rank':
        handles = plot_rank(frames, klist, axes)
    else:
        raise Exception("Unknown type of figure")
    
    # set layout
    fig.align_ylabels()
    fig.tight_layout()

    # set legend
    leg = plt.legend(handles=[handles[0], handles[3], handles[1], handles[2]], title=None, 
        loc="lower center", bbox_to_anchor=(-0.1, -0.5), ncol=4, fancybox=True, frameon=True)
    leg.get_frame().set_linewidth(1.0)
    
    ensure_dir(filename)
    # save as pdf
    # if this return some exception you probably need to install latex 
    # or set USE_LATEX = False (default)
    pdf = filename.replace('.png', '.png')
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(pdf)


def visu_convergence(k=30):
    # Visualize convergence of each of the 4 methods on each of the 6 datasets
    data = load_file('results/visdata/convergence%d.pkl' % k)
    frames = data[0]
    best_lines = data[1]
    figure_plotter(frames, best_lines, filename=os.path.join(IMG_DIR, 'convergence%d.png' % k), 
        xlabel='Iteration', ylabel='Objective value', mode='convergence')
    
    
def visu_rank():
    # Visualize runtime depending on rank for each of the 4 methods on each of the 6 datasets
    data = load_file('results/visdata/rank.pkl')
    lines = data[0]
    klist = data[1]
    figure_plotter(lines, klist, filename=os.path.join(IMG_DIR, 'rank.png'), 
        xlabel='Factorization rank', ylabel='Number of iterations', mode='rank')

if __name__ == '__main__':
    # calculate averages and parse history traces
    # save to results/visdata/
    stat_convergence(k=20)
    stat_rank()
    # run fnmtf/visualize.py from repository root directory
    visu_convergence(k=20)
    visu_rank()