#!/usr/bin/env python

import os
import csv
import subprocess
import argparse

from loader import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import ticker

IMG_DIR = 'img'
USE_LATEX = False

def order_frame(df, order=None):
    if order:
        frames = []
        for d in order:
            frames.append(df[df['Method'] == d])
        df = pd.concat(frames)
    
    return df

def matplotlib_plot(datasets, frames, best_lines, filename=None, labels=None, x='Iteration', y='Objective value', 
    z='Method', w='w', skipxlabels=10, xlabelrange=100, title=None):
    sns.set_context("paper", font_scale=1)
    sns.set(style="whitegrid")#, font_scale=1.0)
    
    matplotlib.rcParams['text.usetex'] = USE_LATEX
    matplotlib.rcParams['lines.linewidth'] = 2.0

    sns.set_style('ticks', {'xtick.major.size': 5.0, 'xtick.minor.size': 5.0, 'ytick.major.size': 5.0, 'ytick.minor.size': 5.0})
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8))

    plt.grid(False)
    
    xlabel = x
    ylabel = y

    
    axind = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    axsub = [321, 322, 323, 324, 325, 326]
    handles = []
    colors = {'MUR': 'b', 'PG': 'g', 'ALS': 'y', 'COD': 'r'}


    for i, dataname in enumerate(datasets):
        I, J = axind[i]
        #ax = axes[I,J]
        ax = axes[I,J]

        if dataname in ['STRING', 'MovieLens']:
            handles = []
        ylim0 = 1.0
        ylim1 = 0.0
        
        for technique in ['MUR', 'PG', 'COD', 'ALS']:
            _maxX = 0
            
            lasts = []
            if technique not in frames[dataname]:
                continue
            
            data = frames[dataname][technique]
            if len(data) == 0:
                continue
            
            #x = np.arange(len(data.keys()))
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
            
        
        ax.set_ylim([ylim0-(ylim1-ylim0)*0.1, ylim1+(ylim1-ylim0)*0.1])
        
        for technique in ['MUR', 'PG', 'COD', 'ALS']:
            _maxX = 0
            
            lasts = []
            if technique not in frames[dataname]:
                continue
            data = frames[dataname][technique]
            if len(data) == 0:
                continue
            
            #x = np.arange(len(data.keys()))
            x = sorted(data.keys())
            mx = max(x)
            if mx > _maxX:
                _maxX = mx
                
            last = x[-1]
            lasts.append(np.max(data[last]))
            CIS0 = []
            CIS1 = []
            EST = best_lines[dataname][technique]
            for i in x:
                est = np.mean(data[i])
                #est = best_lines[technique]
                sd = np.std(data[i])
                mins = np.min(data[i])
                maxs = np.max(data[i])
                #cis = (est - sd, est + sd)
                #EST.append(est)
                CIS0.append(mins)
                CIS1.append(maxs)


            #EST = [EST[i] if i < len(EST) else EST[-1] for i in x ]
            #print x
            #print range(len(EST))
            if technique == 'ALS':
                ax.fill_between(x,CIS0,CIS1,alpha=0.2, color='y')
                hand, = ax.plot(range(1,len(EST)+1),EST,label=technique, color=colors[technique])
                if dataname in ['STRING', 'MovieLens']:
                    handles.append(hand)
            else:
                ax.fill_between(x,CIS0,CIS1,alpha=0.2, color=colors[technique])
                hand, = ax.plot(range(1,len(EST)+1),EST,label=technique, color=colors[technique])
                if dataname in ['STRING', 'MovieLens']:
                    handles.append(hand)
            #ax.margins(x=0.2)
            
            #if technique == 'COD':
            #    x1,x2,y1,y2 = plt.axis()
            lhigh = np.max(lasts)
            
        if J == 0:
            ax.set_ylabel(ylabel)
        if I == 2:
            ax.set_xlabel(xlabel)
        
        ax.set_xscale('log')
        #ax.set_yscale('log')
        
        plt.grid(False)
        sns.despine(left=False)
        
        ax.set_title(dataname)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.2f$'))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

    fig.align_ylabels()
    fig.tight_layout()

    
    leg = plt.legend(handles=[handles[0], handles[3], handles[1], handles[2]], title=None, loc="lower center", bbox_to_anchor=(-0.1, -0.5), ncol=4, fancybox=True, frameon=True)
    leg.get_frame().set_linewidth(1.0)
    ensure_dir(filename)
    pdf = filename.replace('.png', '.pdf')
    
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(pdf)
    

def matplotlib_plot2(frames, datasets, klist, filename=None, labels=None, x='Factorization rank', y='Number of iterations', 
    z='Method', w='w', skipxlabels=10, xlabelrange=100, title=None):
    sns.set_context("paper", font_scale=1)
    sns.set(style="whitegrid")#, font_scale=1.0)
    
    matplotlib.rcParams['text.usetex'] = USE_LATEX
    matplotlib.rcParams['lines.linewidth'] = 2.0
        
    #sns.set_context('paper', font_scale=1) 

    sns.set_style('ticks', {'xtick.major.size': 5.0, 'xtick.minor.size': 5.0, 'ytick.major.size': 5.0, 'ytick.minor.size': 5.0})
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,8))#, sharex=True )
    plt.grid(False)
    
    xlabel = x
    ylabel = y
    axind = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    axsub = [321, 322, 323, 324, 325, 326]
    handles = []
    colors = {'MUR': 'b', 'PG': 'g', 'ALS': 'y', 'COD': 'r'}
    for i, dataname in enumerate(datasets):
        I, J = axind[i]
        ax = axes[I,J]

        if dataname in ['Mutations', 'MovieLens']:
            handles = []

        if J == 0:
            ax.set_ylabel(ylabel)
            #ax.get_yaxis().set_label_coords(-0.14,0.5)
        if I == 2:
            ax.set_xlabel(xlabel)
        
        plt.grid(False)
        sns.despine(left=False)
        
        ax.set_title(dataname)
        
        for technique in ['MUR', 'PG', 'COD', 'ALS']:
            if technique not in frames[dataname]:
                continue
            data = frames[dataname][technique]
            hand, = ax.plot(klist, data, label=technique, color=colors[technique])    
            handles.append(hand)
    
    fig.align_ylabels()
    fig.tight_layout()

    leg = plt.legend(handles=[handles[0], handles[3], handles[1], handles[2]], title=None, 
        loc="lower center", bbox_to_anchor=(-0.1, -0.5), ncol=4, fancybox=True, frameon=True)
    leg.get_frame().set_linewidth(1.0)
    ensure_dir(filename)
    pdf = filename.replace('.png', '.pdf')
    
    
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(pdf)#, bbox_inches="tight")#, pad_inches=0.2)


def visu_convergence(k=30):
    data = load_file('results/visdata/convergence%d.pkl' % k)
    dnames = data[0]
    frames = data[1]
    best_lines = data[2]
    matplotlib_plot(dnames, frames, best_lines, filename='img/convergence%d.png' % k)
    
    
def visu_rank():
    data = load_file('results/visdata/rank.pkl')
    lines = data[0]
    titles = data[1]
    klist = data[2]

    matplotlib_plot2(lines, titles, klist, y="Number of iterations", filename=os.path.join(IMG_DIR, 'rank.png'))
        #analyze(frames, dataset=dataset)

if __name__ == '__main__':
    visu_convergence(k=20)
    visu_rank()