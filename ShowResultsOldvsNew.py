# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:37:34 2015

@author: jo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

new_results_file = 'NewOutputnoUpdatetest200-30'
original_results_file = 'RNNoriginaloutputNoUpdatetest200-32'

with open((new_results_file), 'rb') as f1:
     NewLosses = pickle.load(f1)
     NewStats = pickle.load(f1)
     
NewLosses = np.array(NewLosses)

with open((original_results_file), 'rb') as f1:
     OldLosses = pickle.load(f1)
     OldStats = pickle.load(f1)
     
OldLosses = np.array(OldLosses)

plt.figure(1)
plt.plot((NewLosses[:,0]/1000), (NewLosses[:,2]), 'r--')
plt.plot((OldLosses[:,0]/1000), (OldLosses[:,2]), 'y--')
plt.axis([0, 310, 0, 120])
plt.xlabel('iterations: 1,000s')
plt.ylabel('New test loss - red, Old test loss - yellow')
plt.show()

NewStatsArray = []
for i, n in enumerate(NewStats):
    if i > 0:
        NewStatsArray.append(np.array(n))
NewStatsArray = np.array(NewStatsArray)

OldStatsArray = []
for i, n in enumerate(OldStats):
    if i > 0:
        thisArray = []
        for j, m in enumerate(n):
            if j < 7:
                thisArray.append(np.array(m))
        OldStatsArray.append(np.array(thisArray))
OldStatsArray = np.array(OldStatsArray)

plt.figure(2)
plt.plot((NewStatsArray[:,0]/1000), (NewStatsArray[:,6]), 'r--')
plt.plot((OldStatsArray[:,0]/1000), (OldStatsArray[:,6]), 'y--')
plt.axis([0, 310, 0, 4])
plt.xlabel('iterations: 1,000s')
plt.ylabel('New avg valid word length - red, Old avg valid word length - yellow')
plt.show()

plt.figure(2)
plt.plot((NewStatsArray[:,0]/1000), (NewStatsArray[:,3]), 'r--')
plt.plot((OldStatsArray[:,0]/1000), (OldStatsArray[:,3]), 'y--')
plt.axis([0, 310, 0, 3000])
plt.xlabel('iterations: 1,000s')
plt.ylabel('New number of N-grams - red, Old number of N-grams - yellow')
plt.show()