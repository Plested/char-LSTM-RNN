# -*- coding: utf-8 -*-
"""
Created on Mon Nov  23 10:14:21 2015

@author: jo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

results_file = 'RNNoriginaloutputNoUpdatetest510-30'
text_output_file = 'RNNoriginaloutputNoUpdatetest510-30.txt'

with open((results_file), 'rb') as f1:
     losses = pickle.load(f1)
     stats = pickle.load(f1)
     text = pickle.load(f1)
losses = np.array(losses)

plt.figure(1)
plt.plot((losses[:,0]/1000), (losses[:,2]), 'r--')
plt.plot((losses[:,0]/1000), (losses[:,1]), 'y--')
plt.axis([0, 310, 0, 120])
plt.xlabel('iterations: 1,000s')
plt.ylabel('test loss - red, training loss - yellow')
plt.show()

statsArray = []
for i, n in enumerate(stats):
    if type(stats[i]) == list:
        thisArray = []
        for j, m in enumerate(n):
            if j < 7:
                thisArray.append(np.array(m))
        statsArray.append(np.array(thisArray))
statsArray = np.array(statsArray)

plt.figure(2)
plt.plot((statsArray[:,0]/1000), (statsArray[:,6]), 'r--')
plt.plot((losses[:,0]/1000), (1/(losses[:,2]/100)), 'y--')
plt.plot((statsArray[:,0]/1000), (statsArray[:,3]/400), 'b--')
plt.axis([0, 310, 0, 6])
plt.xlabel('iterations: 1,000s')
plt.ylabel('avg word len - red, inv(train loss/100) - yellow, (N-grams)/400 - blue')
plt.show()


f = open((text_output_file), 'w')
f.write(str(losses) + '\n')
f.write(str(stats))
f.write( '----\n %s \n----' % (text, ))

