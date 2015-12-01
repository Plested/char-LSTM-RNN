# -*- coding: utf-8 -*-
"""
Created on Mon Nov  23 10:14:21 2015

@author: jo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

results_file = 'NewOutputnoUpdatetest200-30'
text_output_file = 'textNewOutputnoUpdatetest200-30.txt'

with open((results_file), 'rb') as f1:
     losses = pickle.load(f1)
     stats = pickle.load(f1)
     text = pickle.load(f1)
losses = np.array(losses)

plt.plot((losses[:,0]/1000), (losses[:,2]), 'ro')
plt.plot((losses[:,0]/1000), (losses[:,1]), 'yo')
plt.axis([0, 310, 0, 120])
plt.xlabel('iterations: 1,000s')
plt.ylabel('test loss - red, training loss - yellow')
plt.show()

f = open(("testOutput" + ".txt"), 'w')
f.write(str(losses) + '\n')
f.write(str(stats))
f.write( '----\n %s \n----' % (text, ))

