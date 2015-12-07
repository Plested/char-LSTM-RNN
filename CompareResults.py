# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 08:37:34 2015

Outputs graphs comparing output statistics for new vs old or new vs new models.
When doing comparisons with original model output files they must be entered in the 
original_results_file variable, not the new_results_file as they have different output
formats. 

There is a bit of hacking with if statements to allow the method to deal with any 
format of output stats as they changed a lot over time as new statistics were deemed 
useful and added.

@author: jo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

new_results_file = 'NewOutputMom0.1_200-29'
original_results_file = 'RNNoriginaloutputNoUpdatetest510-41'

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
plt.axis([0, 300, 0, 120])
plt.xlabel('iterations: 1,000s')
plt.ylabel('model 1 test loss - red, model 2 test loss - yellow')
plt.show()

NewStatsArray = []
if len(NewStats[1])>7 and (len(NewStats[1][7].shape) > 1):
    print 'yes new'
    NewNgramLens = np.zeros((len(NewStats), 20, 2))
else:
    NewNgramLens = np.zeros((len(NewStats), 20, 1))
totalNewNgramLens = np.zeros((len(NewStats)))
for i, n in enumerate(NewStats):
  if type(NewStats[i]) == list:
    thisArray = []
    for j, m in enumerate(n):
            if j < 7:
                thisArray.append(np.array(m))
            else:                
                totalNgramLen = 0
                totalNgrams = 0
                for k in xrange(20):                    
                    NewNgramLens[i,k]+=m[k]
                    if(len(m[k].shape)>0):                      
                        totalNewNgramLens[i] += m[k][1]                    
                        totalNgramLen += k*m[k][0]
                        totalNgrams += m[k][0]                                                              
                if(totalNgrams > 0):
                    thisArray.append(totalNgramLen/totalNgrams)
                else:
                    thisArray.append(0)                   
    NewStatsArray.append(np.array(thisArray))
NewStatsArray = np.array(NewStatsArray)

OldStatsArray = []
Oldstyle = False
if(not(type(OldStats[0])==list)): 
    Oldstyle = True
if len(OldStats[1])>7 and (len(OldStats[1][7].shape)>1): 
    print 'yes old'
    if not(Oldstyle):
        OldNgramLens = np.zeros((len(OldStats), 20, 2))
    else:
        OldNgramLens = np.zeros((len(OldStats)-1, 20, 2))
else:
    if not(Oldstyle):
        OldNgramLens = np.zeros((len(OldStats), 20, 1))
    else:
        OldNgramLens = np.zeros((len(OldStats)-1, 20, 1))
if not(Oldstyle):
    totalOldNgramLens = np.zeros((len(OldStats)))
else:
    totalOldNgramLens = np.zeros((len(OldStats)-1))
for i, n in enumerate(OldStats):
   if type(OldStats[i]) == list:
        thisArray = []
        for j, m in enumerate(n):
            if j < 7:
                thisArray.append(np.array(m))
            else:                
                totalNgramLen = 0
                totalNgrams = 0
                for k in xrange(20):
                    if not(Oldstyle):
                        OldNgramLens[i,k]+=m[k]
                    else:
                        OldNgramLens[i-1,k]+=m[k]
                    if(len(m[k].shape)>0):
                        if not(Oldstyle):
                            totalOldNgramLens[i] += m[k][1]
                        else:
                            totalOldNgramLens[i-1] += m[k][1] 
                        totalNgramLen += k*m[k][0]
                        totalNgrams += m[k][0]
                if(totalNgrams > 0):
                    thisArray.append(totalNgramLen/totalNgrams)
                else:
                    thisArray.append(0)
        OldStatsArray.append(np.array(thisArray))
OldStatsArray = np.array(OldStatsArray)

plt.figure(2)
plt.plot((NewStatsArray[:,0]/1000), (NewStatsArray[:,6]), 'r--') #average valid word length at each iteration
plt.plot((OldStatsArray[:,0]/1000), (OldStatsArray[:,6]), 'y--')
plt.axis([0, 300, 0, 4])
plt.xlabel('iterations: 1,000s')
plt.ylabel('model 1 - red, model 2 - yellow')
plt.show()

plt.figure(3)
plt.plot((NewStatsArray[:,0]/1000), (NewStatsArray[:,3]), 'r--') #total number of N-grams produced of any length at each iteration
plt.plot((OldStatsArray[:,0]/1000), (OldStatsArray[:,3]), 'y--')
plt.axis([0, 300, 0, 3000])
plt.xlabel('iterations: 1,000s')
plt.ylabel('model 1 - red, model 2 - yellow')
plt.show()

plt.figure(4)
plt.plot((NewStatsArray[:,0]/1000), (NewNgramLens[:,5,0]), 'r--') #number of N-grams of a certain length at each test iteration change the
plt.plot((OldStatsArray[:,0]/1000), (OldNgramLens[:,5,0]), 'y--') #second dimension of the second axis in each to the required N-gram length 
plt.axis([0,300, 0, 200])
plt.xlabel('iterations: 1,000s')
plt.ylabel('model 1 - red, model 2 - yellow')
plt.show()

    

if (len(NewStats[1][7].shape) > 1) and (len(NewStats[1][7].shape) > 1):
    plt.figure(5)
    plt.plot((NewStatsArray[:,0]/1000), (totalNewNgramLens[:]/NewStatsArray[:,3]), 'r--') #average length in characters of all N-grams at each test iteration
    plt.plot((OldStatsArray[:,0]/1000), (totalOldNgramLens[:]/OldStatsArray[:,3]), 'y--')
    plt.axis([0,300, 0,15])
    plt.xlabel('iterations: 1,000s')
    plt.ylabel('model 1 - red, model 2 - yellow')
    plt.show()

if(NewStatsArray.shape[1]>7) and (OldStatsArray.shape[1]>7):
    plt.figure(6)
    plt.plot((NewStatsArray[:,0]/1000), (NewStatsArray[:,7]), 'r--') #average length in words of all N-grams at each test iteration
    plt.plot((OldStatsArray[:,0]/1000), (OldStatsArray[:,7]), 'y--') 
    plt.axis([0,300, 0, 5])
    plt.xlabel('iterations: 1,000s')
    plt.ylabel('mdoel 1 - red, model 2 - yellow')
    plt.show()
