# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 07:28:51 2015

@author: jo
"""

"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import string
import pickle
import copy
import time

# hyperparameters
hidden_size = 200 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1
input_txt = 'bigtxt.txt'
output_results = 'OriginalOutput/RNNoriginaloutputNoUpdatetest200-'
output_model = 'OriginalModels/OriginalModelNoUpdatetest200-'

words = []
haveWords = False
allWords = []
avgWordLen = 0

def getNGrams(data):
   """
   returns valid words and N-grams from a test set
   """
   text = data.split()
   Ngrams = {}
   theseWords = []
   #allWords = []
   i = 0
   w = []
   length = 0
   totalLen = 0
   
   for word in text:
        word = word.strip(string.punctuation)
        if(haveWords):
            if(len(word) >0):
             if(word in words):
                w.append(word)
                totalLen += len(word)
                length += 1
                theseWords.append(word)
                
                if length > 1: #collect N-grams
                    for i in xrange(length-1):
                        if not(Ngrams.has_key(theseWords[-2])):
                            Ngrams[theseWords[-2]] = []
                        thisNgram = []
                        for j in xrange(i, length):
                            thisNgram.append(theseWords[j])
                        Ngrams[theseWords[i]].append(thisNgram)
                
             else: #not a valid word
                length = 0
                del theseWords[:]
                
        else: #in training set no need to find N-grams
            w.append(word)
            #allWords.append(word)
            totalLen += len(word)
            
        i += 1
   #get unique N-grams       
   for key in Ngrams.keys():
       newLists = []
       for currentList in Ngrams[key]:
           if currentList not in newLists:
               newLists.append(currentList)
       Ngrams[key] = newLists
   
   if(len(w)>0):
       avgWordLen = ((float(totalLen))/len(w))   
       print('average word length: %f' % (avgWordLen))
   
   #get unique words
   theseWords = list(set(w))

   print 'data has %d unique words.' %(len(theseWords))
   print 'data has %d total words' %(len(w))
   
   return w, theseWords, Ngrams, avgWordLen


# data I/O
with open(input_txt, 'r') as f:
    data = f.read()# should be simple plain text file
    chars = list(set(data))
    testSet = data[-len(data)/10:] #use last 1/10 of the data as text set
    data = data[:-len(data)/10]
    data_size, vocab_size = len(data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
allWords, words, twoGrams, avgWordLen = getNGrams(data)
haveWords = True        


# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
  

def sample(n):
  """ 
  sample a sequence of integers from the model 
  warm up on the test set, seed_ix is seed letter for first time step
  """
  hprev = np.zeros((hidden_size,1)) #start from scratch
  p = 0
  inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
  loss, dWxh, dWhh, dWhy, dbh, dby, h = lossFun(inputs, targets, hprev) #warm up on test set
  p += seq_length
   
  x = np.zeros((vocab_size, 1))
  x[char_to_ix[testSet[p]]] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh) 
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
  
  
def getStats(txt):
    
    totalWords, sampleWords, sampleNGrams, avgWLen = getNGrams(txt)
    t = time.clock()
    dataNgrams = []   
    startIndicies = {}
    #find start indicies for potential sample N-grams in data set.
    #loop through the data once and include and index if it is the start of at least a 2-gram
    for key in sampleNGrams.keys():
        startIndicies[key] = []
    for i, word in enumerate(allWords):
        if word in sampleNGrams.keys():
            if(i<(len(allWords)-1)):
                for Ngram in sampleNGrams[word]:
                    if(Ngram[1] == allWords[i+1]):
                        startIndicies[word].append(i)
                        break
            
    t0 = time.clock() - t
    print(t0)
    
    n = 0
    #find sample N-grams that are in the data set.
    for startWord in sampleNGrams.keys():
        for Ngram in sampleNGrams[startWord]:
            n += 1
            indicies = copy.copy(startIndicies[startWord])
            for i in xrange(1, len(Ngram)):
                for index in indicies:
                    if(index+i)>=len(allWords):
                        indicies.remove(index)
                    else:
                        if(not(allWords[index+i]==Ngram[i])):
                            indicies.remove(index)
                if(len(indicies)<1):
                    break
                if(i==(len(Ngram)-1)):
                    dataNgrams.append(Ngram)
    
    no_Ngram_sizes = np.zeros((20))               
    i = 0
    for Ngram in dataNgrams:
        no_Ngram_sizes[len(Ngram)] += 1
        if(len(Ngram)>i):
            i = len(Ngram)
    t1 = time.clock() - t0
    print(str(t1))
            
            
    print 'no of sampleWords: %d' % (len(sampleWords))
    print 'no of sample N-grams: %d' % (n)
    print 'no of sample N-grams from data: %d' % (len(dataNgrams))
    print 'max length of data N-gram: %d' % (i)
    print 'precentage of sample words: %f' % ((float(len(sampleWords)))*100/len(words))
    
    stats = []
    stats.append(len(sampleWords))
    stats.append(n)
    stats.append(len(dataNgrams))
    stats.append(i)
    stats.append((float(len(sampleWords)))*100/len(words))
    stats.append(avgWLen)
    stats.append(no_Ngram_sizes)
    return stats
    
def getTestLoss():
    hprev = np.zeros((hidden_size,1)) #start from scratch
    p = 0
    inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev) #warm up on test set
    p += seq_length
    totalLoss = 0
    for n in range(100):
        inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]] #characters from test set
        targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev) #calculate loss on inputs from test set
        totalLoss += loss
        p += seq_length
    avgLoss = totalLoss/100
    return avgLoss
    
iterVloss = [] #for storing number of iterations with training and test losses at regular intervals as training progresses
allStats = [] #for storing stats on sample text produced at regular intervals as training progresses.
allStats.append(avgWordLen)
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
total_loss = 0

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 1000 == 0:
        sample_ix = sample(10000)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        stats = getStats(txt)
        stats.insert(0, n)
        allStats.append(stats)
        if n % 10000 == 9000:
            with open((output_results+ str(n/10000)), 'wb') as f1:
                pickle.dump(iterVloss, f1)
                pickle.dump(allStats, f1)
                pickle.dump( '----\n %s \n----' % (txt, ), f1)
        
            model = {'Wxh': Wxh, 'Whh': Whh, 'Why': Why, 'bh': bh, 'by':by}
            with open((output_model + str(n/10000)), 'wb') as f:
                pickle.dump(model, f)  

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  total_loss += loss
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 1000 == 0: 
      if(n>0):
          loss = total_loss/1000
      print 'iter %d, loss: %f' % (n, loss) # print progress
      testLoss = getTestLoss()
      print testLoss      
      thisLoss = [n, loss, testLoss]
      iterVloss.append(thisLoss)
      total_loss = 0
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
   
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
   
  p += seq_length # move data pointer
  n += 1 # iteration counter 


