# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:18:31 2015

@author: jo
"""

"""
character-level LSTM RNN model based on LSTM model in http://arxiv.org/pdf/1308.0850.pdf
and adapted from Vanilla RNN model written by Andrej Karpathy (@karpathy).
BSD License
"""
import numpy as np
import time
import pickle
import string

# hyperparameters
h = 300 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = .005

words = []
haveWords = False

def getNGrams(data):
   """
   gets words for data set or words and 2-Grams for sample text
   """
   text = data.split()
   previousWord = text[0].strip(string.punctuation)
   print(previousWord)
   Twograms = []
   i = 0
   w = []
   previous = False
   totalLen = 0
   for word in text:
        word = word.strip(string.punctuation)
        if(haveWords):
            if(len(word) >0):
             if(word in words):
                w.append(word)
                totalLen += len(word)
                if previous:
                    Twograms.append(previousWord + " " + word)
                previousWord = word 
                previous = True
             else:
                previous = False
        else:
            w.append(word)
            if previous:
                Twograms.append(previousWord + " " + word)
            previousWord = word 
            previous = True
               
        i += 1
    
   #if(len(Twograms) > 2):
    #   print(Twograms[0])
     #  print(Twograms[1])
      # print(Twograms[2])
   Twograms = list(set(Twograms))
   #print(len(Twograms))
   if(len(w)>0):
       print('average word length: %f' %((float(totalLen))/len(w)))

   theseWords = list(set(w))  
   print 'data has %d unique words.' %(len(theseWords))
   
   return theseWords, Twograms
   
# data I/O
with open('bigtxt.txt', 'r') as f:
    data = f.read()# should be simple plain text file
    chars = list(set(data))
    testSet = data[-len(data)/10:] #use last 1/10 of the data as text set
    data = data[:-len(data)/10]
    data_size, vocab_size = len(data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    

words, twoGrams = getNGrams(data)
haveWords = True 



# model parameters
WIF = np.random.randn((vocab_size + h + 1), 2*h)*0.01
WIFc = np.random.randn(h,2*h)*0.01
WC = np.random.randn((vocab_size + h + 1), h)*0.01
WO = np.random.randn((vocab_size + h + 1), h)*0.01
WOc = np.random.randn(h,h)*0.01
Why = np.random.randn(h, vocab_size)*0.01 # hidden to output
by = np.zeros((vocab_size)) # output bias

def lossFun(inputs, targets, hprev, c):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, 
  last hidden state and last memory cell state
  """
  xs = np.zeros((len(inputs), vocab_size))
  ys = np.zeros((len(inputs), vocab_size)) 
  ps = np.zeros((len(inputs), vocab_size))
  Hin = np.zeros((len(inputs), WC.shape[0]))
  Hout = np.zeros((len(inputs), h))
  IF = np.zeros((len(inputs), 2*h))
  IFs = np.zeros_like(IF)
  C = np.zeros((len(inputs), h))
  Cs = np.zeros_like(C)
  O = np.zeros((len(inputs), h))
  Os = np.zeros_like(O)
    
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t][inputs[t]] = 1
    Hin[t,0] = 1 #bias for each gate
    Hin[t,1:1+vocab_size] = xs[t] #input
    Hin[t,1+vocab_size:1+vocab_size+h] = hprev if t == 0 else Hout[t-1]  #h[t-1]
    
    if(t>0):
        IF[t] = (Hin[t]).dot(WIF) + C[t-1].dot(WIFc)
    else:
        IF[t] = (Hin[t]).dot(WIF) + c.dot(WIFc)
    IFs[t] = 1.0/(1.0+np.exp(-IF[t]))
    C[t] = Hin[t].dot(WC)
    if(t>0):
        Cs[t] = IFs[t, h:]*Cs[t-1] + IFs[t,:h]*np.tanh(C[t])
    else:
        Cs[t] = IFs[t, h:]*c + IFs[t,:h]*np.tanh(C[t])
    O[t] = Hin[t].dot(WO) + Cs[t].dot(WOc)
    #print(sum(IF[t]))
    Os[t] = 1.0/(1.0+np.exp(-O[t]))
    #print(sum(Os[t]))
    if((np.isnan(Os[t])).any()):
        print(O[t])
    Hout[t] = Os[t]*np.tanh(Cs[t])    
    
    ys[t] = np.dot(Hout[t], Why) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(1e-10 + ps[t][targets[t]]) # softmax (cross-entropy loss)
    
  # backward pass: compute gradients going backwards
  dWIF, dWIFc, dWC, dWO, dWOc = np.zeros_like(WIF), np.zeros_like(WIFc), np.zeros_like(WC), np.zeros_like(WO), np.zeros_like(WOc)
  dWhy, dby =  np.zeros_like(Why), np.zeros_like(by)
     
  dHout = np.zeros_like(Hout)
  dIF = np.zeros_like(IF)
  dIFs = np.zeros_like(IFs)
  dC = np.zeros_like(C)
  dCs = np.zeros_like(C)
  dO = np.zeros_like(O)
  dOs = np.zeros_like(Os)
    
  for t in reversed(xrange(len(inputs))): 
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 
    dWhy += np.outer(Hout[t], dy) 
    dby += dy 
    dHout[t] = np.dot(dy, Why.T) 
    dOs[t] += dHout[t]*np.tanh(Cs[t]) 
    dCs[t] += (dHout[t]*Os[t])*(1-((np.tanh(Cs[t]))**2)) 
    dO[t] += Os[t]*(1-Os[t])*dOs[t] 
    dWO += np.outer(Hin[t], dO[t]) 
    dCs[t] += dO[t].dot(WOc.T) 
    dWOc += np.outer(C[t], dO[t]) 
    if(t>0):
        dIFs[t, h:] += dCs[t]*Cs[t-1]
        dCs[t-1] += dCs[t]*IFs[t, h:]
    else:
        dIFs[t, h:] += dCs[t]*c
        
    dIFs[t, :h] += dCs[t]*np.tanh(C[t])
    dC[t] += dCs[t]*(1-(np.tanh(C[t]))**2)
    dWC += np.outer(Hin[t].T, dC[t])
    #print(np.sum(dIFs[t]))
    dIF[t] = dIFs[t]*IFs[t]*(1-IFs[t]) 
    #print(np.sum(dIF[t]))
    dWIF += np.outer(Hin[t], dIF[t])
    if(t>0):
        dWIFc += np.outer(Cs[t-1], dIF[t])
    else:
        dWIFc += np.outer(c, dIF[t])
        
    #print(sum(dO[t]))
        
  for dparam in [dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, Hout[len(inputs)-1], Cs[len(inputs)-1]

def sample(prevh, cprev, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((n, vocab_size))
  #print(x.shape)
 
  Hin = np.zeros((n, WC.shape[0]))
  #HinIF = np.zeros((len(inputs), WIF.shape[0]))
  Hout = np.zeros((n, h))
  IF = np.zeros((n, 2*h))
  IFs = np.zeros_like(IF)
  C = np.zeros((n, h))
  Cs = np.zeros_like(C)
  O = np.zeros((n, h))
  Os = np.zeros_like(O)
  ixes = []
  ix = seed_ix
  
  for t in xrange(n):
    x[t][ix] = 1
    Hin[t,0] = 1
    Hin[t,1:1+vocab_size] = x[t] #input
    Hin[t,1+vocab_size:1+vocab_size+h] = hprev if t == 0 else Hout[t-1]  #h[t-1]
    if(t>0):
        IF[t] = (Hin[t]).dot(WIF) + C[t-1].dot(WIFc)
    else:
        IF[t] = (Hin[t]).dot(WIF) + cprev.dot(WIFc)
    IFs[t] = 1.0/(1.0+np.exp(-IF[t]))
    C[t] = Hin[t].dot(WC)
    if(t>0):
        Cs[t] = IFs[t, h:]*Cs[t-1] + IFs[t,:h]*np.tanh(C[t])
    else:
        Cs[t] = IFs[t, h:]*cprev + IFs[t,:h]*np.tanh(C[t])
    O[t] = Hin[t].dot(WO) + Cs[t].dot(WOc)
    #print(sum(IF[t]))
    Os[t] = 1.0/(1.0+np.exp(-O[t]))
    #print(sum(Os[t]))
    Hout[t] = Os[t]*np.tanh(Cs[t])    

    y = np.dot(Hout[t], Why) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    ixes.append(ix)   
    letter = np.argmax(p)
       
    if(t == (n-1)):
        print(letter)
        print(p[letter])
   
  return ixes
  
def getStats(txt):
    """
    checks how many words and n-grams are in the sample text
    """    
    sampleWords, sampleTwoGrams = getNGrams(txt)
    #for word in sampleWords:
     #   if(not(word in words)):
      #      sampleWords.remove(word)
    for s in sampleTwoGrams:
        if(not(s in twoGrams)):
            sampleTwoGrams.remove(s)
    print 'no of sampleWords: %d' % (len(sampleWords))
    print 'no of sample 2-grams: %d' % (len(sampleTwoGrams))
    print 'precentage of sample words: %f' % ((float(len(sampleWords)))/len(words))
    print 'precentage of sample 2-grams: %f' % ((float(len(sampleTwoGrams)))/(len(twoGrams)))
    
def getTestLoss():
    """
    calculates loss on the test set
    """
    hprev = np.zeros((h))
    cprev = np.zeros((h))
    p = 0
    inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
    loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev) #warm up set
    p += seq_length
    totalLoss = 0
    for n in range(100):
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
        loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
        #print(loss)
        totalLoss += loss
        p += seq_length
    avgLoss = totalLoss/100
    return avgLoss

n, p = 0, 0
mWIF, mWIFc, mWC, mWO, mWOc, mWhy = np.zeros_like(WIF), np.zeros_like(WIFc), np.zeros_like(WC), np.zeros_like(WO), np.zeros_like(WOc), np.zeros_like(Why)
mby = np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
#with open(('newModel118'), 'rb') as f1:
 #     model = pickle.load(f1)
#j = 0
#for weightVector, i in enumerate(model['WIF']):
 #   for k in range(2):
  #      WIF[j, k*h:((k*h)+len(weightVector)/2)] = weightVector[((k/2.0)*(len(weightVector))):(((k+1)/2.0)*(len(weightVector)))]
#WIF = model['WIF']
#WIFc = model['WIFc']
#WC = model['WC']
#WO = model['WO']
#WOc = model['WOc']
#Why = model['Why']
#by = model['by']

t0 = time.clock()
iterVloss = []
total_loss = 0

while True:
   
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((h)) # reset RNN memory
    cprev = np.zeros((h))
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  
  # sample from the model now and then
  if n % 10000 == 99:
        sample_ix = sample(hprev, cprev, inputs[0], 1000)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print( '----\n %s \n----' % (txt, ))
        with open("RNNLSTMoutput300", 'wb') as f1:
            pickle.dump(iterVloss,f1)
            pickle.dump( '----\n %s \n----' % (txt, ), f1)
        
        getStats(txt)
        
    
  if n % 10000 == 999:
        model = {'WIF': WIF, 'WIFc': WIFc, 'WC': WC, 'WO': WO, 'WOc': WOc, 'Why': Why, 'by':by}
        with open(('NewModels/newModel300-' + str(n/10000)), 'wb') as f:
           pickle.dump(model, f)  
    

  # forward seq_length characters through the net and fetch gradient
  loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
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
    
  # perform parameter update
  for param, dparam, mem in zip([WIF, WIFc, WC, WO, WOc, Why, by], 
                                [dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby], 
                                [mWIF, mWIFc, mWC, mWO, mWOc, mWhy, mby]):
        
    #if n%1000 == 1:
     #   print("sum")
        #for d in dparam:
      #  print(np.sum(dparam))
    mem += dparam * dparam
    #if n%1000 == 1:
     #   print(np.sum(param))
    #param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    param += learning_rate*-dparam
    #if n%1000 == 1:
     #   print(np.sum(param))

  p += seq_length # move data pointer
  n += 1 # iteration counter 
