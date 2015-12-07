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
import copy

# hyperparameters
h = 200 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = .005
learning_rate_decay = .99999 #to not use learning rate decay make this 1.0
momentum = 0.1
inputFile = 'bigtxt.txt'
outputFile = "NewOutput/NewOutputRandomStartsMom0.1_200-" #for saving the output
modelOutput = 'NewModels/NewModelRandomStartsMom0.1_200-' #for saving the trained weights

words = []
haveWords = False

def getNGrams(data):
   """
   returns valid words and al N-grams consisting of valid words from a test set
   """
   text = data.split()
   Ngrams = {}
   theseWords = []
   allWords = []
   i = 0
   w = []
   length = 0
   totalLen = 0
   for word in text:
        word = word.strip(string.punctuation)
        if(haveWords):
            if(len(word) >0):
             if(word in words): #this is a valid word
                w.append(word)
                totalLen += len(word)
                length += 1
                theseWords.append(word)
                                               
                if length > 1:  #collect N-grams              
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
            allWords.append(word)
            totalLen += len(word)            
        i += 1
   
   #get unique N-grams  
   for key in Ngrams.keys():
       newLists = []
       for currentList in Ngrams[key]:
           if currentList not in newLists:
               newLists.append(currentList)
       Ngrams[key] = newLists
       
   #get unique words
   if(len(w)>0):
       avgWordLen = ((float(totalLen))/len(w))   
       print('average word length: %f' % (avgWordLen))
   else:
       avgWordLen = 0
  
   theseWords = list(set(w))

   print 'data has %d unique words.' %(len(theseWords))
   print 'data has %d total words' %(len(w))
   
   return w, theseWords, Ngrams, avgWordLen

   
# data I/O
with open(inputFile, 'r') as f:
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
WIF = np.random.randn((vocab_size + h + 1), 2*h)*0.01 #input and forget gate input weights
WIFc = np.random.randn(h,2*h)*0.01 #input and forget gate input from memory cell weights
WC = np.random.randn((vocab_size + h + 1), h)*0.01 #memory cell input weights
WO = np.random.randn((vocab_size + h + 1), h)*0.01 #output gate input weights
WOc = np.random.randn(h,h)*0.01 #output gate input from memory cell weights
Why = np.random.randn(h, vocab_size)*0.01 # hidden to output
by = np.zeros((vocab_size)) # output bias

def lossFun(inputs, targets, hprev, c, update):
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
    #calculate gate and memory cell current states
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
    Os[t] = 1.0/(1.0+np.exp(-O[t]))
    #calculate final hidden layer output
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
    
  if(update):  # to save time only calculate derivatives if they are needed for updating weights.  
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
    dIF[t] = dIFs[t]*IFs[t]*(1-IFs[t]) 
    dWIF += np.outer(Hin[t], dIF[t])
    if(t>0):
        dWIFc += np.outer(Cs[t-1], dIF[t])
    else:
        dWIFc += np.outer(c, dIF[t])
                
  for dparam in [dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, Hout[len(inputs)-1], Cs[len(inputs)-1]

def sample(n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((n, vocab_size))
  Hin = np.zeros((n, WC.shape[0]))
  Hout = np.zeros((n, h))
  IF = np.zeros((n, 2*h))
  IFs = np.zeros_like(IF)
  C = np.zeros((n, h))
  Cs = np.zeros_like(C)
  O = np.zeros((n, h))
  Os = np.zeros_like(O)
  
  hprev = np.zeros((h)) #reset internal states
  cprev = np.zeros_like(hprev)
  p = 0
  inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
  loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev, False) #warm up set
  p += seq_length  
  ixes = []
  ix = char_to_ix[testSet[p]]
  
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
    Os[t] = 1.0/(1.0+np.exp(-O[t]))
    Hout[t] = Os[t]*np.tanh(Cs[t])    

    y = np.dot(Hout[t], Why) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel()) #output character and input for next time step
    ixes.append(ix) 
    
    #prints the character with the highest probability and its associated probability on the final time step for curiosity purposes.
    letter = np.argmax(p)       
    if(t == (n-1)):
        print(letter)
        print(p[letter])
   
  return ixes
  
def getStats(txt):
    """
    checks how many words and n-grams are in the sample text
    """    
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
            #print(word)
            if(i<(len(allWords)-1)):
                for Ngram in sampleNGrams[word]:
                    if(Ngram[1] == allWords[i+1]):
                        startIndicies[word].append(i)
                        break
            
    t0 = time.clock() - t
    print(t0)
    
    #find sample N-grams that are in the data set.
    n = 0
    for startWord in sampleNGrams.keys():        
        for Ngram in sampleNGrams[startWord]:
            n += 1            
            indicies = copy.copy(startIndicies[startWord]) #check potential places where N-gram could start and remove indicies that don't match            
            for i in xrange(1, len(Ngram)):
                for index in indicies:
                    if(index+i)>=len(allWords):
                        indicies.remove(index)
                    else:
                        if(not(allWords[index+i]==Ngram[i])):
                            indicies.remove(index)
                if(len(indicies)<1): #if no more indicies to check then stop checking
                    break
                if(i==(len(Ngram)-1)): #if we're at the end of the Ngram add it to the list of valid Ngrams
                    dataNgrams.append(Ngram)
    
    #count number of Ngrams and length in characters of Ngrams of different sizes                
    no_Ngram_sizes = np.zeros((20,2))               
    i = 0
    for Ngram in dataNgrams:
        no_Ngram_sizes[len(Ngram),0] += 1
        no_char = 0
        for word in Ngram:
            no_char += len(word)
        no_Ngram_sizes[len(Ngram), 1] += no_char
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
    """
    calculates loss on the test set
    """
    hprev = np.zeros((h)) #reset hidden states
    cprev = np.zeros((h))
    p = 0
    inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]] 
    targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
    loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev, False) #warm up set
    p += seq_length
    totalLoss = 0
    for n in range(100):
        inputs = [char_to_ix[ch] for ch in testSet[p:p+seq_length]] #characters from test set
        targets = [char_to_ix[ch] for ch in testSet[p+1:p+seq_length+1]]
        loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev, False) #calculate loss on inputs and targets from test set        
        totalLoss += loss
        p += seq_length
    avgLoss = totalLoss/100
    return avgLoss


n, p = 0, 0
mWIF, mWIFc, mWC, mWO, mWOc, mWhy = np.zeros_like(WIF), np.zeros_like(WIFc), np.zeros_like(WC), np.zeros_like(WO), np.zeros_like(WOc), np.zeros_like(Why)
mby = np.zeros_like(by) # memory variables for Adagrad or momentum
#smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
#with open(('NewModels/NewModelnoUpdateAtStartMom0_400-36'), 'rb') as f1:
 #     model = pickle.load(f1)

#WIF = model['WIF']
#WIFc = model['WIFc']
#WC = model['WC']
#WO = model['WO']
#WOc = model['WOc']
#Why = model['Why']
#by = model['by']

t0 = time.clock()
iterVloss = [] #for storing number of iterations with training and test losses at regular intervals as training progresses
allStats = [] #for storing stats on sample text produced at regular intervals as training progresses.
total_loss = 0
#momentumUpdate = 0

while True:
   
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((h)) # reset hidden state memory   
    cprev = np.zeros((h)) #reset memory cell memory
    p = 0 # go from start of data   
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  
  #hprev1 = np.copy(hprev)
  #cprev1 = np.copy(cprev)
  
  # sample from the model now and then
  if n % 1000 == 0:
        sample_ix = sample(10000)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        stats = getStats(txt)
        stats.insert(0, n)
        allStats.append(stats)
        
        if n % 10000 == 9000: #save both stats and weights every 10,000 iterations. Creates a new file every time so all is not lost if the pickling process is interupted.
            with open((outputFile + str(n/10000)), 'wb') as f1:
                pickle.dump(iterVloss, f1)
                pickle.dump(allStats, f1)
                pickle.dump( '----\n %s \n----' % (txt, ), f1)
        
            model = {'WIF': WIF, 'WIFc': WIFc, 'WC': WC, 'WO': WO, 'WOc': WOc, 'Why': Why, 'by':by}
            with open((modelOutput + str(n/10000)), 'wb') as f:
                pickle.dump(model, f)  
  
   # forward seq_length characters through the net and fetch gradient
  loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev, True)
  total_loss += loss
   #smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 1000 == 0: 
      if(n>0):
          loss = total_loss/1000 #use average loss instead of smooth_loss as it can be compared to test loss
      print 'iter %d, loss: %f' % (n, loss) # print progress
      testLoss = getTestLoss() #get current loss on test set
      print testLoss      
      thisLoss = [n, loss, testLoss]
      iterVloss.append(thisLoss)
      total_loss = 0
    
   # perform parameter update
  for param, dparam, mem in zip([WIF, WIFc, WC, WO, WOc, Why, by], 
                                [dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby], 
                                [mWIF, mWIFc, mWC, mWO, mWOc, mWhy, mby]):
                                    
    param += -learning_rate*dparam
    
    if momentum > 0:
        if(n>0):
            param += momentum*copy.copy(mem) + -momentum*learning_rate*dparam
        else:
            param += -learning_rate*dparam
        mem = momentum*mem + -momentum*learning_rate*dparam
   
    #param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    
  #loss, dWIF, dWIFc, dWC, dWO, dWOc, dWhy, dby, hprev, cprev = lossFun(inputs, targets, hprev1, cprev1, False)
  p += seq_length # move data pointer
  n += 1 # iteration counter 
  learning_rate *= learning_rate_decay
  