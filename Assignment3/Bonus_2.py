#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 18:54:49 2018

@author: yiqian
"""


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
import numpy as np

import math

def getCorpus(file):
    
    corpus = []
    
    for sentence in file:
        if sentence == '\n':
            continue
        corpus.append(sentence[:-1])
    
    return corpus

def getR(posX, negX, Corpus):
    
    p = np.sum(posX, axis=0)
    q = np.sum(negX, axis=0)
    
    np.apply_along_axis(lambda x: x+1, 0, p)
    np.apply_along_axis(lambda x: x+1, 0, q)
    
    p1 = sum(p)+len(p)
    q1 = sum(q)+len(q)
    
    Xr = []
    for i in range(len(p)):
        if p[i] == 0:
            p[i] = 1
        if q[i] == 0:
            q[i] = 1
        temp = (p[i]*q1)/(q[i]/p1)
        Xr.append(math.log(temp, 10))
        
    result = np.multiply(Corpus, Xr)
    return result
    

def tenCrossVali(posCorpus, negCorpus):
    
    y = [1 for i in range(len(posCorpus))]
    y += [0 for i in range(len(negCorpus))]
    
    
    Corpus = posCorpus + negCorpus
    
    # with unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    model = vectorizer.fit_transform(Corpus)
    X = model.toarray()
    
    posL = len(posCorpus)
    
    Xr = getR(X[:posL], X[posL:], X)
    
    log = LogisticRegression(penalty='l2')
    
    print('========Accuracy=========')
    scorces1 = cross_validate(log, Xr, y, cv=10, scoring='accuracy')
    print(scorces1)
    print('===========F1============')
    scorces2 = cross_validate(log, Xr, y, cv=10, scoring='f1_macro')
    print(scorces2)


with open('CR/txt/pos.tok') as posFile:
    pos = posFile.readlines()

posCorpus = getCorpus(pos)
    
with open('CR/txt/neg.tok') as negFile:
    neg = negFile.readlines()
negCorpus = getCorpus(neg)

tenCrossVali(posCorpus, negCorpus)