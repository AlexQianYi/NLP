#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:58:16 2018

@author: yiqian
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_validate

def getCorpus(file):
    
    corpus = []
    
    for sentence in file:
        if sentence == '\n':
            continue
        corpus.append(sentence[:-1])
    
    return corpus

def tenCrossVali(posCorpus, negCorpus):
    
    y = [1 for i in range(len(posCorpus))]
    y += [0 for i in range(len(negCorpus))]
    
    Corpus = posCorpus + negCorpus
    
    # with unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    
    model = vectorizer.fit_transform(Corpus)
    X = model.toarray()
    
    clf = MultinomialNB(alpha=1.0)
    
    print('========Accuracy=========')
    scorces1 = cross_validate(clf, X, y, cv=10, scoring='accuracy')
    print(scorces1)
    print('===========F1============')
    scorces2 = cross_validate(clf, X, y, cv=10, scoring='f1_macro')
    print(scorces2)


with open('CR/txt/pos.tok') as posFile:
    pos = posFile.readlines()

posCorpus = getCorpus(pos)
    
with open('CR/txt/neg.tok') as negFile:
    neg = negFile.readlines()
negCorpus = getCorpus(neg)

tenCrossVali(posCorpus, negCorpus)