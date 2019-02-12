#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import re
import json
from os import listdir
from os.path import isfile, join
from collections import Counter
from supportFunction import mergeDict, writeDictFromFile, writeDictToFile, mergeNestedDict


# Можно ли как-то избежать такого пути?
mypath = '/Users/mihailageev/BayesClassifier/train_text/1'
modelpah = '/Users/mihailageev/BayesClassifier/model'

# Тест
def getwords(doc):
    splitter = re.compile(r'\W')
    # splitter = re.compile(r'\W*')
    # Разбить на слова по небуквенным символам
    decodableText = splitter.split(doc)
    words = [s.lower() for (s) in decodableText
             if len(s) > 2 and len(s) < 20]
    # Вернуть набор уникальных слов
    return dict([(w, 1) for w in words])

def sampletrain(cl):

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        f = open(mypath + '/' + file, encoding="utf8", errors='ignore')
        text = f.read()

        file = file.split('_')
        title = (file[0])
        cl.train(text, title)

    ccDict = writeDictFromFile(modelpah + '/' +'cc')
    fcDict = writeDictFromFile(modelpah + '/' +'fc')

    ccFinal = mergeDict(ccDict, cl.cc)
    fcFinal = mergeNestedDict(fcDict, cl.fc)

    writeDictToFile(ccFinal, modelpah + '/' +'cc')
    writeDictToFile(fcFinal, modelpah + '/' +'fc')

    # with open(modelpah + '/' +'cc', 'w') as file:
    #     file.write(json.dumps(cl.cc))
    #
    # with open(modelpah + '/' +'fc', 'w') as file:
    #     file.write(json.dumps(cl.fc))



class classifier:
    def __init__(self, getfeatures, filename=None):
        # Счетчики комбинаций признак/категория
        self.fc = {}
        # Счетчики документов в каждой категории
        self.cc = {}
        self.getfeatures = getfeatures
        self.thresholds = {}

    # def __init__(self, getfeatures):
    #     classifier.__init__(self, getfeatures)
    #     self.thresholds = {}

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds: return 1.0
        return self.thresholds[cat]

    def classify(self, withTrain, item, default = None, rightAnswer = None):
        if withTrain == False:
            with open(modelpah + '/' + 'cc') as f:
                self.cc = json.load(f)
            with open(modelpah + '/' + 'fc') as f:
                self.fc = json.load(f)

        probs = {}
        # Найти категорию с максимальной вероятностью
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        # Убедиться, что найденная вероятность больше чем threshold*следующая по
        # величине
        for cat in probs:
            if cat == best: continue
            if probs[cat] * self.getthreshold(best) > probs[best]: return default

        if rightAnswer != best:
            return "Ошибка при классификации!!! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best

        return "Классификатор сработал верно! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best

    # Увеличить счетчик пар признак/категория
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    # Увеличить счетчик применений категории
    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    # Сколько раз признак появлялся в данной категории
    def fcount(self,f,cat):
         if f in self.fc and cat in self.fc[f]:
           return float(self.fc[f][cat])
         return 0.0

    # Сколько образцов отнесено к данной категории
    def catcount(self,cat):
         if cat in self.cc:
           return float(self.cc[cat])
         return 0

    # Общее число образцов
    def totalcount(self):
         return sum(self.cc.values( ))

    # Список всех категорий
    def categories(self):
         return self.cc.keys( )


    def train(self, item, cat):
        features = (self.getfeatures(item))
        # Увеличить счетчики для каждого признака в данной классификации
        for f in features:
            self.incf(f, cat)
        # Увеличить счетчик применений этой классификации
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0: return 0
        # Общее число раз, когда данный признак появлялся в этой категории,
        # делим на количество образцов в той же категории
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        # Вычислить текущую вероятность
        basicprob = prf(f, cat)
        # Сколько раз этот признак встречался во всех категориях
        totals = sum([self.fcount(f, c) for c in self.categories()])
        # Вычислить средневзвешенное значение
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp

class bayes(classifier):
    def docprob(self, item, cat):
        features = self.getfeatures(item)

        # Перемножить вероятности всех признаков
        p = 1
        for f in features: p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob


