#!/usr/bin/env python
# -*- coding: utf-8 -*-


from math import exp, log
import re
import json
from os import listdir
from os.path import isfile, join
import pymorphy2
from supportFunction import mergeDict, readDictFromFile, writeDictToFile, mergeNestedDict, isStopWord, listsum


# Можно ли как-то избежать такого пути?
mypath = '/Users/mihailageev/BayesClassifier/train_text/all_texts'
modelpah = '/Users/mihailageev/BayesClassifier/model'
morph = pymorphy2.MorphAnalyzer()

# Тест
def getwords(doc):
    splitter = re.compile(r'\W')

    decodableText = splitter.split(doc)


    words = []
    for (s) in decodableText:
        # print("wait...")
        word = morph.parse(s.lower())[0]
        normalWord = word.normal_form
        if len(s) > 2 and len(s) < 20 :
                partOfSpeech = word.tag.POS
                if partOfSpeech != "NPRO" and partOfSpeech != "CONJ" and partOfSpeech != "PREP" and partOfSpeech != "PRCL" and partOfSpeech != "INTJ" and partOfSpeech != "PRCL" and partOfSpeech != None:
                    words.append(normalWord)

    return dict([(morph.parse(w)[0].normal_form, 1) for w in words])

def sampletrain(cl):

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        f = open(mypath + '/' + file, encoding="utf8", errors='ignore')
        text = f.read()

        file = file.split('_')
        title = (file[0])
        cl.train(text, title)

    ccDict = readDictFromFile(modelpah + '/' + 'cc')
    fcDict = readDictFromFile(modelpah + '/' + 'fc')

    ccFinal = mergeDict(ccDict, cl.cc)
    fcFinal = mergeNestedDict(fcDict, cl.fc)

    writeDictToFile(ccFinal, modelpah + '/' +'cc')
    writeDictToFile(fcFinal, modelpah + '/' +'fc')


def crossValidationTrain(cl, textsForTrain):

    print("train begin")
    i = len(textsForTrain)
    for file in textsForTrain:

        f = open(mypath + '/' + file, encoding="utf8", errors='ignore')
        text = f.read()

        file = file.split('_')
        title = (file[0])
        cl.train(text, title)
        i -= 1
        print("Text left: " + str(i))

    # clean file with empty list
    writeDictToFile({}, modelpah + '/' +'cc')
    writeDictToFile({}, modelpah + '/' +'fc')

    writeDictToFile(cl.cc, modelpah + '/' +'cc')
    writeDictToFile(cl.fc, modelpah + '/' +'fc')
    print("train end")


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
        # else:
            # sampletrain(self)

        probs = {}
        # Найти категорию с максимальной вероятностью
        max = 0.0
        ver = []
        setMax = False
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)

            ver.append(probs[cat])

            if setMax == False:
                setMax = True
                max = probs[cat]
            if probs[cat] >= max:
                max = probs[cat]
                best = cat

        res = 1 / ( 1 + exp(listsum(ver) - max))
        newVer = 1 - (max / listsum(ver))

        # Убедиться, что найденная вероятность больше чем threshold*следующая по
        # величине
        for cat in probs:
            if cat == best: continue
            if probs[cat] * self.getthreshold(best) > probs[best]: return default

        if rightAnswer != best:
            return "Ошибка при классификации!!! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best + " c вероятностью " + str(newVer)

        return "Классификатор сработал верно! Правильный ответ: " + rightAnswer + " Классификатор определил: " + best + "c вероятностью " + str(newVer)

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
             g = float(self.cc[cat])
             return float(self.cc[cat])
         return 0

    # Общее число образцов
    def totalcount(self):
        h =(sum(self.cc.values( )))
        return sum(self.cc.values())

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

        totals = sum([self.fcount(f, c) for c in self.categories()])
        sumWordInClass = 0
        if self.fc.get(f):
            for key, value in self.fc[f].items():
                if cat == key:
                    sumWordInClass = value

        bp = ((weight) + (sumWordInClass)) / (totals + len(self.fc))

        return bp

class bayes(classifier):
    def docprob(self, item, cat):
        features = self.getfeatures(item)
        # Перемножить вероятности всех признаков
        p = 1
        ver = 1
        for f in features:
            ver += log(self.weightedprob(f, cat, self.fprob))
            # p *= self.weightedprob(f, cat, self.fprob)
        # print(ver)
        return ver

    def prob(self, item, cat):
        catprob = log(self.catcount(cat) / self.totalcount())
        docprob = self.docprob(item, cat)
        return docprob + catprob


