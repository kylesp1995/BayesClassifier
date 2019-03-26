#!/usr/bin/env python
# -*- coding: utf-8 -*-

import docclass
import pymorphy2
import json
from supportFunction import mergeDict, readDictFromFile, writeDictToFile, mergeNestedDict
import datetime
from os import listdir
from os.path import isfile, join

pathWritingFile = '/Users/mihailageev/BayesClassifier/res'

resFile = open(pathWritingFile, 'w')

unknownAnswer = 'unknown'

def test_func(cl, onlyfiles):
    mypath = '/Users/mihailageev/BayesClassifier/train_text/all_texts'

    for file in onlyfiles:
        f = open(mypath + '/' + file, encoding="utf8", errors='ignore')
        text = f.read()

        file = file.split('_')
        title = (file[0])
        res = (cl.classify(False, text, default=unknownAnswer, rightAnswer=title)) + '\n'
        resFile.write(res)
        print(res)
        file.clear()


mypath = '/Users/mihailageev/BayesClassifier/train_text/all_texts'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

countTheme = 10
countTexts = len(onlyfiles)

window = 10

textsInTheme = int(countTexts / countTheme)

iteration = int(textsInTheme / countTheme)

for i in range(1, iteration + 1):
    textsForTrain = []
    textsForDetecting = []

    cl = docclass.bayes(docclass.getwords)

    for text in onlyfiles:
        file = text.split('_')
        number = int(file[1])

        if number > (i * window - window) and number <= i * window:
            textsForDetecting.append(text)
        else:
            textsForTrain.append(text)

    docclass.crossValidationTrain(cl, textsForTrain)

    test_func(cl, textsForDetecting)
    resFile.write("************************** \n")

resFile.close()