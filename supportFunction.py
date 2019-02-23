import json
from collections import Counter
import ast

def mergeDict(dict1, dict2):
    inp = [dict(x) for x in (dict1, dict2)]
    count = Counter()
    for y in inp:
        count += Counter(y)
    return count

def mergeNestedDict(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                mergeNestedDict(a[key], b[key], path + [str(key)])
            else:
                a[key] += b[key]
        else:
            a[key] = b[key]
    return a

def readDictFromFile(filePath):
    with open(filePath) as f:
        data = json.load(f)
    return data

def writeDictToFile(dict, path):
    with open(path, 'w') as file:
        file.write(json.dumps(dict))



def isStopWord(wordForChecking):
    path = '/Users/mihailageev/BayesClassifier/stop_words.txt'
    with open(path, 'r', encoding="utf16", errors='ignore') as f:
        s = f.read()
        stopWordsDict = ast.literal_eval(s)
    return wordForChecking in stopWordsDict.values()



    # path = '/Users/mihailageev/BayesClassifier/stop_words.txt'
    # with open(path, encoding="utf16", errors='ignore') as f:
    #     stopWord = f.read()
    #
    # stopWord = stopWord.splitlines()
    #
    # cleanStopWord = ([(w) for w in stopWord])
    #
    # for stop in cleanStopWord:
    #     if morph.parse(stop)[0].normal_form == wordForChecking:
    #         print("DETECT STOP WORD!!!!!!!!")
    #         return True
    # return False