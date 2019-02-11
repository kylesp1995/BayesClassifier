import json
from collections import Counter

def mergeDict(dict1, dict2):
    inp = [dict(x) for x in (dict1, dict2)]
    count = Counter()
    for y in inp:
        count += Counter(y)
    return count

def writeDictFromFile(filePath):
    with open(filePath) as f:
        data = json.load(f)
    return data

def writeDictToFile(dict, path):
    with open(path, 'w') as file:
        file.write(json.dumps(dict))