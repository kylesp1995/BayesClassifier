import json
from collections import Counter

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
            elif a[key] == b[key]:
                pass
            else:
                a[key] += b[key] - a[key]
        else:
            a[key] = b[key]
    return a

def writeDictFromFile(filePath):
    with open(filePath) as f:
        data = json.load(f)
    return data

def writeDictToFile(dict, path):
    with open(path, 'w') as file:
        file.write(json.dumps(dict))