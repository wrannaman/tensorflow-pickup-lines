import time
import json

file = open("outputs.txt", "r")
allLines = []
for line in file:
    line = line.strip()
    if not line:
        continue;
    if '.' in line:
        line = line.split('.')
        allLines = allLines + line
    elif '!' in line:
        line = line.split('!')
        allLines = allLines + line
    else:
        allLines.append(line)

_allLines = []
for line in allLines:
    if line:
        _allLines.append(line.strip())

data = {}
data['lines'] = _allLines
with open('lines.json', 'w') as outfile:
    json.dump(data, outfile)
