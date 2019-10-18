import os
import re

dir_path = os.path.dirname(os.path.realpath(__file__))

def readfile(filename,dir_path):
 
    f = open(dir_path+"\\Data\\"+filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        delimiters = " ","\n"
        regexPattern = '|'.join(map(re.escape, delimiters))
        splits = re.split(regexPattern, line)
        #splits = pattern.split(line)
        sentence.append([splits[0],splits[-2]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences

