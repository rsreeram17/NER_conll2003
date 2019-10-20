import os
import re
import pandas as pd

def get_data(filename,dir_path,caselookup,labels_lookup):
 
    f = open(dir_path+"\\Data\\"+filename)
    sentences_labels = []
    sentence = []  
    casing_info = []
    sentences_casing = []
    vocab = {}
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences_labels.append(sentence)
                sentence = []
                sentences_casing.append(casing_info)
                casing_info = []
            continue
        delimiters = " ","\n"
        regexPattern = '|'.join(map(re.escape, delimiters))
        splits = re.split(regexPattern, line)
        #splits = pattern.split(line)
        label_id = labels_lookup[splits[-2]]
        sentence.append([splits[0],label_id])
        casing = casingmap(splits[0],caselookup)
        casing_info.append([splits[0],casing])     

    if len(sentence) >0:
        sentences_labels.append(sentence)
        sentences_casing.append(casing_info)
        sentence = []
        casing_info= []
        
    return sentences_labels,sentences_casing

def casingmap(word,caselookup):
   
    casing = 'others'
    
    if word.isdigit():
        casing = 'numeric'
    elif word.islower():
        casing = 'alllower'
    elif word.isupper():
        casing = 'allupper'
    elif word[0].isupper():
        casing = 'initupper'

    return caselookup[casing]

def get_word_to_ix(sentences_labels,sentence_labels_val,sentence_labels_test):
    word_to_ix = {}
    for sentence in sentences_labels:
        for token in sentence:
            if token[0] not in word_to_ix:
                word_to_ix[token[0]] = len(word_to_ix)
    for sentence in sentence_labels_val:
        for token in sentence:
            if token[0] not in word_to_ix:
                word_to_ix[token[0]] = len(word_to_ix)
    for sentence in sentence_labels_test:
        for token in sentence:
            if token[0] not in word_to_ix:
                word_to_ix[token[0]] = len(word_to_ix)
    return word_to_ix

def get_training_data(sentences_labels,sentences_casing,word_to_ix):
  
    training_data = []
    for i,sentence in enumerate(sentences_labels):
        idxs = []
        casing_idxs = []
        labels_idxs = []
        for j,token in enumerate(sentence):     
            idxs.append(word_to_ix[token[0]])
            casing_idxs.append(sentences_casing[i][j][1])
            labels_idxs.append(token[1])
            
        sample_tuple = (idxs,casing_idxs,labels_idxs)
        training_data.append(sample_tuple)
    
    return training_data
            

    
    












