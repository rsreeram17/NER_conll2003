from statistics import mean
from data_batching import Datasetclass
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence

def testing(testing_data,model,word_to_ix,caselookup):

    dataset_test = Datasetclass(testing_data,word_to_ix,caselookup)
    
    pred_combined = []
    labels_combined = []
    
    for i in range(len(dataset_test)):
        
        input_data,labels = dataset_test[i]
        input_data = input_data[None,:,:]
        length = [input_data.shape[1]]
        
        model.eval()
        outputs = model(input_data,length)
        _,preds = torch.max(outputs,2)
        
        preds_list = preds[0,:].tolist()
        labels_list = labels.tolist()
        
        pred_combined.extend(preds_list)
        labels_combined.extend(labels_list)
        
    precision_list = []
    recall_list = []
    
    for i in range(9):
        
        indices = [index for index,value in enumerate(pred_combined) if value == i]
        actual_labels = [labels_combined[j] for j in indices]
        number_correct_class = actual_labels.count(i)
        precision = number_correct_class/len(actual_labels)
        precision_list.append(precision)
        
        indices = [index for index,value in enumerate(labels_combined) if value == i]
        pred_labels= [pred_combined[j] for j in indices]
        number_correct_class = pred_labels.count(i)
        recall = number_correct_class/len(pred_labels)
        recall_list.append(recall)
    
    average_precision = mean(precision_list)
    average_recall = mean(recall_list)
    
    fscore = (2*average_precision*average_recall)/(average_precision + average_recall)
    
    return average_precision,average_recall,fscore,precision_list,recall_list



