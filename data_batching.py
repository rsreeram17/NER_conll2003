import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence

class Datasetclass (data.Dataset): 
    def __init__(self,data_list,word_to_ix,caselookup):
        super(Datasetclass, self).__init__()
        self.data_list = data_list
        self.no_of_samples = len(self.data_list)
        self.caselookup = caselookup
        self.word_to_ix = word_to_ix
        #self.embedding_token = nn.Embedding(len(word_to_ix),100)
        #self.embedding_case = nn.Embedding(len(self.caselookup),len(self.caselookup))
        #self.embedding_case.weight.data = torch.eye(len(self.caselookup))
        
    def __getitem__(self,index):
        
        data = self.data_list[index]
        label_list = torch.tensor(data[2])
        token_indices_list,token_casing_list = self.vectorize(data)
        
        return token_indices_list,token_casing_list,label_list, len(self.word_to_ix)
         
    def __len__(self):
        
        return self.no_of_samples
    
    def vectorize (self,data):
        
        #data_indices = []
        token_indices_list = torch.LongTensor(data[0])
        token_casing_list = torch.LongTensor(data[1])
        
        #data_indices.append(token_indices_list)
        #data_indices.append(token_casing_list)
        #token_embedding = self.embedding_token(torch.tensor(token_indices_list))
        #case_embedding = self.embedding_case(torch.tensor(token_casing_list))
        #sentence_embedding = torch.cat((token_embedding,case_embedding),1)

        return token_indices_list,token_casing_list
        
def collate_fn (data):
    
    sorted_batch = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
    sequences_token = [x[0] for x in sorted_batch]
    sequences_casing = [x[1] for x in sorted_batch]
    labels = [x[2] for x in sorted_batch]
    len_word_to_ix_list = [x[3] for x in sorted_batch]
    len_w2ix = len_word_to_ix_list[0]
    sequences_tokens_padded = torch.nn.utils.rnn.pad_sequence(sequences_token, batch_first=True,padding_value = len_w2ix )
    sequences_casing_padded = torch.nn.utils.rnn.pad_sequence(sequences_casing, batch_first=True,padding_value = 4)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels,batch_first = True,padding_value = 9)
    lengths = torch.LongTensor([len(x) for x in sequences_token])
    #labels = torch.LongTensor([x for x in labels])
    return sequences_tokens_padded, sequences_casing_padded, lengths, labels_padded


    
        

   
