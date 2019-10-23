from torch.optim import lr_scheduler
import torch.optim as optim
import time
import copy
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.nn.utils.rnn import pad_packed_sequence

from statistics import mean

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class LSTMTagger(nn.Module):
    def __init__(self, weights_matrix, hidden_dim_1,hidden_dim_2, case_size, tagset_size):
        super(LSTMTagger, self).__init__()
        
        self.embedding_token, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        #self.embedding_token = nn.Embedding(vocab_size,100)
        self.embedding_case = nn.Embedding(case_size,case_size)
        self.embedding_case.weight.data = torch.eye(case_size)
        #self.hidden_dim = hidden_dim
        self.lstm_1 = nn.LSTM((embedding_dim + case_size), hidden_dim_1,batch_first = True)
        self.lstm_2 = nn.LSTM(hidden_dim_1,hidden_dim_2,batch_first = True)
        self.hidden2tag = nn.Linear(hidden_dim_2, tagset_size)
        
    def forward(self,sequences_tokens,sequences_casing,lengths):
        
        sequences_token_embedding = self.embedding_token(sequences_tokens)
        sequences_casing_embedding = self.embedding_case(sequences_casing)
        
        sequences_embedding_packed = torch.cat((sequences_token_embedding,sequences_casing_embedding),2)  
        
        x_packed = PACK(sequences_embedding_packed,lengths, batch_first=True)
        output_1, _ = self.lstm_1(x_packed)
        output_2,_ = self.lstm_2(output_1)
        output_padded, output_lengths = pad_packed_sequence(output_2, batch_first=True)
        tag_space = self.hidden2tag(output_padded)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train_model(model,optimizer,scheduler,num_epochs,dir_path,data_loaders):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        
        for phase in ['train','val']:
            if (phase == 'train'):
            
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            total_number_of_preds = 0.0
            
            ##Iterate over data
            
            for sequences_tokens,sequences_casing,lengths,labels in data_loaders[phase]:
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(sequences_tokens,sequences_casing,lengths)
                    _,preds = torch.max(outputs,2)
                    number_of_preds = preds.shape[0]*preds.shape[1]
                    loss = 0
                    for i in range (outputs.size(0)):
                        nllloss = F.nll_loss(outputs[i],labels[i],ignore_index=9)
                        loss += nllloss
                    #loss = criterion(outputs,labels)
                    if(phase=='train'):
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss
                running_corrects += torch.sum(preds == labels)
                total_number_of_preds += number_of_preds
                
            epoch_loss = running_loss
            epoch_acc = running_corrects.double() / total_number_of_preds
            
            model_name = 'model_ner.tar'
            path = dir_path+"\\"+model_name
            
            torch.save({
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      }, path)
           
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model
