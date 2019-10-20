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

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first = True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def forward(self,sentences,lengths):
        
        x_packed = PACK(sentences,lengths, batch_first=True)
        output_packed, _ = self.lstm(x_packed)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
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
            
            for sentences,lengths,labels in data_loaders[phase]:
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(sentences,lengths)
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
