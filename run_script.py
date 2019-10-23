from data_import import *
from data_batching import *
from Model import *


dir_path = os.path.dirname(os.path.realpath(__file__))

caselookup = {'numeric':0,'alllower':1,'allupper':2,'initupper':3,'others':4}
labels_lookup = {'B-ORG':0,'O':1,'B-MISC':2,'B-PER':3,'I-PER':4,'B-LOC':5,'I-ORG':6,'I-MISC':7,'I-LOC':8}

##Importing the necessary data
sentences_labels,sentences_casing = get_data("train.txt",dir_path,caselookup,labels_lookup)
sentences_labels_val,sentence_casing_val = get_data("valid.txt",dir_path,caselookup,labels_lookup)
sentences_labels_test,sentence_casing_test = get_data("test.txt",dir_path,caselookup,labels_lookup)
word_to_ix = get_word_to_ix(sentences_labels,sentences_labels_val,sentences_labels_test)
training_data = get_training_data(sentences_labels,sentences_casing,word_to_ix)      
validation_data = get_training_data(sentences_labels_val,sentence_casing_val,word_to_ix)
testing_data = get_training_data(sentences_labels_test,sentence_casing_test,word_to_ix)

glove_model = loadGloveModel('glove.6B.50d.txt')
weight_matrix = create_weightmatrix(glove_model,len(word_to_ix)+1,50,word_to_ix)


##Dataset preparation
dataset_training = Datasetclass(training_data,word_to_ix,caselookup)
dataset_val = Datasetclass(validation_data,word_to_ix,caselookup)
train_data_loader = torch.utils.data.DataLoader(dataset = dataset_training,batch_size=128,collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(dataset = dataset_val,batch_size=128,collate_fn=collate_fn)
data_loaders = {}
data_loaders['train'] = train_data_loader
data_loaders['val'] = val_data_loader
data_lengths = {}
data_lengths['train'] = len(dataset_training)
data_lengths['val'] = len(dataset_val)

##Model building
model = LSTMTagger(weight_matrix,512,512,len(caselookup),10)
#optimizer_ft = torch.optim.Adam(model.parameters(), lr=0.001,  betas=(0.9, 0.99), weight_decay=0.0002)
optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.CyclicLR(optimizer_ft, base_lr = 0.000001 ,max_lr = 0.01)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
num_epochs = 30
model = train_model(model,optimizer_ft,exp_lr_scheduler,num_epochs,dir_path,data_loaders)






