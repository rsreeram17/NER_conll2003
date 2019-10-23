from test import *
from data_batching import *
from Model import *

dir_path = os.path.dirname(os.path.realpath(__file__))

##Inference
model_name = 'model_ner.tar'
path = dir_path+"\\"+model_name
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

average_precision,average_recall,fscore,precision_list,recall_list,labels_combined,pred_combined = testing (testing_data,model,word_to_ix,caselookup)


f_score_list = []
for i in range(5):
    f_score_item = (2*precision_list[i]*recall_list[i])/(precision_list[i] + recall_list[i])
    f_score_list.append(f_score_item)
 