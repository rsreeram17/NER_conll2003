# LSTM model for entity recognition on conll2003

This repository is a try to build a LSTM based entity recognition model for conll2003 dataset. The code is completely written in Pytorch. Some ideas are derived from the paper 

[NER with bi-LSTM-CNNs]: https://arxiv.org/pdf/1511.08308.pdf



## Approach

- As a first step, a basic LSTM model (with one hidden layer) is used
- Word level embedding and word casing mapping are used as input features
- Negative likelihood loss is the cost function used
- Adam optimizer is the optimizer used

### Dataset preparation and feature extraction

The data available was three .txt files. *train.txt* was used for training, *valid.txt* was used for validation and *test.txt* was used to evaluate the model. Each of the files contained the sentences separated by spaces. Each token had the POS tag and the entity tag. 

Sentences are broken into individual sentences using the space and \n delimiters. (Logic can be seen in *get_data()*). The functions throws out the entire training data separated into different sentences along with the casing index (explained below) for each token and correct label index.

#### Features extraction:

Two input features are used as input:

1. Token embedding
2. Casing embedding

*1.Token embedding*

A vocab is created using the entire train, valid and test data. There are 30298 words in the vocab and this is used for indexing the token in each sentence to look up for the relevant embedding from the embedding table. The embedding table is a random embedding generated. For further iterations pre trained embeddings can be used. 

*2.Casing embedding*

Each of the tokens is given a casing index to extract the information based on the casing pattern of the token. As a first step 5 different casing categories are used: numeric, all lower, all upper, initial upper, others. This casing index is one hot encoded to convert to a feature and used as input along with the token embedding.

#### Dataset preparation:

The get_data() returns the token indices, case indices and label indices for each sentence. This is then converted to the required format for training, validation using the get_training_data (). 

*Dataset* and *dataloader* wrappers are used to prepare the dataset and batch the data into the model.

The token indices are vectorized in the dataset wrapper using a random embedding of 100 dimension and the one hot casing embedding (5 dimensions). These **two embeddings are concatenated into a 105 dimension feature** and this is the input feature for each token.

Batching of data is done suing the dataloader wrapper. Sentence embeddings and label vector in a batch were padded to the length sentence with max. length. This was done using the *pad_sequence()*. Details can be found in *collate_fn ()*.

The input format for the model was a tensor of **shape (N,l,e), where N = batch size, l = length of sentences, e = embedding dimension**

### Model details

The model is a basic LSTM model with a single hidden layer of dimension 128. The padded sequences are packed before being fed into the model. This is to make sure that the padded tokens are not used for back propogation. The same is also excluded while calcuting the loss.

### Hyperparameters used

Most of the hyperparameters used are based on standard/general ways of starting the modelling exercise. 

Embedding dimension for token embedding: 100

Started with 32, then 64 and then 100. This was a gradual increase based on the better performance of the model. 

Batch size: 10,64,128,512,1024

1024 was used as the final batch size to faster the model running. There was a slight trade off on the the model performance here.





