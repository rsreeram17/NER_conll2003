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

**Embedding dimension for token embedding: 100**

Started with 32, then 64 and then 100. This was a gradual increase based on the better performance of the model. 

**Batch size: 10,64,128,512,1024**

1024 was used as the final batch size to faster the model running. There was a slight trade off on the the model performance here.

**Number of epochs: 10**

Selected based on model running time

**Optimizer and parameters: Adam optimizer with default parameters** 

SGD with momentum (lr = 0.001, mom = 0.9) was used in the first iteration, but Adam with default parameters gave better performance.

**Learning rate scheduler: period of lr decay - 7 steps and gamma 0.1 (Default)**

Default used for the initial iterations

### Effect of batch size in model performance:

As the batch size went really small the model started taking longer time to converge and also the model running time increased a lot. As batch size increased the model started converging faster and for really large batches the model did not converge. This is primarily because as batch sizes become extremely small there will be more weight updates and this will make the update erratic and convergence tougher. As the batch size is very large, the weight updates are really less and this makes it difficult for the model to converge. A batch size of 256 & 512 gave the best performance out of the tried combinations.

### Precision, Recall and F score

Average precision: 0.21

Average recall: 0.30

Fscore = 0.25

These scores are the scores from the first iteration of testing and without any debugging. Because of limited time, I am reporting the scores directly without debugging and finetuning

### Other modelling options to explore

The model used here is a basic LSTM model. Other than this some of the other models that can be used are:

- **bidirectional LSTM:** This would work better because of the additional input of information from the other side of the sentence and there would be more context for the model to learn from
- **bi-directional GRU:** This should work as good as an LSTM model and contains less complexity
- **pre-trained embeddings:** Use of pre trained embeddings to embed the tokens will definitely improve the model because of the captured context in the embeddings
- **Character level feature:** A character level feature can be added as an additional input along with the token and case embeddings to add more information to the model. (Idea from the above referenced paper)

### Problems faced during implementation

**Variable sized inputs:** Since the sentences were of different lengths, while batching the shorter sentences had to be padded. This was done using the pad_sequence function and later when the batch was fed into the model the padded sentences had to be packed. This was done using the pack_padded_sequence function. And later the outputs had to be padded back. This is mainly done to not consider the padded input for back propogation. 

**Model getting stuck at local minima:** SGD with momentum was the choice made for the first few iterations and the optimizer seemed to be not training the model without getting stuck on a local minima. Later, Adam was used to rectify this problem. 

**Model not learning in the first iteration:** After the entire data preparation and model building pipeline was completed, when running the model, the model was not learning at each epoch. This was then solved by debugging the data preparation pipeline in the first place. And later the model building pipleline was debugged. The issue of not calculating loss correctly was then resolved.

### Effect of imbalanced dataset

One thing that I observed is that, there is a very big skew towards the tokens with entity 'O'. Since there are a lot of tokens with label 'O' the model is not able to learn the pattrern for the remaining classes. This can be visible in the individual entity precision scores where label 'O' has very high precision because of this class imbalance.

