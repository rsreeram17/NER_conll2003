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

model = LSTMTagger(105,512,23623,10)
sentences,lengths,labels = next(iter(train_data_loader))
op = model(sentences,lengths)

batch_loss = 0
for i in range (op.size(0)):
    nllloss = F.nll_loss(op[i],labels[i],ignore_index=9)
    batch_loss += nllloss