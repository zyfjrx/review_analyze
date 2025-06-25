import torch.nn as nn
import config
import torch

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_idx=0):
        super(ReviewAnalyzeModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM,padding_idx=padding_idx)
        # self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,num_layers=6,batch_first=True)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,num_layers=config.NUM_LAYERS,batch_first=True,bidirectional=True)
        self.linear = nn.Linear(2* config.HIDDEN_SIZE,1)

    def forward(self,x):
        # x.shape [batch_size, seq_len]
        embedding = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        # 单层单向LSTM
        # lstm_out, (h_n,c_n) = self.lstm(embedding) # lstm_out.shape [batch_size,seq_len, hidden_dim]
        # last_hidden = lstm_out[:,-1,:] # last_hidden.shape [batch_size, hidden_dim]
        # 双层双向LSTM
        # lstm_out.shape [batch_size,seq_len, 2*hidden_dim]
        # h_n.shape [2 * num_layers,batch_size, hidden_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedding)
        # 取最后一层正向的最后一步隐藏状态
        last_hidden_forward = h_n[-2]
        # 取最后一层反向的最后一步隐藏状态
        last_hidden_backward = h_n[-1]
        # 拼接正反向隐藏状态
        last_hidden = torch.cat((last_hidden_forward, last_hidden_backward), dim=1)
        out = self.linear(last_hidden) # out.shape [batch_size, 1]
        return out.squeeze(1) # out.shape [batch_size]