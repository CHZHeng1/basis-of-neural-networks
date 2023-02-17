import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, polling_type=None):
        super(LSTM, self).__init__()
        self.polling_type = polling_type
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        # self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        if polling_type == 'last_term':
            self.polling_layer = nn.Linear(hidden_dim, num_class)
        elif polling_type == 'forward_and_backward':
            self.polling_layer = nn.Linear(hidden_dim*2, num_class)
        else:
            raise TypeError

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        # 使用pack_padded_sequence函数将变长序列打包
        x_pack = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        if self.polling_type == 'last_term':
            polling_out = hn[-1]
            outputs = self.polling_layer(polling_out)  # 取反向最后一个时刻输出的数据
        elif self.polling_type == 'forward_and_backward':
            polling_out = torch.cat([hn[0], hn[-1]], dim=1)  # 前向最后一个时刻和反向最后一个时刻输出的数据进行拼接
            outputs = self.polling_layer(polling_out)
        else:
            raise TypeError
        return outputs
