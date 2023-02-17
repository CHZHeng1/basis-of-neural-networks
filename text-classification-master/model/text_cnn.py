import torch
from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        """
        CNN用于文本分类
        :param vocab_size: 词表大小
        :param embedding_dim: 经过embedding转换后词向量的维度
        :param filter_size: 卷积核的大小
        :param num_filter: 卷积核的个数
        :param num_class: 分类类别数
        """
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # padding=1 表示在卷积操作之前，将序列的前后各补充1个输入，这里没有找到详细的解释，考虑是为了让卷积核充分学习序列的信息
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        # 输入数据维度 （batch_size, max_src_len）
        embedding = self.embedding(inputs)
        # 经过embedding层后，维度变为（batch_size, max_src_len, embedding_dim）
        # 但卷积层要求输入数据的形状为（batch_size, in_channels, max_src_len）,因此这里对输入数据进行维度交换
        # embedding.permute(0,2,1) 交换embedding中维度2和1的位置
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        # 经过卷积层输出以后的维度为（batch_size, out_channels, out_seq_len）
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])  # 池化层聚合
        # 经过池化层输出以后的维度为（batch_size,out_channels,1）
        # 但由于全连接层要求输入数据的最后一个维度为卷积核的个数，即out_channels,因此这里需要降维
        probs = self.linear(pooling.squeeze(dim=2))
        return probs


class CNN(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=128, n_filters=50, filter_sizes=None, dropout=0.5, output_dim=150,
                 pad_idx=0):
        """
        包含三种卷积核宽度的卷积神经网络
        :param vocab_size: 词表大小
        :param embedding_dim: 经过embedding转换后词向量的维度
        :param n_filters: 输出通道个数（卷积核的个数）
        :param filter_sizes: 卷积核的宽度 [3, 4, 5]
        :param output_dim:  输出层个数
        :param dropout:
        :param pad_idx:
        """
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.activate = F.relu
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, inputs):
        """
        :param inputs: [batch size, max_src_len]
        :return:
        """
        embedding = self.embedding(inputs)  # [batch_size, max_src_len, embedding_dim]
        embedding = embedding.permute(0, 2, 1)  # [batch_size, embedding_dim, max_src_len]
        # 单个Conv: [batch_size, out_channels, out_seq_len]
        convolutions = [self.activate(conv(embedding)) for conv in self.convs]
        # 单个pooling: [batch_size, out_channels]
        pooling = [F.max_pool1d(convolution, kernel_size=convolution.shape[2]).squeeze(dim=2) for convolution in
                   convolutions]
        outputs = self.dropout(torch.cat(pooling, dim=1))  # [batch_size, out_channels * len(filter_sizes)]
        outputs = self.fc(outputs)  # [batch_size, out_dim]
        return outputs

