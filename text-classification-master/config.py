import os
import torch


class Config:
    """此类用于定义超参数"""
    def __init__(self):
        self.project_dir = os.getcwd()  # 获取当前脚本所在路径
        self.dataset_dir = os.path.join(self.project_dir, 'data')  # 数据集文件夹
        self.train_data_filepath = os.path.join(self.dataset_dir, 'train.csv')
        self.val_data_filepath = os.path.join(self.dataset_dir, 'val.csv')
        self.test_data_filepath = os.path.join(self.dataset_dir, 'test.csv')
        self.model_save_dir = os.path.join(self.project_dir, 'result')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.0001  # 当batch_size较小时，应适当调大学习率；当模型参数体系较为庞大时，应适当调小学习率

        self.embedding_dim = 128
        self.num_class = 2

        # TextCNN参数
        self.filter_size = 3  # 卷积核宽度
        self.filter_sizes = [3, 4, 5]
        self.num_filter = 50  # 卷积核个数

        self.dropout = 0.5
        self.hidden_dim = 512  # 隐含层参数
        self.num_head = 8  # 多头注意力机制的头数
        self.num_layers = 1  # 编码器的层数
