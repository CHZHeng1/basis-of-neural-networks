# 此模块进行数据的处理、封装等，将会完成模型训练前的所有数据操作
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.data_preprocessor import Preprocessor


class DataMapping(Dataset):
    """数据映射"""
    def __init__(self, data):
        self.dataset = data
        self.lens = len(data)

    def __getitem__(self, index):
        sen, label = self.dataset[index]
        return sen, label

    def __len__(self):
        return len(self.dataset)


class CollateFn:
    """迭代训练前的数据整理"""
    @staticmethod
    def generate_batch_cnn(examples):
        """cnn：对一个批次内的数据进行处理"""
        inputs = [torch.tensor(ex[0]) for ex in examples]
        labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        # 对批次内的样本进行补齐，使其具有相同长度
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, labels

    @staticmethod
    def generate_batch_lstm(examples):
        """rnn：对一个批次内的数据进行处理"""
        lengths = torch.tensor([len(ex[0]) for ex in examples])
        inputs = [torch.tensor(ex[0], dtype=torch.long) for ex in examples]
        labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, lengths, labels


class FakeNewsDataset:
    def __init__(self, train_data_filepath=None, val_data_filepath=None, test_data_filepath=None, batch_size=32,
                 model_name=None):
        self.train_data_filepath = train_data_filepath
        self.val_data_filepath = val_data_filepath
        self.test_data_filepath = test_data_filepath
        self.vocab = Preprocessor.build_vocab(train_data_filepath)
        self.vocab_size = len(self.vocab)
        self.batch_size = batch_size
        self.model_name = model_name

    def data_process(self, data_filepath):
        """数据处理"""
        raw_iter = pd.read_csv(data_filepath)
        data = []
        for raw in tqdm(raw_iter.values, desc='Data Processing'):
            label, s = raw[-1], raw[1]  # 标签和文本
            s = Preprocessor.basic_pipeline(s)
            s_ids = self.vocab.convert_tokens_to_ids(list(s))
            data.append((s_ids, label))
        data = DataMapping(data)
        return data

    def load_data(self, collate_fn=None, only_test=False):
        """封装数据用于迭代训练"""
        test_data = self.data_process(self.test_data_filepath)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                               collate_fn=collate_fn)
        if only_test:
            return test_iter

        train_data = self.data_process(self.train_data_filepath)
        val_data = self.data_process(self.val_data_filepath)

        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                collate_fn=collate_fn)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False,
                              collate_fn=collate_fn)

        return train_iter, val_iter


if __name__ == '__main__':
    train_filepath = '../data/train.csv'
    val_filepath = '../data/val.csv'
    test_filepath = '../data/test.csv'
    dataset = FakeNewsDataset(train_filepath)
    data = dataset.data_process(test_filepath)
