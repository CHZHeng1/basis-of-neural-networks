# 此模块用于进行数据的预处理操作，包括字符串替换、繁转简等数据清洗操作，分词，字典构建等等
import re
import pandas as pd
from hanziconv import HanziConv
from collections import defaultdict  # 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值


class Preprocessor:
    """
    此类专门用于进行数据预处理
    """

    @staticmethod
    def basic_pipeline(s):
        """流程化处理"""
        s = Preprocessor.replace_invalid_char(s)
        s = Preprocessor.traditional_to_simplified(s)
        return s

    @staticmethod
    def replace_invalid_char(s):
        """
        将url链接、图片链接、@.. 替换为指定标记
        :param s: 一条样本 -> str
        :return:
        """
        url_token = '<URL>'
        img_token = '<IMG>'
        at_token = '<@ID>'
        s = re.sub(r'(http://)?www.*?(\s|$)', url_token + '\\2', s)  # URL containing www
        s = re.sub(r'http://.*?(\s|$)', url_token + '\\1', s)  # URL starting with http
        s = re.sub(r'\w+?@.+?\\.com.*', url_token, s)  # email
        s = re.sub(r'\[img.*?\]', img_token, s)  # image
        s = re.sub(r'< ?img.*?>', img_token, s)
        s = re.sub(r'@.*?(\s|$)', at_token + '\\1', s)  # @id...
        s = re.sub('\u200B', '', s)
        s = s.strip()
        return s

    @staticmethod
    def traditional_to_simplified(s):
        """繁体转简体"""
        return HanziConv.toSimplified(s.strip())

    @staticmethod
    def build_vocab(train_data_filepath):
        """
        词典实例化
        vocab.idx_to_token  # ['<unk>','希','望','大','家',...]
        vocab.token_to_idx  # {'<unk>': 0, '希': 1, '望': 2, '大': 3, '家': 4,...}
        vocab.convert_tokens_to_ids(['希','望'])  # [1, 2]
        vocab.convert_ids_to_tokens([1,2])  # ['希', '望']
        :param train_data_filepath: 训练集路径 -> str
        :return: 实例化后的词典
        """
        raw_iter = pd.read_csv(train_data_filepath)
        train_sentence = [Preprocessor.basic_pipeline(raw[1]) for raw in raw_iter.values]
        return Vocab.build(train_sentence)


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()  # 词表
        self.token_to_idx = dict()  # 词表及对应单词位置

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 标记每个单词的位置
            self.unk = self.token_to_idx['<unk>']  # 开始符号的位置

    @classmethod
    # 不需要实例化，直接类名.方法名()来调用 不需要self参数，但第一个参数需要是表示自身类的cls参数,
    # 因为持有cls参数，可以来调用类的属性，类的方法，实例化对象等
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小，即词表中有多少个互不相同的标记
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入标记对应的索引值，如果该标记不存在，则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]


if __name__ == '__main__':
    vocab = Preprocessor.build_vocab('../data/train.csv')
    print(len(vocab))
