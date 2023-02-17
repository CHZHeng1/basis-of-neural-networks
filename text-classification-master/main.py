import os
from tqdm.auto import tqdm
import torch
from torch import nn
from utils.data_process import FakeNewsDataset, CollateFn
from utils.metrics import cal_precision, cal_recall, cal_f1
from model.text_cnn import TextCNN, CNN
from model.lstm import LSTM
from model.transformer import Transformer
from config import Config


def train(config, model_name=None):
    data_loader = FakeNewsDataset(train_data_filepath=config.train_data_filepath,
                                  val_data_filepath=config.val_data_filepath,
                                  test_data_filepath=config.test_data_filepath,
                                  batch_size=config.batch_size,
                                  model_name=model_name)
    if model_name == 'TextCNN':
        # 模型保存位置
        model_save_path = os.path.join(config.model_save_dir, 'model_text_cnn.pt')
        # 数据
        train_iter, val_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_cnn, only_test=False)
        # 模型
        model = TextCNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                        filter_size=config.filter_size, num_filter=config.num_filter, num_class=config.num_class)
    elif model_name == 'CNN':
        model_save_path = os.path.join(config.model_save_dir, 'model_cnn.pt')
        train_iter, val_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_cnn, only_test=False)
        model = CNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                    n_filters=config.num_filter, filter_sizes=config.filter_sizes, dropout=config.dropout,
                    output_dim=config.num_class)
    elif model_name == 'Bi-LSTM':
        model_save_path = os.path.join(config.model_save_dir, 'model_text_bi_lstm.pt')
        train_iter, val_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_lstm, only_test=False)
        # polling_type = 'last_term' or 'forward_and_backward'
        model = LSTM(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                     hidden_dim=config.hidden_dim, num_class=config.num_class, polling_type='forward_and_backward')
    elif model_name == 'Transformer':
        model_save_path = os.path.join(config.model_save_dir, 'model_transformer.pt')
        train_iter, val_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_lstm, only_test=False)
        model = Transformer(vocab_size=data_loader.vocab_size, num_class=config.num_class, d_model=config.embedding_dim,
                            dim_feedforward=config.hidden_dim, num_head=config.num_head, num_layers=config.num_layers)
    else:
        raise ValueError('模型参数输入有误')

    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()

    # 训练
    max_f = 0
    for epoch in range(config.epochs):
        total_loss = 0
        total_acc = 0
        labels = []  # 整个数据集上的标签
        for batch in tqdm(train_iter, desc=f'Training Epoch {epoch + 1}'):
            if model_name == 'TextCNN' or model_name == 'CNN':
                # 将数据加载至GPU
                inputs, targets = [x.to(config.device) for x in batch]
                # 将特征带入到模型
                probs = model(inputs)
            elif model_name == 'Bi-LSTM' or model_name == 'Transformer':
                inputs, lengths, targets = [x.to(config.device) for x in batch]
                probs = model(inputs, lengths)
            else:
                raise ValueError('模型参数输入有误')

            # 计算损失
            loss = criterion(probs, targets)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            acc = (probs.argmax(dim=1) == targets).sum().item()  # item()用于在只包含一个元素的tensor中提取值
            total_acc += acc  # 最终得到整个epoch的准确率
            total_loss += loss.item()  # 最终得到整个epoch的损失

            batch_labels = targets.tolist()
            labels.extend(batch_labels)

        # 打印的是整个epoch上的样本损失的平均值以及准确率
        print(f'Train Loss:{total_loss:.4f}    Train Accuracy:{total_acc/len(labels):.4f}')

        val_loss, acc, p, r, f, _, _ = evaluate(model, criterion, val_iter, config.device, model_name=model_name)
        print(f'Val Loss:{val_loss:.4f}    Val Accuracy:{acc:.4f}')
        print(f"Val Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate(model, criterion, data_iter, device, model_name=None):
    test_acc = 0
    test_loss = 0
    model.eval()  # 切换到测试模式
    with torch.no_grad():  # 不计算梯度
        y_true, y_pred = [], []
        for batch in data_iter:
            if model_name == 'TextCNN' or model_name == 'CNN':
                inputs, targets = [x.to(device) for x in batch]
                probs = model(inputs)

            elif model_name == 'Bi-LSTM' or model_name == 'Transformer':
                inputs, lengths, targets = [x.to(device) for x in batch]
                probs = model(inputs, lengths)

            else:
                raise ValueError('模型参数输入有误')

            loss = criterion(probs, targets)
            acc = (probs.argmax(dim=1) == targets).sum().item()
            test_acc += acc
            test_loss += loss.item()

            batch_pred = probs.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = targets.tolist()

            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        acc = test_acc / len(y_true)
        p = cal_precision(y_true, y_pred)
        r = cal_recall(y_true, y_pred)
        f = cal_f1(y_true, y_pred)
    model.train()  # 切换到训练模式
    return test_loss, acc, p, r, f, y_true, y_pred


def predict(config, model_name=None):
    """模型效果预测"""
    data_loader = FakeNewsDataset(train_data_filepath=config.train_data_filepath,
                                  val_data_filepath=config.val_data_filepath,
                                  test_data_filepath=config.test_data_filepath,
                                  batch_size=config.batch_size,
                                  model_name=model_name)
    if model_name == 'TextCNN':
        test_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_cnn, only_test=True)
        model = TextCNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                        filter_size=config.filter_size, num_filter=config.num_filter, num_class=config.num_class)
        model_save_path = os.path.join(config.model_save_dir, 'model_text_cnn.pt')

    elif model_name == 'CNN':
        test_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_cnn, only_test=True)
        model = CNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                    n_filters=config.num_filter, filter_sizes=config.filter_sizes, dropout=config.dropout,
                    output_dim=config.num_class)
        model_save_path = os.path.join(config.model_save_dir, 'model_cnn.pt')

    elif model_name == 'Bi-LSTM':
        test_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_lstm, only_test=True)
        # polling_type = 'last_term' or 'forward_and_backward'
        model = LSTM(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                     hidden_dim=config.hidden_dim, num_class=config.num_class, polling_type='forward_and_backward')
        model_save_path = os.path.join(config.model_save_dir, 'model_text_bi_lstm.pt')

    elif model_name == 'Transformer':
        test_iter = data_loader.load_data(collate_fn=CollateFn.generate_batch_lstm, only_test=True)
        model = Transformer(vocab_size=data_loader.vocab_size, num_class=config.num_class, d_model=config.embedding_dim,
                            dim_feedforward=config.hidden_dim, num_head=config.num_head, num_layers=config.num_layers)
        model_save_path = os.path.join(config.model_save_dir, 'model_transformer.pt')

    else:
        raise ValueError('模型参数输入有误')

    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        print('成功载入已有模型，进行预测......')

    model = model.to(config.device)
    # print(list(model.modules()))
    criterion = nn.CrossEntropyLoss()
    test_loss, acc, p, r, f, _, _ = evaluate(model, criterion, test_iter, config.device, model_name=model_name)

    print(f'Test Loss:{test_loss:.4f}    Test Accuracy:{acc:.4f}')
    print(f"Test Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


if __name__ == '__main__':
    torch.manual_seed(2022)
    model_config = Config()
    train(model_config, model_name='CNN')  # TextCNN or CNN or Bi-LSTM or Transformer
    predict(model_config, model_name='CNN')
