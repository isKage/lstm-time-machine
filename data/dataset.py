try:
    from .load_time_machine import LoadTimeMachine
except ImportError:
    from data.load_time_machine import LoadTimeMachine

import random
import torch


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列
    :param corpus: 转换为数字索引的列表
    :param batch_size: 批量大小
    :param num_steps: 时间步
    """
    offset = random.randint(0, num_steps - 1)  # 随机取一个开始点
    corpus = corpus[offset:]  # -1 是因为 y 要后移一位，防止溢出

    num_subseqs = (len(corpus) - 1) // num_steps  # 最后会产生多少个 X (或 y)

    # 长度为 num_steps (时间步) 的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # 随机抽样: 故打乱这些子序列的起始点
    random.shuffle(initial_indices)

    def sub_list_time_steps(pos):
        # 返回从 pos 位置开始的长度为 num_steps 的子序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size  # batch 个数

    for i in range(0, batch_size * num_batches, batch_size):
        # 每隔 batch_size 就是一个 batch 的起始点
        initial_indices_per_batch = initial_indices[i: i + batch_size]

        # 生成 X, y (batch_size, num_steps)
        X = [sub_list_time_steps(j) for j in initial_indices_per_batch]
        Y = [sub_list_time_steps(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列
    :param corpus: 转换为数字索引的列表
    :param batch_size: 批量大小
    :param num_steps: 时间步
    """
    offset = random.randint(0, num_steps)  # 随机取一个开始点

    # 计算总共会消耗原列表的多长 (len(corpus) - offset - 1) // batch_size 算出每一个 batch 消耗原列表的长度
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size  # -1 是因为 y 要后移一位，防止溢出
    # 这一步是保证了整数倍关系，可以将原序列划分到多个 batch 里

    Xs = torch.tensor(corpus[offset: offset + num_tokens])  # 输入序列 X
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])  # y 后移一位

    # 转换为 (batch_size, num_tokens // batch_size)
    # Xs, Ys 长度仍然是 num_tokens 解决原长度 len(corpus)
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps  # 计算 batch 个数

    for i in range(0, num_steps * num_batches, num_steps):
        # 从每一个 batch 中生成 X, y 此时才是 (batch_size, num_steps)
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(self, data, batch_size, num_steps, use_random_iter):
        if use_random_iter:
            # 随机采样
            self.data_iter_fn = seq_data_iter_random
        else:
            # 顺序采样
            self.data_iter_fn = seq_data_iter_sequential

        self.corpus, self.vocab = data.get_corpus(), data.get_vocab()

        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


if __name__ == '__main__':
    file_path = "../the-time-machine.txt"

    data = LoadTimeMachine(file_path=file_path, token='word', min_freq=0, reserved_tokens=None)
    vocab = data.get_vocab()

    token_idx_dict = vocab.get_token_to_idx()
    corpus = data.get_corpus()

    dataloader = SeqDataLoader(data=data, batch_size=2, num_steps=5, use_random_iter=False)
    for X, y in dataloader:
        print("X {}\ny {}".format(X.shape, y.shape))
        break
        # X torch.Size([2, 5]),
        # y torch.Size([2, 5])
