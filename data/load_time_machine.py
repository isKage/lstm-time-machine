import collections
import string


def load_text(dir: str) -> list:
    """加载图书内容，去除标点符号并转换为小写"""
    with open(dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 去除标点符号并转换为小写
    lines = [line.translate(str.maketrans('', '', string.punctuation)).lower() for line in lines]
    return lines


def tokenize(lines: list, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    """统计词元的频率"""
    # 将词元列表展平成一个列表
    res = []
    for line in tokens:
        for token in line:
            res.append(token)

    # 返回字典：键为词元，值为次数
    return collections.Counter(res)


class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化
        :param tokens: 输入的词元列表（如单词列表）。如果未提供，默认为空列表。
        :param min_freq: 词元的最低频率阈值。低于此频率的词元将被忽略。
        :param reserved_tokens: 保留词元列表（如特殊符号 <unk>、<pad> 等）。如果未提供，默认为空列表。
        """
        if tokens is None:
            # 没有传入内容，则返回空列表
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 按次元出现频率排序，存入 _token_freqs
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # idx_to_token 列表：从下标索引词元/单词
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 未知词元 <unk> 的索引为 0

        # token_to_idx 字典：词元/单词对应到编号/下标
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}  # 未知词元 <unk> 的索引为 0

        for token, freq in self._token_freqs:
            """构建词表"""
            # self._token_freqs 形如 [(词元, 频率), (词元, 频率), ...]
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)  # 将词元加入 (对应位置)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 生成下标

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """将词元 (token) 转换为索引 (index) 数字"""
        if not isinstance(tokens, (list, tuple)):  # 若传入的是单个 token
            # .get(a, b) a 存在返回 a，不存在返回 b
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]  # 否则返回 token 对应 idx 的列表

    def to_tokens(self, idx):
        """将索引转换为词元"""
        if not isinstance(idx, (list, tuple)):  # 若传入的是单个 idx 数字
            return self.idx_to_token[idx]  # 直接返回对应的 token
        return [self.idx_to_token[index] for index in idx]  # 否则返回一系列 token 列表

    def get_token_to_idx(self):
        """对照表"""
        return self.token_to_idx

    @property  # @property 指定为属性，可以不加括号使用 obj.unk
    def unk(self):
        # 设定未知词元的索引为 0
        return 0

    @property
    def token_freqs(self):
        # 获取词频，形如 [(词元, 频率), (词元, 频率), ...]
        return self._token_freqs


class LoadTimeMachine:
    """整合，返回词表类、全书索引列表"""

    def __init__(self, file_path, token='word', min_freq=0, reserved_tokens=None):
        # 1. 读文件 txt 获取列表 [['a line'], ['another line'], ...]
        lines = load_text(file_path)

        # 2. 分词得到 tokens = [['a', 'line'], ['another', 'line'], [], ...]
        tokens = tokenize(lines, token=token)

        # 3. 词表类
        self._vocab = Vocab(tokens, min_freq=min_freq, reserved_tokens=reserved_tokens)

        corpus = []  # 词元索引列表
        for line in tokens:
            for token in line:
                # 得到全书的索引表，每一个数字代表了这个位置的单词在单词表里的索引
                # 可以通过 vocab[num] 查看 num 对应的单词 token
                corpus.append(self._vocab[token])

        self._corpus = corpus

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return self._vocab

    def get_corpus(self):
        return self._corpus


if __name__ == '__main__':
    # 1. 读文件 txt 获取列表 [['a line'], ['another line'], ...]
    lines = load_text('../the-time-machine.txt')

    # 2. 分词得到 tokens = [['a', 'line'], ['another', 'line'], [], ...]
    tokens = tokenize(lines, token='word')
    # for i in range(11):
    #     print(tokens[i])

    # 3. 词表类
    vocab = Vocab(tokens, min_freq=0, reserved_tokens=None)

    corpus = []  # 词元索引列表
    for line in tokens:
        for token in line:
            # 得到全书的索引表，每一个数字代表了这个位置的单词在单词表里的索引
            # 可以通过 vocab[num] 查看 num 对应的单词 token
            corpus.append(vocab[token])

    print(corpus[:20])  # 转换为数字的索引表
    print(vocab)  # 词表类
    print(vocab.token_freqs[:10])  # 每个单词 token 的总频率

    # 可以查看一下 corpus[:20] 的原始句子，对应 the-time-machine.txt 书的开头 20 个单词
    token_list = [vocab.to_tokens(i) for i in corpus[:20]]
    print(token_list)

    """开头 20 个单词
    The Project Gutenberg eBook of The Time Machine
    
    This ebook is for the use of anyone anywhere in the United
    """
