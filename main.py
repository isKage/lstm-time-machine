from data import LoadTimeMachine
from data import SeqDataLoader


def dataloader(file_path, batch_size, num_steps, use_random_iter=False, token='word', min_freq=0,
               reserved_tokens=None):
    data_ = LoadTimeMachine(file_path=file_path, token=token, min_freq=min_freq, reserved_tokens=reserved_tokens)
    vocab_ = data_.get_vocab()

    dataloader_ = SeqDataLoader(data=data_, batch_size=batch_size, num_steps=num_steps, use_random_iter=use_random_iter)
    del data_

    return dataloader_, vocab_


if __name__ == '__main__':
    file_path = "the-time-machine.txt"
    data_iter, vocab = dataloader(file_path=file_path, batch_size=2, num_steps=4)
    for X, y in data_iter:
        print(X.shape, y.shape)
        break
