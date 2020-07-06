import numpy as np
import os

from torch.utils.data.dataset import Dataset
from PIL import Image

INPUT_DIR = '../data'



class VQADataset(Dataset):

    def __init__(self, input_dir, input_file, max_qu_len = 30, max_ans_len = 10):

        self.input_data = np.load(os.path.join(input_dir, input_file), allow_pickle=True)
        self.qu_vocab = Vocab(input_dir+'/question_vocabs.txt')
        self.ans_vocab = Vocab(input_dir+'/annotation_vocabs.txt')
        self.max_qu_len = max_qu_len
        self.max_ans_len = max_ans_len
        self.labeled = True if not "test" in input_file else False

    def __getitem__(self, idx):

        path = self.input_data[idx]['img_path']
        img = np.array(Image.open(path))
        qu_tokens = self.input_data[idx]['qu_tokens']
        qu2idx = np.array([self.qu_vocab.word2idx('<pad>')] * self.max_qu_len)
        qu2idx[:len(qu_tokens)] = [self.qu_vocab.word2idx(token) for token in qu_tokens]
        sample = {'image': img, 'question': qu2idx}

        if self.labeled:
            ans2idx = [self.ans_vocab.word2idx(ans) for ans in self.input_data[idx]['valid_ans']]
            ans2idx = np.random.choice(ans2idx)
            sample['answer'] = ans2idx

        return sample

    def __len__(self):

        return len(self.input_data)


class Vocab:

    def __init__(self, vocab_file):

        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}


    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2idx(self, vocab):

        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

if __name__ == "__main__":

    torchdata = VQADataset(INPUT_DIR, 'train.npy')
    print(torchdata)

    qu_vocab = Vocab(INPUT_DIR+'/question_vocabs.txt')
    print(qu_vocab)
    for i in range(3):
        print(torchdata[i])
