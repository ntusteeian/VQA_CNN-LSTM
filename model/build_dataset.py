import numpy as np
import os

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

INPUT_DIR = '../data'

class VQADataset(Dataset):

    def __init__(self, input_dir, input_file, max_qu_len = 30, transform = None):

        self.input_data = np.load(os.path.join(input_dir, input_file), allow_pickle=True)
        self.qu_vocab = Vocab(input_dir+'/question_vocabs.txt')
        self.ans_vocab = Vocab(input_dir+'/annotation_vocabs.txt')
        self.max_qu_len = max_qu_len
        self.labeled = True if not "test" in input_file else False
        self.transform = transform

    def __getitem__(self, idx):

        path = self.input_data[idx]['img_path']
        img = np.array(Image.open(path).convert('RGB'))
        qu_id = self.input_data[idx]['qu_id']
        qu_tokens = self.input_data[idx]['qu_tokens']
        qu2idx = np.array([self.qu_vocab.word2idx('<pad>')] * self.max_qu_len)
        qu2idx[:len(qu_tokens)] = [self.qu_vocab.word2idx(token) for token in qu_tokens]
        sample = {'image': img, 'question': qu2idx, 'question_id': qu_id}

        if self.labeled:
            ans2idx = [self.ans_vocab.word2idx(ans) for ans in self.input_data[idx]['valid_ans']]
            ans2idx = np.random.choice(ans2idx)
            sample['answer'] = ans2idx

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.input_data)

def data_loader(input_dir, batch_size, max_qu_len, num_worker):

    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
    ])

    vqa_dataset = {
        'train': VQADataset(
            input_dir=input_dir,
            input_file='train.npy',
            max_qu_len=max_qu_len,
            transform=transform),
        'val': VQADataset(
            input_dir=input_dir,
            input_file='val.npy',
            max_qu_len=max_qu_len,
            transform=transform)
    }

    dataloader = {
        key: DataLoader(
            dataset=vqa_dataset[key],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_worker)
        for key in ['train', 'val']
    }

    return dataloader

class Vocab:

    def __init__(self, vocab_file):

        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):

        with open(vocab_file) as f:
            vocab = [v.strip() for v in f]

        return vocab

    def word2idx(self, vocab):

        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

    def idx2word(self, idx):

        return self.vocab[idx]
