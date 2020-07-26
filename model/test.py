import os
import json
import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from model import VQAModel
from build_dataset import VQADataset

device = torch.device('cuda')
data_dir = '../data'
ckpt_dir = '../ckpt/best_model.pth'
res_dir = '../Results'

BATCH_SIZE = 128
FEATURE_SIZE, WORD_EMBED = 1024, 300
MAX_QU_LEN, NUM_HIDDEN, HIDDEN_SIZE = 30, 2, 512

def test(input_dir, data_type, batch_size, num_worker):

    """
    results = [result]
    result{ "question_id": int,
            "answer": str }......
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # mean=0; std=1
    ])
    vqa_dataset = VQADataset(input_dir, f'{data_type}.npy', max_qu_len=MAX_QU_LEN, transform= transform)
    dataloader = DataLoader(vqa_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    qu_vocab_size = vqa_dataset.qu_vocab.vocab_size
    ans_vocab_size = vqa_dataset.ans_vocab.vocab_size

    model = VQAModel(feature_size=FEATURE_SIZE, qu_vocab_size=qu_vocab_size, ans_vocab_size=ans_vocab_size,
                     word_embed=WORD_EMBED, hidden_size=HIDDEN_SIZE, num_hidden=NUM_HIDDEN).to(device)
    model.load_state_dict(torch.load(ckpt_dir))
    model.eval()
    results = []

    for idx, sample in enumerate(dataloader):

        image = sample['image'].to(device)
        question = sample['question'].to(device)
        question_id = sample['question_id'].tolist()

        with torch.no_grad():
            logits = model(image, question)
            predict = torch.log_softmax(logits, dim=1)

        predict = torch.argmax(predict, dim=1).tolist()
        predict = [vqa_dataset.ans_vocab.idx2word(idx) for idx in predict]
        ans_qu_pair = [{'answer': ans, 'question_id': id} for ans, id in zip(predict, question_id)]
        results.extend(ans_qu_pair)
        if (idx+1) % 50 == 0:
            print(f'finishing {data_type} set : {(idx+1)*batch_size} / {len(vqa_dataset)}')

    if not os.path.exists(res_dir): os.makedirs(res_dir)
    with open(os.path.join(res_dir, f'v2_OpenEnded_mscoco_{data_type}2014_results.json'), 'w') as f:
        f.write(json.dumps(results))

if __name__ == "__main__":

    test(data_dir, 'val', batch_size=BATCH_SIZE, num_worker=8)