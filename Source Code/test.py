# -*- coding: utf-8 -*-
"""test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_YTBsoaNgivAFKtgzrqE7qnxhdi7fJAY
"""

!pip install sentencepiece --quiet
!pip install sacrebleu --quiet
!pip install torchdata --quiet

import math
import os
from dataclasses import dataclass
import numpy as np
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)


train_en_file = '/kaggle/input/anlp-a2/train.en'
train_fr_file = '/kaggle/input/anlp-a2/train.fr'
dev_en_file = '/kaggle/input/anlp-a2/dev.en'
dev_fr_file = '/kaggle/input/anlp-a2/dev.fr'
test_en_file = '/kaggle/input/anlp-a2/test.en'
test_fr_file = '/kaggle/input/anlp-a2/test.fr'

print(f"train.en has {count_lines(train_en_file)} lines.")
print(f"train.fr has {count_lines(train_fr_file)} lines.")
print(f"dev.en has {count_lines(dev_en_file)} lines.")
print(f"dev.fr has {count_lines(dev_fr_file)} lines.")
print(f"test.en has {count_lines(test_en_file)} lines.")
print(f"test.fr has {count_lines(test_fr_file)} lines.")

def get_sentence_lengths(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lengths = [len(line.strip().split()) for line in file]
    return lengths

en_lengths = get_sentence_lengths(train_en_file)
fr_lengths = get_sentence_lengths(train_fr_file)

max_en_length = max(en_lengths)
max_fr_length = max(fr_lengths)

percentile_95_en = int(np.percentile(en_lengths, 95))
percentile_95_fr = int(np.percentile(fr_lengths, 95))

print(f"English: Longest Sentence Length = {max_en_length}, 95th Percentile Length = {percentile_95_en}")
print(f"French: Longest Sentence Length = {max_fr_length}, 95th Percentile Length = {percentile_95_fr}")

max_seq_len = max(percentile_95_en, percentile_95_fr)
max_seq_len = (max_seq_len + 7) // 8 * 8  # Rounding up to the nearest multiple of 8
print(max_seq_len)

import matplotlib.pyplot as plt
import numpy as np

def plot_sentence_length_distribution(en_lengths, fr_lengths):
    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.hist(en_lengths, bins=30, color='blue', alpha=0.7)
    plt.title('English Sentence Length Distribution')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(fr_lengths, bins=30, color='green', alpha=0.7)
    plt.title('French Sentence Length Distribution')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_sentence_length_distribution(en_lengths, fr_lengths)

en_vocab_size = 8000
fr_vocab_size = 10000
vocab_sizes = {"en": en_vocab_size, "fr": fr_vocab_size}

spm.SentencePieceTrainer.train(
    f'--input={train_en_file} --model_prefix=spm_en --user_defined_symbols=<bos>,<eos>,<pad> --vocab_size={en_vocab_size}')
spm.SentencePieceTrainer.train(
    f'--input={train_fr_file} --model_prefix=spm_fr --user_defined_symbols=<bos>,<eos>,<pad> --vocab_size={fr_vocab_size}')

en_sp = spm.SentencePieceProcessor()
en_sp.load('spm_en.model')
fr_sp = spm.SentencePieceProcessor()
fr_sp.load('spm_fr.model')

tokenizers = {"en": en_sp.encode_as_ids, "fr": fr_sp.encode_as_ids}
detokenizers = {"en": en_sp.decode_ids, "fr": fr_sp.decode_ids}

UNK, BOS, EOS, PAD = 0, 1, 2, 3

def load_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as src_f, open(tgt_file, 'r', encoding='utf-8') as tgt_f:
        data = [(src.strip(), tgt.strip()) for src, tgt in zip(src_f, tgt_f)]
    return data

train_set = load_data(train_en_file, train_fr_file)
valid_set = load_data(dev_en_file, dev_fr_file)
test_set = load_data(test_en_file, test_fr_file)

max_seq_len = 48
def tokenize_dataset(dataset, src_lang, tgt_lang):
    'Tokenize and add BOS and EOS tokens to both source and target sequences'
    return [
        (
            torch.tensor([BOS] + tokenizers[src_lang](src_text)[:max_seq_len-2] + [EOS], dtype=torch.long),
            torch.tensor([BOS] + tokenizers[tgt_lang](tgt_text)[:max_seq_len-2] + [EOS], dtype=torch.long)
        )
        for src_text, tgt_text in dataset
    ]

train_tokenized = tokenize_dataset(train_set, "en", "fr")
valid_tokenized = tokenize_dataset(valid_set, "en", "fr")
test_tokenized = tokenize_dataset(test_set, "en", "fr")

class TranslationDataset(Dataset):
    'Custom Dataset for torch.utils.data.DataLoader()'
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pad_sequence(batch):
    'Collate function for padding sentences so all sentences in the batch have the same length'
    src_seqs = [src for src, trg in batch]
    trg_seqs = [trg for src, trg in batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True, padding_value=PAD)
    return src_padded, trg_padded

batch_size = 32

class Dataloaders:
    'Create train_loader, valid_loader, and test_loader for training and evaluation'
    def __init__(self):
        self.train_dataset = TranslationDataset(train_tokenized)
        self.valid_dataset = TranslationDataset(valid_tokenized)
        self.test_dataset = TranslationDataset(test_tokenized)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence)

dataloaders = Dataloaders()

for src_batch, tgt_batch in dataloaders.train_loader:
    print("Source batch:", src_batch)
    print("Target batch:", tgt_batch)
    break

print(src_batch.shape)
print(tgt_batch.shape)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0) #to get the batch size
        # Linear projections inorder to get the multi-head query, key and value multidim tensors.
        # Dimensions of x_query, x_key, x_value: nbatch * seq_len * d_embed.
        # Dimensions of query, key, value: nbatch * h * seq_len * d_k.
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key   = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        # Attention scores matrix dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        # Masking out padding tokens and future tokens wherever necessary
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        #dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        #dimensions: nbatch * h * seq_len * d_k
        x = torch.matmul(p_atten, value)
        #dimensions: nbtach * seq_len * d_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x)


class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))

# I haven't used fixed positional embeddings but let the model learn the positional embeddings as a learnable parameter.

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        # First self-attention layer
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        # then followed by position-wise fully connected feed-forward NN layer
        return self.residual2(x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.norm = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.decoder_vocab_size)

    def future_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), diagonal=1)!=0).to(DEVICE)
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, memory, src_mask, trg, trg_pad_mask):
        seq_len = trg.size(1)
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len))
        x = self.tok_embed(trg) + self.pos_embed[:, :trg.size(1), :]
        x = self.dropout(x)
        for layer in self.decoder_blocks:
            x = layer(memory, src_mask, x, trg_mask)
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atten1 = MultiHeadedAttention(config.h, config.d_embed)
        self.atten2 = MultiHeadedAttention(config.h, config.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout)
                                       for i in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        # keys,values->encoder and Query->decoder
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        return self.residuals[2](y, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, trg, trg_pad_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)

@dataclass
class ModelConfig:
    encoder_vocab_size: int
    decoder_vocab_size: int
    d_embed: int
    d_ff: int
    h: int
    N_encoder: int
    N_decoder: int
    max_seq_len: int
    dropout: float

def make_model(config):
    model = Transformer(Encoder(config), Decoder(config)).to(DEVICE)

    # initializing model parameters using xavier initialization
    for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

def make_batch_input(x, y):
        src = x.to(DEVICE)
        trg_in = y[:, :-1].to(DEVICE)
        trg_out = y[:, 1:].contiguous().view(-1).to(DEVICE)
        src_pad_mask = (src == PAD).view(src.size(0), 1, 1, src.size(-1))
        trg_pad_mask = (trg_in == PAD).view(trg_in.size(0), 1, 1, trg_in.size(-1))
        return src, trg_in, trg_out, src_pad_mask, trg_pad_mask

from numpy.lib.utils import lookfor
def train_epoch(model, dataloaders):
    model.train()
    grad_norm_clip = 1.0
    losses, acc, count = [], 0, 0
    num_batches = len(dataloaders.train_loader)
    pbar = tqdm(enumerate(dataloaders.train_loader), total=num_batches)
    for idx, (x, y)  in  pbar:
        optimizer.zero_grad()
        src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x,y)
        pred = model(src, src_pad_mask, trg_in, trg_pad_mask).to(DEVICE)
        pred = pred.view(-1, pred.size(-1))
        loss = loss_fn(pred, trg_out).to(DEVICE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if idx>0 and idx%50 == 0:
            pbar.set_description(f'train loss={loss.item():.3f}, lr={scheduler.get_last_lr()[0]:.5f}')
    return np.mean(losses)


def train(model, dataloaders, epochs):
    global early_stop_count
    best_valid_loss = float('inf')
    best_epoch = -1
    best_model_state = None
    train_size = len(dataloaders.train_loader)*batch_size
    train_losses = []
    valid_losses = []
    for ep in range(epochs):
        train_loss = train_epoch(model, dataloaders)
        valid_loss = validate(model, dataloaders.valid_loader)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'ep: {ep}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = ep
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pt')
            print(f'Best model saved with validation loss {best_valid_loss:.5f} at epoch {best_epoch}')
        else:
            if scheduler.last_epoch>2*warmup_steps:
                early_stop_count -= 1
                if early_stop_count<=0:
                    break
    model.load_state_dict(best_model_state)
    return train_losses, valid_losses,best_epoch


def validate(model, dataloder):
    'compute the validation loss'
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloder):
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x,y)
            pred = model(src, src_pad_mask, trg_in, trg_pad_mask).to(DEVICE)
            pred = pred.view(-1, pred.size(-1))
            losses.append(loss_fn(pred, trg_out).item())
    return np.mean(losses)

def translate(model, x):
#     while inference, translating source sentences into the target language in a auto regressive manner i.e. without looking at actual ground truth and relying on previous token generated only.
    with torch.no_grad():
        dB = x.size(0)
        y = torch.tensor([[BOS]*dB]).view(dB, 1).to(DEVICE)
        x_pad_mask = (x == PAD).view(x.size(0), 1, 1, x.size(-1)).to(DEVICE)
        memory = model.encoder(x, x_pad_mask)
        for i in range(max_seq_len):
            y_pad_mask = (y == PAD).view(y.size(0), 1, 1, y.size(-1)).to(DEVICE)
            logits = model.decoder(memory, x_pad_mask, y, y_pad_mask)
            last_output = logits.argmax(-1)[:, -1]
            last_output = last_output.view(dB, 1)
            y = torch.cat((y, last_output), 1).to(DEVICE)
    return y

def remove_pad(sent):
    if sent.count(EOS)>0:
        sent = sent[0:sent.index(EOS)+1]
    while sent and sent[-1] == PAD:
            sent = sent[:-1]
    return sent

def decode_sentence(detokenizer, sentence_ids):
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()
    sentence_ids = remove_pad(sentence_ids)
    return detokenizer(sentence_ids).replace("<bos>", "")\
           .replace("<eos>", "").strip().replace(" .", ".")

def evaluate(model, dataloader, num_batch=None):
    model.eval()
    refs, cans, bleus = [], [], []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x,y)
            translation = translate(model, src)
            trg_out = trg_out.view(x.size(0), -1)
            refs = refs + [decode_sentence(detokenizers[TRG], trg_out[i]) for i in range(len(src))]
            cans = cans + [decode_sentence(detokenizers[TRG], translation[i]) for i in range(len(src))]
            if num_batch and idx>=num_batch:
                break
        print(min([len(x) for x in refs]))
        bleus.append(sacrebleu.corpus_bleu(cans, [refs]).score)
        # print some examples
        for i in range(3):
            print(f'src:  {decode_sentence(detokenizers[SRC], src[i])}')
            print(f'trg:  {decode_sentence(detokenizers[TRG], trg_out[i])}')
            print(f'pred: {decode_sentence(detokenizers[TRG], translation[i])}')
        return np.mean(bleus)

SRC = "en"
TRG = "fr"

config = ModelConfig(encoder_vocab_size = vocab_sizes[SRC],
                     decoder_vocab_size=vocab_sizes[TRG],
                     d_embed=768,
                     d_ff=2028,
                     h=8,
                     N_encoder=3,
                     N_decoder=3,
                     max_seq_len=max_seq_len,
                     dropout=0.1
                     )

data_loaders = Dataloaders()
train_size = len(data_loaders.train_loader)*batch_size
model = make_model(config)
model_size = sum([p.numel() for p in model.parameters()])
print(f'model_size: {model_size}, train_set_size: {train_size}')
warmup_steps = 3*len(data_loaders.train_loader)
# lr increases during the warmup steps and then descreases accordingly
lr_fn = lambda step: config.d_embed**(-0.5) * min([(step+1)**(-0.5), (step+1)*warmup_steps**(-1.5)])
optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95), eps=1.0e-8)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
early_stop_count = 2
train_losses, valid_losses,best_epoch = train(model, data_loaders, epochs=15)
test_loss  = validate(model, data_loaders.test_loader)
best_train_loss = train_losses[best_epoch]
best_valid_loss = valid_losses[best_epoch]
best_test_loss = validate(model, data_loaders.test_loader)
print("train set examples:")
train_bleu = evaluate(model, data_loaders.train_loader, 20)
print("validation set examples:")
valid_bleu = evaluate(model, data_loaders.valid_loader)
print("test set examples:")
test_bleu  = evaluate(model, data_loaders.test_loader)
print(f'train_loss: {best_train_loss:.4f}, valid_loss: {best_valid_loss:.4f}, test_loss: {best_test_loss:.4f}')
print(f'test_bleu: {test_bleu:.4f}, valid_bleu: {valid_bleu:.4f} train_bleu: {train_bleu:.4f}')

def translate_this_sentence(text: str):
    'translate the source sentence in string formate into target language'
    input = torch.tensor([[BOS] + tokenizers[SRC](text) + [EOS]]).to(DEVICE)
    output = translate(model, input)
    return decode_sentence(detokenizers[TRG], output[0])

translate_this_sentence("What do you do for a living?")

def evaluate_and_save_bleu(model, dataloader, filename='testbleu.txt', num_batch=None):
    model.eval()
    refs, cans, bleus = [], [], []

    with open(filename, 'w') as f:
        with torch.no_grad():
            for idx, (x, y) in enumerate(dataloader):
                src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)

                translation = translate(model, src)

                trg_out = trg_out.view(x.size(0), -1)

                for i in range(len(src)):
                    ref_sentence = decode_sentence(detokenizers[TRG], trg_out[i])
                    gen_sentence = decode_sentence(detokenizers[TRG], translation[i])

                    bleu_score = sacrebleu.corpus_bleu([gen_sentence], [[ref_sentence]]).score

                    f.write(f"{gen_sentence} {bleu_score:.2f}\n")

                    refs.append(ref_sentence)
                    cans.append(gen_sentence)

                if num_batch and idx >= num_batch:
                    break

            overall_bleu_score = sacrebleu.corpus_bleu(cans, [refs]).score

            print(f"Overall BLEU score: {overall_bleu_score:.2f}")
            print(f"BLEU scores for individual sentences saved to {filename}")

            for i in range(min(3, len(src))):
                print(f'src:  {decode_sentence(detokenizers[SRC], src[i])}')
                print(f'trg:  {decode_sentence(detokenizers[TRG], trg_out[i])}')
                print(f'pred: {decode_sentence(detokenizers[TRG], translation[i])}')

            return overall_bleu_score

evaluate_and_save_bleu(model, data_loaders.test_loader, filename='testbleu.txt')

import matplotlib.pyplot as plt

hyperparams_list = [
    {'d_embed': 256, 'd_ff': 512, 'h': 4, 'N_encoder': 2, 'N_decoder': 2, 'dropout': 0.1},
    {'d_embed': 512, 'd_ff': 512, 'h': 8, 'N_encoder': 2, 'N_decoder': 2, 'dropout': 0.1},
    {'d_embed': 512, 'd_ff': 1024, 'h': 8, 'N_encoder': 3, 'N_decoder': 3, 'dropout': 0.2},
    {'d_embed': 768, 'd_ff': 2048, 'h': 8, 'N_encoder': 3, 'N_decoder': 3, 'dropout': 0.2},
    {'d_embed': 768, 'd_ff': 2048, 'h': 8, 'N_encoder': 4, 'N_decoder': 4, 'dropout': 0.1}
]

results = []

for i, params in enumerate(hyperparams_list):
    print(f"\n=== Hyperparameter Combination {i+1} ===")
    print(f"Params: {params}")

    config = ModelConfig(
        encoder_vocab_size=vocab_sizes[SRC],
        decoder_vocab_size=vocab_sizes[TRG],
        d_embed=params['d_embed'],
        d_ff=params['d_ff'],
        h=params['h'],
        N_encoder=params['N_encoder'],
        N_decoder=params['N_decoder'],
        max_seq_len=max_seq_len,
        dropout=params['dropout']
    )

    model = make_model(config)
    model_size = sum([p.numel() for p in model.parameters()])
    print(f'Model size: {model_size}')

    warmup_steps = 3 * len(data_loaders.train_loader)
    lr_fn = lambda step: config.d_embed**(-0.5) * min([(step + 1)**(-0.5), (step + 1) * warmup_steps**(-1.5)])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    print("Training model...")
    train_losses, valid_losses,best_epoch = train(model, data_loaders, epochs=15)
    test_loss  = validate(model, data_loaders.test_loader)
    best_train_loss = train_losses[best_epoch]
    best_valid_loss = valid_losses[best_epoch]
    best_test_loss = validate(model, data_loaders.test_loader)

    print("Evaluating BLEU score on train, validation, and test sets...")
    train_bleu = evaluate(model, data_loaders.train_loader, 20)
    valid_bleu = evaluate(model, data_loaders.valid_loader)
    test_bleu = evaluate(model, data_loaders.test_loader)

    print(f'train_loss: { best_train_loss:.4f}, valid_loss: {best_valid_loss:.4f}, test_loss: {best_test_loss:.4f}')
    print(f'test_bleu: {test_bleu:.4f}, valid_bleu: {valid_bleu:.4f}, train_bleu: {train_bleu:.4f}')

    results.append({
        'params': params,
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'test_loss': test_loss,
        'train_bleu': train_bleu,
        'valid_bleu': valid_bleu,
        'test_bleu': test_bleu
    })

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves - Hyperparameter Combination {i+1}")
    plt.legend()
    plt.show()

    print("=============================================")