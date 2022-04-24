import codecs
from functools import partial

import requests
import numpy as np
import os

from nltk import WordPunctTokenizer
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import sentencepiece as spm


URL = 'https://raw.githubusercontent.com/ermakovvova/nmt_experiments/dataset/data.txt'
DATASETS_PATH = 'dataset'
DATASET_FILENAME = 'data'
FILE_FORMAT = '.txt'
data_filepath = os.path.join(DATASETS_PATH, DATASET_FILENAME + FILE_FORMAT)
VOCAB_SIZE = 10000

src_data_filepath = os.path.join(DATASETS_PATH, DATASET_FILENAME + '_src' + FILE_FORMAT)
trg_data_filepath = os.path.join(DATASETS_PATH, DATASET_FILENAME + '_trg' + FILE_FORMAT)

src_data_codec_filepath = os.path.join(DATASETS_PATH, DATASET_FILENAME + '_src_codec' + FILE_FORMAT)
trg_data_codec_filepath = os.path.join(DATASETS_PATH, DATASET_FILENAME + '_trg_codec' + FILE_FORMAT)


def download_data():
    if not os.path.exists(data_filepath):
        os.makedirs(DATASETS_PATH, exist_ok=True)
        data = requests.get(URL)
        with open(data_filepath, 'wb') as fout:
            fout.write(data.content)


def split_src_trg_datasets():
    source, target = [], []
    with open(data_filepath, 'r') as fin:
        for line in fin:
            trg, src = line.strip().split('\t')
            source.append(src.lower())
            target.append(trg.lower())

    with open(src_data_filepath, 'w') as fout:
        for line in source:
            fout.write(line + '\n')
    with open(trg_data_filepath, 'w') as fout:
        for line in target:
            fout.write(line + '\n')


def _learn_bpe_for_src_trg():
    print('start learning bpe with subword-nmt lib...')
    with open(src_data_filepath, 'r') as fin, open(src_data_codec_filepath, 'w') as fout:
        learn_bpe(fin, fout, num_symbols=1000)
    with open(trg_data_filepath, 'r') as fin, open(trg_data_codec_filepath, 'w') as fout:
        learn_bpe(fin, fout, num_symbols=1000)


def learn_bpe_for_src_trg():
    print('start learning bpe with sentencepiece lib...')
    spm.SentencePieceTrainer.train(f'--input={src_data_filepath} --model_prefix=src --vocab_size={VOCAB_SIZE}')
    spm.SentencePieceTrainer.train(f'--input={trg_data_filepath} --model_prefix=trg --vocab_size={VOCAB_SIZE}')


def init_tokenizer(tokenizer, *args, **kwargs):
    return tokenizer(*args, **kwargs)


def init_tokenizers(args):
    if args.bpe:
        src_tokenizer = spm.SentencePieceProcessor()
        src_tokenizer.load('src.model')

        trg_tokenizer = spm.SentencePieceProcessor()
        trg_tokenizer.load('trg.model')
    else:
        src_tokenizer = init_tokenizer(WordPunctTokenizer)
        trg_tokenizer = init_tokenizer(WordPunctTokenizer)
    common_tokenizer = init_tokenizer(WordPunctTokenizer)

    def tokenize(tokenizer, x):
        if args.bpe:
            return tokenizer.encode_as_pieces(x.lower())
        return tokenizer.tokenize(x.lower())

    def untokenize(tokenizer, x):
        if args.bpe:
            return tokenizer.decode_pieces(x)
        return ' '.join(x)

    return partial(tokenize, src_tokenizer), partial(tokenize, trg_tokenizer),\
           partial(untokenize, src_tokenizer), partial(untokenize, trg_tokenizer), \
           lambda x: common_tokenizer.tokenize(x), lambda x: ' '.join(x)


def preprocess_data(args):
    src = Field(tokenize=args.tokenize_src,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    trg = Field(tokenize=args.tokenize_trg,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    dataset = TabularDataset(
        path=data_filepath,
        format='tsv',
        fields=[('trg', trg), ('src', src)]
    )

    train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
    src.build_vocab(train_data, min_freq=args.min_freq)
    trg.build_vocab(train_data, min_freq=args.min_freq)
    args.src_vocab_size, args.trg_vocab_size = len(src.vocab), len(trg.vocab)
    print('src vocabulary size', args.src_vocab_size)
    print('trg vocabulary size', args.trg_vocab_size)
    args.train_data, args.valid_data, args.test_data, args.src, args.trg = train_data, valid_data, test_data, src, trg

    example_idx = np.random.choice(np.arange(len(args.test_data)))
    src = vars(args.train_data.examples[example_idx])['src']
    trg = vars(args.train_data.examples[example_idx])['trg']
    print('src example:', src)
    print('trg example:', trg)


def _len_sort_key(x):
    return len(x.src)


def init_dataloaders(args):
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (args.train_data, args.valid_data, args.test_data),
        batch_size=args.batch,
        device=args.device,
        sort_key=_len_sort_key
    )
    args.train_iterator, args.valid_iterator, args.test_iterator = train_iterator, valid_iterator, test_iterator
