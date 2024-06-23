import random

import numpy as np
import torch

from data import download_data, preprocess_data, init_dataloaders, split_src_trg_datasets, learn_bpe_for_src_trg, \
    init_tokenizers
from parser import get_parser, setup_train_parser, setup_pretrain_parser, setup_translate_parser

PROGRAM_NAME = 'nmt'
SEED = 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def setup_parser(subparsers):
    train_subparsers = setup_train_parser(subparsers)
    setup_pretrain_parser(train_subparsers)
    setup_translate_parser(subparsers)


def device(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser, subparsers = get_parser(PROGRAM_NAME)
    setup_parser(subparsers)
    args = parser.parse_args()
    download_data()
    if args.bpe:
        split_src_trg_datasets()
        learn_bpe_for_src_trg()
    device(args)
    args.tokenize_src, args.tokenize_trg,\
        args.untokenize_src, args.untokenize_trg,\
        args.tokenize, args.untokenize = init_tokenizers(args)
    preprocess_data(args)
    init_dataloaders(args)
    args.callback(args)


if __name__ == '__main__':
    main()
