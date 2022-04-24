import configargparse
import yaml

from train import main
from pretrain import pretrain_model


def get_parser(name, ):
    parser = configargparse.ArgumentParser(
        name,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    subparsers = parser.add_subparsers(
        help='choose command'
    )
    return parser, subparsers


def setup_pretrain_parser(subparsers):
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help='pretrain encoder/decoder'
    )
    pretrain_parser.add_argument('--pretrain-encoder', action='store_true')
    pretrain_parser.add_argument('--pretrain-decoder', action='store_true')
    pretrain_parser.set_defaults(
        callback=pretrain_model
    )


def setup_train_parser(subparsers):
    train_parser = subparsers.add_parser(
        "train",
        help='train nmt model',
        default_config_files=['config/*.yaml']
    )
    train_parser.add_argument('--epochs', type=yaml.safe_load)
    train_parser.add_argument('--batch', type=yaml.safe_load)
    train_parser.add_argument('--bpe',  action='store_true')

    train_parser.add_argument('--enc-type', help='for example: rnn, cnn, transformer')
    train_parser.add_argument('--enc-name', help='model name')
    train_parser.add_argument('--dec-type', help='for example: rnn, cnn, transformer')
    train_parser.add_argument('--dec-name', help='model name')
    train_parser.add_argument('--seq2seq-type', type=yaml.safe_load, help='for example: rnn, cnn, transformer')

    train_parser.add_argument('--clip', type=yaml.safe_load)

    train_parser.add_argument('--lr', type=yaml.safe_load)
    # train_parser.add_argument('--betas', type=yaml.safe_load, nargs=2)
    train_parser.add_argument('--beta-min', type=yaml.safe_load)
    train_parser.add_argument('--beta-max', type=yaml.safe_load)
    train_parser.add_argument('--eps', type=yaml.safe_load)

    train_parser.add_argument('--seq2seq-name', '-n')
    train_parser.add_argument('--debug', action='store_true')
    train_parser.add_argument('--enc-pretrained-filepath')
    train_parser.add_argument('--dec-pretrained-filepath')
    train_parser.add_argument('--teacher-forcing-ratio', type=float)

    train_parser.add_argument('--enc-emb-dim', type=yaml.safe_load)
    train_parser.add_argument('--dec-emb-dim', type=yaml.safe_load)
    train_parser.add_argument('--hid-dim', type=yaml.safe_load)
    train_parser.add_argument('--n-layers', type=yaml.safe_load)
    train_parser.add_argument('--enc-dropout', type=yaml.safe_load)
    train_parser.add_argument('--dec-dropout', type=yaml.safe_load)
    train_parser.add_argument('--min-freq', type=yaml.safe_load)

    # transformer specific args:
    train_parser.add_argument('--nhead', type=yaml.safe_load)
    train_parser.add_argument('--ffn-hid-dim', type=yaml.safe_load)
    train_parser.add_argument('--num-encoder-layers', type=yaml.safe_load)
    train_parser.add_argument('--num-decoder-layers', type=yaml.safe_load)
    train_parser.add_argument('--emb_size', type=yaml.safe_load)

    train_parser.set_defaults(
        callback=main
    )

    subparsers = train_parser.add_subparsers(
        help='choose pretraining'
    )
    return subparsers


def setup_translate_parser(subparsers):
    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument('--model-name')
