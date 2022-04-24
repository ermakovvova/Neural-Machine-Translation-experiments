import models
from models.transformer import Seq2SeqTransformer
from utils import init_weights


def get_model_for_training(args):
    if args.seq2seq_type == 'transformer':
        model = Seq2SeqTransformer(args.seq2seq_type, args.num_encoder_layers, args.num_decoder_layers,
                                   args.emb_size, args.src_vocab_size, args.trg_vocab_size,
                                   args.ffn_hid_dim).to(args.device)
    else:
        enc_type = getattr(models, args.enc_type)
        enc = getattr(enc_type, args.enc_name)
        enc = enc(args.enc_name, args)
        init_weights(enc, args.enc_pretrained_filepath)
        # enc.apply(init_weights, args.enc_pretrained_filepath)

        dec_type = getattr(models, args.dec_type)
        dec = getattr(dec_type, args.dec_name)
        dec = dec(args.dec_name, args)
        init_weights(dec, args.dec_pretrained_filepath)

        model_type = getattr(models, args.seq2seq_type)
        model = getattr(model_type, args.seq2seq_name)

        args.model_name = args.seq2seq_name
        model = model(args.model_name, enc, dec, args.device).to(args.device)

    return model