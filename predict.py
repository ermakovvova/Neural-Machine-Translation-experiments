import tqdm
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from utils import remove_tech_tokens, generate_square_subsequent_mask


def translate_sentence(model, sentence, src_field, trg_field,  device, tokenize_scr, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        # nlp = spacy.load('de')
        tokens = tokenize_scr(sentence)  # [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # src_len = torch.LongTensor([len(src_indexes)]).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for t in range(1, max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        # insert input token embedding, previous hidden state and all encoder hidden states
        # receive output tensor (predictions) and new hidden state
        output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]


def get_example_translation(args):
    example_idx = np.random.choice(np.arange(len(args.test_data)))

    src = vars(args.train_data.examples[example_idx])['src']
    trg = vars(args.train_data.examples[example_idx])['trg']

    src_string = f'src = {args.untokenize_src(src)}'
    trg_string = f'trg = {args.untokenize_trg(trg)}'

    if args.model.name == 'transformer':
        translation = translate_with_transformer(args.model, src, args.src, args.trg, args.device, args.tokenize_src)
    else:
        translation = translate_sentence(args.model, src, args.src, args.trg, args.device, args.tokenize_src)

    translation_string = f'translated = {args.untokenize_trg(translation)}'
    return '\n\n'.join([src_string, trg_string, translation_string])


def get_text(x, trg_vocab, args):
    text = [trg_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []

    if args.bpe:
        pass

    return text


# def generate_translation(src, trg, model, trg_vocab, args):
#     model.eval()
#
#     output = model(src, trg, 0)  # turn off teacher forcing
#     output = output.argmax(dim=-1).cpu().numpy()
#
#     original = get_text(list(trg[:, 0].cpu().numpy()), trg_vocab, args)
#     generated = get_text(list(output[1:, 0]), trg_vocab, args)
#
#     print('Original: {}'.format(' '.join(original)))
#     print('Generated: {}'.format(' '.join(generated)))
#     print()


def score_translation(args, iterator, pad_idx):
    original_text = []
    generated_text = []
    args.model.eval()
    with torch.no_grad():
        if args.model.name == 'transformer':
            for i, batch in tqdm.tqdm(enumerate(iterator)):
                src = batch.src
                trg = batch.trg

                output, probs = translate_with_transformer_by_batch(args.model, src, args.src, args.trg, args.device)
                generated_text.extend([
                    args.tokenize(args.untokenize_trg(get_text(x, args.trg.vocab, args)))
                    for x in output[:, 1:].detach().cpu().numpy()
                ])
                original_text.extend([
                    args.tokenize(args.untokenize_trg(get_text(x, args.trg.vocab, args))) for x in trg.cpu().numpy().T
                ])
                if args.debug:
                    break
        else:
            for i, batch in tqdm.tqdm(enumerate(args.test_iterator)):
                src = batch.src
                trg = batch.trg
                output = args.model(src, trg, 0)  # turn off teacher forcing

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]
                output = output.argmax(dim=-1)
                generated_text.extend([get_text(x, args.trg.vocab, args) for x in output[1:].detach().cpu().numpy().T])

                original_text.extend([get_text(x, args.trg.vocab, args) for x in trg.cpu().numpy().T])

                if args.debug:
                    break

        return corpus_bleu([[text] for text in original_text], generated_text) * 100, original_text, generated_text


def _generate(sentence, args, max_len=50, start_tokens_num=3):
    if args.pretrain_encoder:
        field = args.src
    elif args.pretrain_decoder:
        field = args.trg
    else:
        raise Exception("Should use '--pretrain-encoder' or '--pretrain-decoder' args")
    model = args.model
    model.eval()
    device = args.device

    if isinstance(sentence, str):
        # nlp = spacy.load('de')
        tokens = args.tokenize_src(sentence)  # [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [field.init_token] + tokens + [field.eos_token]
    indexes = [field.vocab.stoi[token] for token in tokens[:start_tokens_num]]
    # tensor = torch.LongTensor(indexes).unsqueeze(1).to(device)
    # text_len = torch.LongTensor([len(indexes)]).to(device)

    with torch.no_grad():
        hidden = model.init_hidden(args, batch_size=1)
        cell = hidden

        for t in range(0, start_tokens_num):
            tensor = torch.LongTensor([indexes[t]]).to(device)
            if args.pretrain_encoder:
                tensor.unsqueeze_(0)
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            _, hidden, cell = model(tensor, hidden, cell)

        for t in range(start_tokens_num, max_len):
            tensor = torch.LongTensor([indexes[-1]]).to(device)
            if args.pretrain_encoder:
                tensor.unsqueeze_(0)

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden, cell = model(tensor, hidden, cell)

            pred_token = output.argmax(-1).item()
            indexes.append(pred_token)
            if pred_token == field.vocab.stoi[field.eos_token]:
                break

    src_tokens = remove_tech_tokens(
        [field.vocab.itos[i] for i in indexes],
        tokens_to_remove=['<sos>']
    )
    return ' '.join(src_tokens[:])


def generate_text(args):
    example_idx = np.random.choice(np.arange(len(args.test_data)))

    if args.pretrain_encoder:
        example_text = vars(args.test_data.examples[example_idx])['src']
    elif args.pretrain_decoder:
        example_text = vars(args.test_data.examples[example_idx])['trg']
    else:
        raise Exception("Should use '--pretrain-encoder' or '--pretrain-decoder' args")

    generated_text = _generate(example_text, args)
    return ' '.join(example_text), generated_text


def greedy_decode(model, src, src_mask, max_len, start_symbol, device, eos_idx):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        # memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys


def translate_with_transformer(model, sentence, src_field, trg_field, device, tokenize_src):
    sos_idx = src_field.vocab.stoi[src_field.init_token]
    eos_idx = trg_field.vocab.stoi[src_field.eos_token]
    model.eval()
    # tokens = [sos_idx] + [src_vocab.stoi[tok] for tok in src_tokenizer.tokenize(sentence)] + [eos_idx]

    if isinstance(sentence, str):
        # nlp = spacy.load('de')
        tokens = tokenize_src(sentence)  # [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    tokens = [src_field.vocab.stoi[token] for token in tokens]

    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=sos_idx,
                               device=device, eos_idx=eos_idx).flatten()
    return [trg_field.vocab.itos[tok] for tok in tgt_tokens]


def translate_with_transformer_by_batch(model, src, src_field, trg_field, device, max_len=None):
    sos_idx = src_field.vocab.stoi[src_field.init_token]
    eos_idx = trg_field.vocab.stoi[src_field.eos_token]
    model.eval()

    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    max_len = max_len or num_tokens + 5
    tgt_tokens, probs = greedy_decode_by_batch(model, src, src_mask, len(trg_field.vocab),
                                               max_len=max_len, start_symbol=sos_idx,
                                               device=device, eos_idx=eos_idx)

    tgt_tokens = tgt_tokens.transpose(0, 1)

    return tgt_tokens, probs


def greedy_decode_by_batch(model, src, src_mask, vocab_size, max_len, start_symbol, device, eos_idx):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, src.shape[1]).fill_(start_symbol).type(torch.long).to(device)
    probs = torch.zeros(max_len - 1, src.shape[1], vocab_size)
    for i in range(max_len - 1):
        memory = memory.to(device)
        # memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)

        # print(probs.shape, prob.shape)
        probs[i, :] = prob
    return ys, probs
