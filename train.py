import os
import time
import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from predict import get_example_translation, score_translation
from utils import epoch_time, count_parameters, create_mask
from models.model import get_model_for_training
from predict import translate_with_transformer_by_batch


def train(model, iterator, optimizer, criterion, clip, args, epoch):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())
        args.writer.add_scalar('loss per batch/train', loss.cpu().data.numpy(), epoch * len(iterator) + i)
        if args.debug:
            print('debug mode on. exiting training...')
            break
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, args, epoch):
    model.eval()
    epoch_loss = 0
    history = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            args.writer.add_scalar('loss per batch/val', loss.cpu().data.numpy(), epoch * len(iterator) + i)

            if args.debug:
                print('debug mode on. exiting validation...')
                break

    return epoch_loss / len(iterator)


def train_transformer(model, iterator, optimizer, criterion, device, epoch, writer, debug, pad_idx):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        tgt_input = trg[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, pad_idx)

        output = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        output = output.reshape(-1, output.shape[-1])
        trg = trg[1:, :].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()

        writer.add_scalar('train loss per batch', loss.cpu().data.numpy(), epoch * len(iterator) + i)
        if debug:
            print('debug mode on. exiting training...')
            break
    return epoch_loss / len(iterator)


# def evaluate_transformer(model, iterator, criterion, device, epoch, writer, debug, pad_idx):
#     model.eval()
#     epoch_loss = 0
#
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#             src = batch.src
#             trg = batch.trg
#
#             tgt_input = trg[:-1, :]
#             src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, device, pad_idx)
#
#             output = model(src, tgt_input, src_mask, tgt_mask,
#                            src_padding_mask, tgt_padding_mask, src_padding_mask)
#
#             output = output.reshape(-1, output.shape[-1])
#             trg = trg[1:, :].reshape(-1)
#             loss = criterion(output, trg)
#             epoch_loss += loss.item()
#
#             writer.add_scalar('loss per step/val', loss.cpu().data.numpy(), epoch * len(iterator) + i)
#
#             if debug:
#                 print('debug mode on. exiting validation...')
#                 break
#     return epoch_loss / len(iterator)


def evaluate_transformer(model, iterator, criterion, device, epoch, writer, debug, args):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # print('src.shape', src.shape, 'trg.shape', trg.shape)
            output, probs = translate_with_transformer_by_batch(model, src, args.src, args.trg, device, trg.shape[0])
            output = probs.reshape(-1, probs.shape[-1]).to(device)
            trg = trg[1:, :].reshape(-1)
            # print('output.type', output.type(), 'trg.type', trg.type())
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            writer.add_scalar('val bleu per batch', loss.cpu().data.numpy(), epoch * len(iterator) + i)

            if debug:
                print('debug mode on. exiting validation...')
                break
    return epoch_loss / len(iterator)


def run_train(model, train_iterator, valid_iterator, criterion, optimizer, pad_idx, args):
    print('start training...')
    n_epochs = args.epochs
    clip = args.clip

    args.writer = SummaryWriter(
        os.path.join('runs', model.name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )

    best_valid_loss = float('inf')
    for epoch in range(n_epochs):

        start_time = time.time()

        if model.name == 'transformer':
            train_loss = train_transformer(model, train_iterator, optimizer, criterion, args.device, epoch,
                                           args.writer, args.debug, pad_idx)
            # valid_loss = evaluate_transformer(model, valid_iterator, criterion, args.device, epoch,
            #                                  args.writer, args.debug, args)
            args.model = model
            valid_loss, _, _ = score_translation(args, valid_iterator, pad_idx)  # actually, it's blue

        else:
            train_loss = train(model, train_iterator, optimizer, criterion, clip, args, epoch)
            valid_loss = evaluate(model, valid_iterator, criterion, args, epoch)
            args.model = model
            bleu, _, _ = score_translation(args, valid_iterator, pad_idx)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model.name + '.pt')

        args.writer.add_scalar('mean loss per epoch/train ', train_loss, global_step=epoch)
        if model.name == 'transformer':
            args.writer.add_scalar('bleu/val', valid_loss, global_step=epoch)
        else:
            args.writer.add_scalar('mean loss per epoch/val', valid_loss, global_step=epoch)
            args.writer.add_scalar('bleu/val', bleu, global_step=epoch)

        args.model = model
        example = get_example_translation(args)
        args.writer.add_text('translation example', example, global_step=epoch)
        args.writer.close()

        print()
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(example)


def print_translation(original_text, generated_text, args):
    print()
    print('Translation examples:')
    for i, (original_sentence, generated_sentence) in enumerate(zip(original_text, generated_text)):
        if i % 1000 == 0:
            print('original_sentence:', args.untokenize(original_sentence))
            print('generated_sentence:', args.untokenize(generated_sentence))
            print()


def main(args):
    print('start running training...')
    model = get_model_for_training(args)
    print('model:', model.name)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    pad_idx = args.trg.vocab.stoi['<pad>']
    if model.name == 'transformer':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta_min, args.beta_max),  eps=float(args.eps))
    else:
        optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    run_train(model, args.train_iterator, args.valid_iterator, criterion, optimizer, pad_idx, args)
    bleu_score, original_text, generated_text = score_translation(args, args.test_iterator, pad_idx)
    print(f'model: {model.name}, bleu score: {bleu_score}')
    print_translation(original_text, generated_text, args)
