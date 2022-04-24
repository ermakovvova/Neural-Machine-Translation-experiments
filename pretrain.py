import time
import math
import random
import os
import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from predict import generate_text
from utils import epoch_time, init_weights, count_parameters
import models


def pretrain(model, iterator, optimizer, criterion, clip, args, epoch):
    model.train()

    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        if args.pretrain_encoder:
            text = batch.src
        elif args.pretrain_decoder:
            text = batch.trg
        else:
            raise Exception('Unknown model for pretraining')

        optimizer.zero_grad()

        hidden = model.init_hidden(args, text.shape[1])
        cell = model.init_hidden(args, text.shape[1])

        if args.pretrain_decoder:
            batch_size = text.shape[1]
            max_len = text.shape[0]
            trg_vocab_size = model.output_dim

            # tensor to store decoder outputs
            predictions = torch.zeros(max_len, batch_size, trg_vocab_size).to(args.device)

            # first input to the decoder is the <sos> tokens
            input = text[0, :]

            for t in range(1, max_len):

                output, hidden, cell = model(input, hidden, cell)
                predictions[t] = output
                teacher_force = random.random() < args.teacher_forcing_ratio
                top1 = output.max(1)[1]
                input = (text[t] if teacher_force else top1)


        elif args.pretrain_encoder:
            predictions, hidden, cell = model(text, hidden, cell)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = predictions[:-1].view(-1, predictions.shape[-1])
        text = text[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, text)
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
            if args.pretrain_encoder:
                text = batch.src
            elif args.pretrain_decoder:
                text = batch.trg
            else:
                raise Exception('Unknown model for pretraining')

            hidden = model.init_hidden(args, text.shape[1])
            cell = model.init_hidden(args, text.shape[1])

            if args.pretrain_decoder:
                batch_size = text.shape[1]
                max_len = text.shape[0]
                trg_vocab_size = model.output_dim

                # tensor to store decoder outputs
                predictions = torch.zeros(max_len, batch_size, trg_vocab_size).to(args.device)

                # first input to the decoder is the <sos> tokens
                input = text[0, :]

                for t in range(1, max_len):

                    output, hidden, cell = model(input, hidden, cell)
                    predictions[t] = output
                    teacher_force = random.random() < args.teacher_forcing_ratio
                    top1 = output.max(1)[1]
                    input = (text[t] if teacher_force else top1)

            elif args.pretrain_encoder:
                predictions, hidden, cell = model(text, hidden, cell)

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = predictions[:-1].view(-1, predictions.shape[-1])
            text = text[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = criterion(output, text)
            epoch_loss += loss.item()

            args.writer.add_scalar('loss per batch/val', loss.cpu().data.numpy(), epoch * len(iterator) + i)

            if args.debug:
                print('debug mode on. exiting validation...')
                break

    return epoch_loss / len(iterator)


def run_pretrain(model, train_iterator, valid_iterator, criterion, optimizer, args):
    print('start training...')
    n_epochs = args.epochs
    clip = args.clip

    args.writer = SummaryWriter(
        os.path.join('runs', args.model_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )

    best_valid_loss = float('inf')
    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = pretrain(model, train_iterator, optimizer, criterion, clip, args, epoch)
        valid_loss = evaluate(model, valid_iterator, criterion, args, epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_name + '.pt')

        args.writer.add_scalar('mean loss per epoch/train ', train_loss, global_step=epoch)
        args.writer.add_scalar('mean loss per epoch/val', valid_loss, global_step=epoch)

        val_example_data = next(iter(valid_iterator))
        to_print = []

        args.model = model
        real_text, generated_text = generate_text(args)
        args.writer.add_text('generated example text', generated_text, global_step=epoch)
        args.writer.close()

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print('real_text:', real_text)
        print('generated_text:', generated_text)


def get_model_for_pretraining(args):
    if args.pretrain_encoder:
        model_type_name, model_name = args.enc_type, args.enc_name
    elif args.pretrain_decoder:
        model_type_name, model_name = args.dec_type, args.dec_name
    else:
        raise Exception("Should specify args '--pretrain-encoder' or '--pretrain-decoder'")

    model_type = getattr(models, model_type_name)
    model = getattr(model_type, model_name)
    args.model_name = model_type_name + '_' + model_name
    model = model(model_name, args).to(args.device)
    return model


def pretrain_model(args):
    print('start running pretraining...')
    model = get_model_for_pretraining(args)
    print('model:', model.name)
    init_weights(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    if args.pretrain_encoder:
        pad_idx = args.src.vocab.stoi['<pad>']
    if args.pretrain_decoder:
        pad_idx = args.trg.vocab.stoi['<pad>']
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    run_pretrain(model, args.train_iterator, args.valid_iterator, criterion, optimizer, args)
    # print('bleu score:', score_translation(args))
