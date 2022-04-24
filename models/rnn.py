import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.input_dim = config.src_vocab_size
        self.emb_dim = config.enc_emb_dim
        self.hid_dim = config.hid_dim
        self.n_layers = config.n_layers
        self.dropout_proba = config.enc_dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=self.input_dim,
            embedding_dim=self.emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_proba
        )

        self.dropout = nn.Dropout(p=self.dropout_proba)
        
    def forward(self, src):
        
        # src = [src sent len, batch size]
        
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        
        output, (hidden, cell) = self.rnn(embedded)
        # embedded = [src sent len, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # outputs are always from the top hidden layer
        
        return hidden, cell


class PretrainEncoder(nn.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.input_dim = config.src_vocab_size
        self.emb_dim = config.enc_emb_dim
        self.hid_dim = config.hid_dim
        self.n_layers = config.n_layers
        self.dropout_proba = config.enc_dropout

        self.embedding = nn.Embedding(
            num_embeddings=self.input_dim,
            embedding_dim=self.emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_proba
        )

        self.dropout = nn.Dropout(p=self.dropout_proba)

        self.fc = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded = [sent len, batch size, emb dim]

        # pass embeddings into LSTM
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(output))
        # predictions = [sent len, batch size, output dim]
        return predictions, hidden, cell

    @staticmethod
    def init_hidden(args, batch_size=None):
        batch_size = batch_size or args.batch_size
        return torch.zeros(args.n_layers, batch_size, args.hid_dim, device=args.device)
    

class Decoder(nn.Module):
    def __init__(self, name, args):
        super().__init__()
        self.name = name
        self.emb_dim = args.dec_emb_dim
        self.hid_dim = args.hid_dim
        self.output_dim = args.trg_vocab_size
        self.n_layers = args.n_layers
        self.dropout_proba = args.dec_dropout
        
        self.embedding = nn.Embedding(
            num_embeddings=self.output_dim,
            embedding_dim=self.emb_dim
        )
        
        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.n_layers,
            dropout=self.dropout_proba
        )
        
        self.fc = nn.Linear(
            in_features=self.hid_dim,
            out_features=self.output_dim
        )

        self.dropout = nn.Dropout(p=self.dropout_proba)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        # input = [1, batch size]
        
        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))
        
        # embedded = [1, batch size, emb dim]
        
        # Compute the RNN output values of the encoder RNN. 
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

    @staticmethod
    def init_hidden(args, batch_size=None):
        batch_size = batch_size or args.batch_size
        return torch.zeros(args.n_layers, batch_size, args.hid_dim, device=args.device)


class Seq2Seq(nn.Module):
    def __init__(self, name, encoder, decoder, device):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
