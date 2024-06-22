# Neural machine translation experiments

This repo presents series of experiments aimed to compare several Seq2Seq neural architectures and techniques for neural machine translation task.

### Criteria for comparison:
- BLUE score
- GPU training time

### Data:

### Experiments include
 - Compare LSTM and Transformer architectures for machine translation task
 - Investigate influence of pretraining of encoder and decoder models on language modeling task
 - Compare word tokenization with BPE.


Training output contains examples of generated texts, values of loss and blue score.
All visualizations are done in tensorboard.
Please, look at [nmt_experiments.ipynb](https://github.com/ermakovvova/nmt_experiments/blob/master/nmt_experiments.ipynb)


## Project Structure

```
├── config
│   ├── train.yaml
├── data.py
├── dataset
│   ├── data.txt
├── notebooks
│   └── Inference.ipynb
├── models
│   ├── __init__.py
│   ├── model.py
│   ├── rnn.py
│   ├── transformer.py
├── parser.py
├── predict.py
├── pretrain.py
├── README.md
├── run.py
├── requirements.txt
├── train.py
└── utils.py
```

- **utils/dataloader.py** - data loader for WikiText-2 and WikiText103 datasets
- **utils/model.py** - model architectures
- **utils/trainer.py** - class for model training and evaluation

- **train.py** - script for training
- **config.yaml** - file with training parameters
- **weights/** - folder where expriments artifacts are stored
- **notebooks/Inference.ipynb** - demo of how embeddings are used

## Usage

Explain argparse under the hood

```
python run.py train --enc-type rnn --enc-name PretrainEncoder --epochs 20 pretrain --pretrain-encoder
```

Before running the command, change the training parameters in the config.yaml, most important:

- model_name ("skipgram", "cbow")
- dataset ("WikiText2", "WikiText103")
- model_dir (directory to store experiment artifacts, should start with "weights/")

## Ablation study
- Training seq2seq model based on standard LSTM-encoder and LSTM-decoder for russian-english translation.
- Pretraining LSTM-encoder on language model task on russian language.
- Pretraining LSTM-decoder on language model task on english language.
- Training pretrained LSTM-encoder and standard LSTM-decoder for translation task.
- Training LSTM-encoder and pretrained LSTM-decoder for translation task.
- Training pretrained LSTM-encoder and pretrained LSTM-decoder for translation task.
- Training Transformer model for translation task.
- Training Transformer model on BPE tokens.