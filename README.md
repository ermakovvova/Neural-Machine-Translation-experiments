# Neural machine translation experiments

This repo presents series of experiments aimed to compare several Seq2Seq neural architectures and techniques for neural machine translation task.

### Criteria for comparison:
- BLUE score
- GPU training time per epoch

### Data:
English-Russian pairs of hotel descriptions.

### Experiments include
 - Comparison of LSTM and Transformer architectures for machine translation task
 - Analysos of influence of pretraining encoder and/or decoder models on language modeling task
 - Comparison of WordPunctTokenizer from nltk lib with BPE.


## Project Structure

```
├── config
│   ├── train.yaml
├── data.py
├── dataset
│   ├── data.txt
├── nmt_experiments.ipynb
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
├── tests
│   ├── __init__.py
├── train.py
└── utils.py
```


## Usage

Command line tool allows to run
- train LSTM based Seq2Seq model:
    ```
    python run.py train --seq2seq-name Seq2Seq --epochs 25`
    ```
    or Transformer model:
    ```
    python run.py train --seq2seq-type transformer--epochs 30
    ```
    or with BPE:
    ```
    python run.py train --seq2seq-type transformer --bpe --epochs 30
    ```

- pretrain encoder or decore:
    ```
    python run.py train --enc-type rnn --enc-name PretrainEncoder --epochs 20 pretrain --pretrain-encoder
    ```
- train from pretrained model:
    ```
    python run.py train --enc-pretrained-filepath rnn_PretrainEncoder.pt --epochs 25
    ```


Training parameters are set in the `config/train.yaml` and could be replaced by command line arguments


## Ablation study
Experiment | BLUE score | GPU training time per epoch 
--- |------------| --- 
LSTM-encoder and LSTM-decoder | 18.5       | 1m 14s
LSTM-encoder and LSTM-decoder; LSTM-encoder pretrained on LM task on russian language | 20.4       | 1m 19s 
LSTM-encoder and LSTM-decoder; LSTM-decoder pretreined on LM task on english language | 17.8       |  1m 19s
LSTM-encoder and LSTM-decoder; both LSTM-encoder and LSTM-decoder pretreined on LM task| 17.2       |  1m 19s
Transformer model with NLTK tokenizer| 28.7       | 50s  
Transformer model with BPE tokenizer| 26.6       | 1m 3s 


Training logs contain examples of generated texts, loss and blue score.
All visualizations are done in Tensorboard.



Translation examples with Transformer model with 30 epochs and BPE:

Original sentence | Generated sentence 
--- |------------
the osmose spa is the perfect place for a relaxing experience . it features a hammam . | guests can relax in the turkish bath or enjoy a drink at the spa centre .
a grocery shop is 50 metres from moncherie studio rooms , while the nearest green market is 500 metres away . | the nearest grocery store is 50 metres away .
easily accessible from all major routes , lodge impresses with personalized service . | all the rooms of the bay ofs bay is a bicycle serviced and the area with all the bay ofs .
rowing boats , paddle boards and snorkeling equipments are available . | it offers home - cooked meals , as well as the upon request .
ar tavern backpackers hostel is a 10 - minute walk from finsbury park station and ar f . c ’ s emirates stadium . | naly park is a 10 - minute walk from metro station , where guests can enjoy the stadium and the est health club .

