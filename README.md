Lab assignment 2: NMT [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ermakovvova/MADE_NLP/blob/main/Lab02_nmt_new/Lab2_NMT.ipynb)

```
rm nmt.zip &&  zip -r nmt.zip ./ -x '.idea/*' '__pycache__.py' 'runs/*' '*.pt' 'dataset/*' '.zip'

python run.py pretrain --model-type rnn -n PretrainEncoder --pretrain-encoder --debug

python run.py pretrain --model-type rnn -n Decoder --pretrain-decoder --debug

python run.py train --enc-type rnn --enc-name Encoder --dec-type rnn --dec-name Decoder --seq2seq-type rnn --seq2seq-name LSTM --debug --enc-pretrained-filepath rnn_PretrainEncoder.pt

python run.py train --seq2seq-type transformer  --epochs 20 --debug
```


