# UER-py
[![Build Status](https://travis-ci.org/dbiir/UER-py.svg?branch=master)](https://travis-ci.org/dbiir/UER-py)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)

<img src="uer-logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks and has led to remarkable improvements. UER-py is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of different pre-training models (e.g. BERT), and provides interfaces for users to further extend upon. UER-py also incorporates many mechanisms for better performance and efficiency. It has been tested on several Chinese datasets and should match or even outperform Google's TF implementation.

**Update: Now ELMO (bilstm encoder + bilm target) is supported by UER. ELMO model pre-trained on mixed large corpus will be released soon. Word-based BERT is now available. Context-dependent word embedding (trained by BERT) is in particular suitable for polysemous words.** 

<br>

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Datasets](#datasets)
  * [Instructions](#instructions)
  * [Scripts](#scripts)
  * [Experiments](#experiments)
  * [Chinese_model_zoo](#chinese_model_zoo)
  
  
<br/>

## Features
UER-py has the following features:
- __Reproducibility.__ UER-py is able to reproduce the results of existing pre-training models (such as [Google BERT](https://github.com/google-research/bert)).
- __Multi-GPU.__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity.__ UER-py is divided into multiple components: subencoder, encoder, target, and fine-tuning. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.
- __Efficiency.__ UER-py refines its pre-processing, pre-training, and fine-tuning stages, which largely improves speed and needs less memory. 
- __SOTA results.__ Our works further improve the results upon Google BERT, providing new baselines for a range of datasets.
- __Chinese model zoo.__ We are pre-training models with different corpora, encoders, and targets. Selecting proper pre-training models is beneficial to the performance of downstream tasks.


<br/>

## Requirements
Python3.6
torch>=1.0
argparse


<br/>

## Quickstart
We use BERT model and [Douban book review classification dataset](https://embedding.github.io/evaluation/) to demonstrate how to use UER-py. We firstly pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

The format of the corpus for BERT is as follows：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained by book review dataset. We remove labels and split a review into two parts from the middle (See *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows (label and instance are separated by \t):
```
label    text_a
1        instance1
0        instance2
1        instance3
```

We use Google's Chinese vocabulary file, which contains 21128 Chinese characters. The format of the vocabulary is as follows:
```
word-1
word-2
...
word-n
```

First of all, we preprocess the book review corpus. We need to specify the model's target in pre-processing stage (--target):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Pre-processing is time-consuming. Multi-process can largely accelerate the pre-processing speed (--processes_num). The raw text is converted to dataset.pt, which is the input of pretrain.py. Then we download [Google's pre-trained Chinese model](https://share.weiyun.com/5s9AsfQ), and put it into *models* folder. We load Google's pre-trained model and train it on book review corpus. We should better explicitly specify model's encoder (--encoder) and target (--target). Suppose we have a machine with 8 GPUs.:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 5000 --encoder bert --target bert

mv models/book_review_model.bin-20000 models/book_review_model.bin
```
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step. We could remove the suffix for ease of use.
Finally, we do classification. We can use *google_model.bin*:
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 32 --encoder bert
```
or use our [*book_review_model.bin*](https://share.weiyun.com/52BEFs2), which is the output of pretrain.py：
```
python3 classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 32 --encoder bert
``` 
It turns out that the result of Google's model is 87.5; The result of *book_review_model.bin* is 88.1. It is also noticable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

BERT consists of next sentence prediction (NSP) target. However, NSP target is not suitable for sentence-level reviews since we have to split a review into two parts. UER-py facilitates the use of different targets. Using masked language modeling (MLM) as target could be a properer choice for pre-training of reviews:

```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin \
                    --output_model_path models/book_review_mlm_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 5000 --encoder bert --target mlm

mv models/book_review_mlm_model.bin-20000 models/book_review_mlm_model.bin

python3 classifier.py --pretrained_model_path models/book_review_mlm_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 32 --encoder bert
```
It turns out that the result of [*book_review_mlm_model.bin*](https://share.weiyun.com/5ScDjUO) is 88.3.

We could search proper pre-trained models in [Chinese model zoo](#chinese_model_zoo) for further improvements. For example, we could download [a model pre-trained on Amazon corpus (over 4 million reviews) with BERT encoder and classification (CLS) target](https://share.weiyun.com/5XuxtFA). It achieves 88.5 accuracy on book review dataset.

BERT is really slow. It could be great if we can speed up the model and still achieve competitive performance. We select a 2-layers LSTM encoder to substitute 12-layers Transformer encoder. We could download [a model pre-trained with LSTM encoder and language modeling (LM) + classification (CLS) targets](https://share.weiyun.com/5B671Ik):
```
python3 classifier.py --pretrained_model_path models/lstm_reviews_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3  --batch_size 64 --encoder lstm --pooling mean --config_path models/rnn_config.json --learning_rate 1e-3
```
We can achieve 86.5 accuracy on testset, which is also a competitive result. Using LSTM without pre-training can only achieve 80.2 accuracy. In practice, above model is around 10 times faster than BERT. One can see Chinese model zoo section for more detailed information about above pre-trained LSTM model.

Besides classification, UER-py also provides scripts for other downstream tasks. We could use tagger.py for sequence labeling:
```
python3 tagger.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                  --train_path datasets/msra/train.tsv --dev_path datasets/msra/dev.tsv --test_path datasets/msra/test.tsv \
                  --epochs_num 5 --batch_size 16 --encoder bert
```
We could download [a model pre-trained on RenMinRiBao (as known as People's Daily, a news corpus)](https://share.weiyun.com/5JWVjSE) and finetune on it: 
```
python3 tagger.py --pretrained_model_path models/rmrb_model.bin --vocab_path models/google_vocab.txt \
                  --train_path datasets/msra/train.tsv --dev_path datasets/msra/dev.tsv --test_path datasets/msra/test.tsv \
                  --epochs_num 5 --batch_size 16 --encoder bert
```
It turns out that the result of Google's model is 92.6; The result of *rmrb_model.bin* is 94.4.

<br/>

## Datasets
This project includes a range of Chinese datasets. Small-scale datasets can be downloaded at [datasets_zh.zip](https://share.weiyun.com/5LQcJJP). datasets_zh.zip contains 7 datasets: XNLI, LCQMC, MSRA-NER, ChnSentiCorp, and nlpcc-dbqa are obtained from [Baidu ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE); Book review (from BNU) and Shopping are two sentiment analysis datasets. Large-scale datasets can be found in [glyph's github project](https://github.com/zhangxiangxiao/glyph).

<br/>

## Instructions
### UER-py's framework
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/: contains encoders such as RNN, CNN, Attention, CNN-RNN, BERT
    |    |--targets/: contains targets such as language modeling, masked language modeling, sentence prediction
    |    |--subencoders/: contains subencoders such as RNN, CNN, and different pooling strategies
    |    |--layers/: contains frequently-used NN layers, such as embedding layer, normalization layer
    |    |--models/: contains model.py, which combines subencoder, embedding, encoder, and target modules
    |    |--utils/: contains frequently-used utilities
    |    |--model_builder.py 
    |    |--model_saver.py
    |    |--trainer.py
    |
    |--corpora/: contains corpora for pre-training
    |--datasets/: contains downstream tasks
    |--models/: contains pre-trained models, vocabularies, and config files
    |--scripts/: contains some useful scripts for pre-training models
    |
    |--preprocess.py
    |--pretrain.py
    |--classifier.py
    |--cloze.py
    |--tagger.py
    |--feature_extractor.py
    |--README.md
```

The code is well-organized. Users can use and extend upon it with little efforts.

### Preprocess the data
```
usage: preprocess.py [-h] --corpus_path CORPUS_PATH --vocab_path VOCAB_PATH
                     [--dataset_path DATASET_PATH]
                     [--tokenizer {bert,char,space}]
                     [--processes_num PROCESSES_NUM]
                     [--target {bert,lm,cls,mlm,nsp,s2s}]
                     [--docs_buffer_size DOCS_BUFFER_SIZE]
                     [--instances_buffer_size INSTANCES_BUFFER_SIZE]
                     [--seq_length SEQ_LENGTH] [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--seed SEED]
```
*--docs_buffer_size* and *--instances_buffer_size* could be used to control memory consumption in pre-processing and pre-training stages. *--preprocesses_num n* denotes that n processes are used for pre-processing. The example of pre-processing on a single machine is as follows：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bert
```
We need to specify the model's target in pre-processing stage since different targets require different data formats. Currently, UER-py consists of the following target modules:
- lm_target.py: language model
- mlm_target.py: masked language model (cloze test)
- nsp_target.py: next sentence prediction
- cls_target.py: classification
- s2s_target.py: supports autoencoder and machine translation
- bert_target.py: masked language model + next sentence prediction

If multiple machines are available, each machine contains a part of corpus. The command is identical with the single machine case.

### Pretrain the model
```
usage: pretrain.py [-h] [--dataset_path DATASET_PATH] --vocab_path VOCAB_PATH
                   [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   --output_model_path OUTPUT_MODEL_PATH
                   [--config_path CONFIG_PATH] [--total_steps TOTAL_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--report_steps REPORT_STEPS]
                   [--accumulation_steps ACCUMULATION_STEPS]
                   [--batch_size BATCH_SIZE]
                   [--emb_size EMB_SIZE] [--hidden_size HIDDEN_SIZE]
                   [--feedforward_size FEEDFORWARD_SIZE]
                   [--kernel_size KERNEL_SIZE] [--heads_num HEADS_NUM]
                   [--layers_num LAYERS_NUM] [--dropout DROPOUT] [--seed SEED]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                   [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                   [--labels_num LABELS_NUM] [--learning_rate LEARNING_RATE]
                   [--warmup WARMUP] [--world_size WORLD_SIZE]
                   [--gpu_ranks GPU_RANKS [GPU_RANKS ...]]
                   [--master_ip MASTER_IP] [--backend {nccl,gloo}]
```

Notice that it is recommended to explicitly specify model's encoder and target. UER-py consists of the following encoder modules:
- rnn_encoder.py: contains (bi-)LSTM and (bi-)GRU
- cnn_encoder.py: contains CNN and gatedCNN
- attn_encoder.py: contains attentionNN
- gpt_encoder.py: contains GPT encoder
- bert_encoder.py: contains BERT encoder
- mixed_encoder.py: contains combinations of basic encoders, such as RCNN (RNN+CNN), CRNN (CNN+RNN)

The target should be coincident with the target in pre-processing stage. Users can try different combinations of encoders and targets by *--encoder* and *--target*.

There are two strategies for pre-training: 1）random initialization 2）loading a pre-trained model.
#### Random initialization
The example of pre-training on CPU：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert
```
The example of pre-training on single GPU (the id of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert --gpu_ranks 3
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
                    --output_model_path models/output_model.bin --encoder bert --target bert --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
The example of pre-training on two machines, each has 8 GPUs (16 GPUs in total): 
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
            --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
            --master_ip tcp://node-0-addr:port
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
            --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
            --master_ip tcp://node-0-addr:port            
```

#### Load a pre-trained model
We recommend to load a pre-trained model. We can specify the pre-trained model by *--pretrained_model_path* .
The example of pre-training on CPU and single GPU:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
                    --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
                    --encoder bert --target bert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
                    --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
                    --encoder bert --target bert --gpu_ranks 3
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
                    --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
                    --encoder bert --target bert --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
The example of pre-training on two machines, each has 8 GPUs (16 GPUs in total): 
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
            --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
            --encoder bert --target bert --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 --master_ip tcp://node-0-addr:port
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
            --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
            --encoder bert --target bert --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 --master_ip tcp://node-0-addr:port
```

#### Try pre-training models with different targets and encoders
UER-py allows users to combine different components (e.g. subencoders, encoders, and targets). Here is an example of trying different targets:

In fact, NSP target and sentence-level reviews are incompatible to some extent. We could replace BERT target with MLM target on book review dataset:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 --encoder bert --target mlm
```
Notice that different targets correspond to different corpus formats. It is important to select proper format for a target. 
If we want to change encoder, only thing we need to do is to specify --encoder in pretrain.py. Here is an example of using LSTM for pre-training. 
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \ 
                    --encoder lstm --target lm --learning_rate 1e-3 --config_path models/rnn_config.json
```


### Fine-tune on downstream tasks
Currently, UER-py consists of 4 downstream tasks, i.e. classification, sequence labeling, cloze test, feature extractor. The encoder of downstream task should be coincident with the pre-trained model.

#### Classification
classifier.py adds two feedforward layers upon encoder layer.
```
usage: classifier.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                     [--output_model_path OUTPUT_MODEL_PATH]
                     [--vocab_path VOCAB_PATH] --train_path TRAIN_PATH
                     --dev_path DEV_PATH --test_path TEST_PATH
                     [--config_path CONFIG_PATH] [--batch_size BATCH_SIZE]
                     [--seq_length SEQ_LENGTH]
                     [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                     [--bidirectional] [--pooling {mean,max,first,last}]
                     [--subword_type {none,char}]
                     [--sub_vocab_path SUB_VOCAB_PATH]
                     [--subencoder {avg,lstm,gru,cnn}]
                     [--sub_layers_num SUB_LAYERS_NUM]
                     [--tokenizer {bert,char,word,space}]
                     [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                     [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                     [--report_steps REPORT_STEPS] [--seed SEED]
                     [--mean_reciprocal_rank]
```
The example of using classifier.py：
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.tsv --dev_path datasets/book_review/dev.tsv --test_path datasets/book_review/test.tsv \
                      --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using classifier.py for pair classification:
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/lcqmc/train.tsv --dev_path datasets/lcqmc/dev.tsv --test_path datasets/lcqmc/test.tsv \
                      --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using classifier.py for dbqa:
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/dbqa/train.tsv --dev_path datasets/dbqa/dev.tsv --test_path datasets/dbqa/test.tsv \
                      --epochs_num 3 --batch_size 64 --encoder bert --mean_reciprocal_rank
```

#### Sequence labeling
tagger.py adds a feedforward layer upon encoder layer.
```
usage: tagger.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                 [--output_model_path OUTPUT_MODEL_PATH]
                 [--vocab_path VOCAB_PATH] [--train_path TRAIN_PATH]
                 [--dev_path DEV_PATH] [--test_path TEST_PATH]
                 [--config_path CONFIG_PATH] [--batch_size BATCH_SIZE]
                 [--seq_length SEQ_LENGTH]
                 [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                 [--bidirectional] [--subword_type {none,char}]
                 [--sub_vocab_path SUB_VOCAB_PATH]
                 [--subencoder {avg,lstm,gru,cnn}]
                 [--sub_layers_num SUB_LAYERS_NUM]
                 [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                 [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                 [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using tagger.py：
```
python3 tagger.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                  --train_path datasets/msra/train.tsv --dev_path datasets/msra/dev.tsv --test_path datasets/msra/test.tsv \
                  --epochs_num 5 --batch_size 32 --encoder bert
```

#### Cloze test
cloze.py predicts masked words. Top n words are returned.
```
usage: cloze.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                [--vocab_path VOCAB_PATH] [--input_path INPUT_PATH]
                [--output_path OUTPUT_PATH] [--config_path CONFIG_PATH]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                [--subword_type {none,char}] [--sub_vocab_path SUB_VOCAB_PATH]
                [--subencoder_type {avg,lstm,gru,cnn}]
                [--tokenizer {bert,char,word,space}] [--topn TOPN]
```
The example of using cloze.py：
```
python3 cloze.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                 --input_path datasets/cloze_input.txt --output_path output.txt

```

#### Feature extractor
feature_extractor.py extracts sentence embeddings.
```
usage: feature_extractor.py [-h] --input_path INPUT_PATH --model_path
                            MODEL_PATH --vocab_path VOCAB_PATH --output_path
                            OUTPUT_PATH [--subword_type {none,char}]
                            [--sub_vocab_path SUB_VOCAB_PATH]
                            [--subencoder {avg,lstm,gru,cnn}]
                            [--sub_layers_num SUB_LAYERS_NUM]
                            [--seq_length SEQ_LENGTH]
                            [--batch_size BATCH_SIZE]
                            [--config_path CONFIG_PATH]
                            [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                            [--target {bert,lm,cls,mlm,nsp,s2s}]
                            [--tokenizer {char,word,space,mixed}]
```
The example of using feature_extractor.py：
```
python3 feature_extractor.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_model.bin \
                             --vocab_path models/google_vocab.txt --output_path output.npy
```

### Finding nearest neighbours
Pre-trained models can learn high-quality word embeddings. Traditional word embeddings such as word2vec and GloVe assign each word a fixed vector. However, polysemy is a pervasive phenomenon in human language, and the meanings of a polysemous word depend on the context. To this end, we use a the hidden state in pre-trained models to represent a word. It is noticeable that Google BERT is a character-based model. To obtain real word embedding (not character embedding), Users should download our [word-based BERT model](https://share.weiyun.com/5s4HVMi) and [vocabulary](https://share.weiyun.com/5NWYbYn).
The example of using scripts/topn_words_indep.py (finding nearest neighbours for context-independent word embedding)：
```
python3 scripts/topn_words_indep.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                                    --cand_vocab_path models/google_vocab.txt --target_words_path target_characters.txt
```
Contexct-independent word embedding is obtained model's embedding layer.
The format of the target_characters.txt is as follows:
```
word-1
word-2
...
word-n
```
The example of using scripts/topn_words_dep.py (finding nearest neighbours for context-dependent word embedding)：
```
python3 scripts/topn_words_dep.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                                  --cand_vocab_path models/google_vocab.txt --sent_path target_characters_with_sentences.txt --config_path models/google_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert
```
We substitute the target word with other words in the vocabulary and feed the sentences into the pretrained model. Hidden state is used as the context-dependent embedding of a word. Select proper tokenizer according to the sentence in target_characters_with_sentences.txt. The format of 
target_words_with_sentences.txt is as follows:
```
sent1 word1
sent1 word1
...
sentn wordn
```
Sentence and word are splitted by \t. 
<br/>

## Scripts
<table>
<tr align="center"><th> Scripts <th> Function description
<tr align="center"><td> average_model.py <td> Take the average of pre-trained models. A frequently-used ensemble strategy for deep learning models 
<tr align="center"><td> build_vocab.py <td> Build vocabulary (multi-processing supported)
<tr align="center"><td> check_model.py <td> Check the model (single GPU or multiple GPUs)
<tr align="center"><td> diff_vocab.py <td> Compare two vocabularies
<tr align="center"><td> dynamic_vocab_adapter.py <td> Change the pre-trained model according to the vocabulary. It can save memory in fine-tuning stage since task-specific vocabulary is much smaller than general-domain vocabulary 
<tr align="center"><td> multi_single_convert.py <td> convert the model (single GPU or multiple GPUs)
<tr align="center"><td> topn_words_indep.py <td> Finding nearest neighbours with context-independent word embedding
<tr align="center"><td> topn_words_dep.py <td> Finding nearest neighbours with context-dependent word embedding
</table>


<br/>

## Experiments
### Speed
```
GPU：Tesla P40

CPU：Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

```
We use BERT to test the speed of distributed training mode. Google BERT is trained for 1 million steps and each step contains 128,000 tokens. It takes around 18 days to reproduce the experiments by UER-py on 3 GPU machines (24 GPU in total).

<table>
<tr align="center"><th> #(machine) <th> #(GPU)/machine <th> tokens/second
<tr align="center"><td> 1 <td> 0 <td> 276
<tr align="center"><td> 1 <td> 1 <td> 7050
<tr align="center"><td> 1 <td> 2 <td> 13071
<tr align="center"><td> 1 <td> 4 <td> 24695
<tr align="center"><td> 1 <td> 8 <td> 44300
<tr align="center"><td> 3 <td> 8 <td> 84386
</table>
 
### Qualitative evaluation
We qualitatively evaluate pre-trained models by finding words' near neighbours.

#### Character-based model
Evaluation of context-independent word embedding:

<table>
<tr align="center"><td> Target word: 苹 <td> <td> Target word: 吃 <td>  <td> Target word: 水 <td> 
<tr align="center"><td> 蘋 <td> 0.762  <td> 喝 <td> 0.539 <td> 河 <td> 0.286 
<tr align="center"><td> apple <td> 0.447  <td> 食 <td> 0.475 <td> 海 <td> 0.278
<tr align="center"><td> iphone <td> 0.400 <td> 啃 <td> 0.340 <td> water <td> 0.276
<tr align="center"><td> 柠 <td> 0.347  <td> 煮 <td> 0.324 <td> 油 <td> 0.266
<tr align="center"><td> ios <td> 0.317  <td> 嚐 <td> 0.322 <td> 雨 <td> 0.259
</table>


Evaluation of context-dependent word embedding:

Target sentence: 其冲积而形成小平原沙土层厚而肥沃，盛产苹果、大樱桃、梨和葡萄。
<table>
<tr align="center"><td> Target word: 苹 <td> 
<tr align="center"><td> 蘋 <td> 0.822
<tr align="center"><td> 莓 <td> 0.714 
<tr align="center"><td> 芒 <td> 0.706
<tr align="center"><td> 柠 <td> 0.704
<tr align="center"><td> 樱 <td> 0.696 
</table>


Target sentence: 苹果削减了台式Mac产品线上的众多产品
<table>
<tr align="center"><td> Target word: 苹 <td> 
<tr align="center"><td> 蘋 <td> 0.892
<tr align="center"><td> apple <td> 0.788 
<tr align="center"><td> iphone <td> 0.743
<tr align="center"><td> ios <td> 0.720
<tr align="center"><td> ipad <td> 0.706 
</table>

#### Word-based model
Evaluation of context-independent word embedding:

<table>
<tr align="center"><td> Target word: 苹果 <td> <td> Target word: 腾讯 <td>  <td>  Target word: 吉利 <td> 
<tr align="center"><td> 苹果公司 <td> 0.419  <td> 新浪 <td> 0.357 <td> 沃尔沃 <td> 0.277
<tr align="center"><td> apple <td> 0.415  <td> 网易 <td> 0.356 <td> 伊利 <td> 0.243
<tr align="center"><td> 苹果电脑 <td> 0.349 <td> 搜狐 <td> 0.356 <td> 长荣 <td> 0.235
<tr align="center"><td> 微软 <td> 0.320  <td> 百度 <td> 0.341 <td> 天安 <td> 0.224
<tr align="center"><td> mac <td> 0.298  <td> 乐视 <td> 0.332 <td> 哈达 <td> 0.220
</table>


Evaluation of context-dependent word embedding:

Target sentence: 其冲积而形成小平原沙土层厚而肥沃，盛产苹果、大樱桃、梨和葡萄。
<table>
<tr align="center"><td> Target word: 苹果 <td> 
<tr align="center"><td> 柠檬 <td> 0.734
<tr align="center"><td> 草莓 <td> 0.725 
<tr align="center"><td> 荔枝 <td> 0.719
<tr align="center"><td> 树林 <td> 0.697
<tr align="center"><td> 牡丹 <td> 0.686 
</table>

 
Target sentence: 苹果削减了台式Mac产品线上的众多产品
<table>
<tr align="center"><td> Target word: 苹果 <td> 
<tr align="center"><td> 苹果公司 <td> 0.836
<tr align="center"><td> apple <td> 0.829
<tr align="center"><td> 福特 <td> 0.796
<tr align="center"><td> 微软 <td> 0.777
<tr align="center"><td> 苹果电脑 <td> 0.773 
</table>

 
Target sentence: 讨吉利是通过做民间习俗的吉祥事，或重现过去曾经得到好结果的行为，以求得好兆头。
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 仁德 <td> 0.749
<tr align="center"><td> 光彩 <td> 0.743
<tr align="center"><td> 愉快 <td> 0.736
<tr align="center"><td> 永元 <td> 0.736
<tr align="center"><td> 仁和 <td> 0.732 
</table>

 
Target sentence: 2010年6月2日福特汽车公司宣布出售旗下高端汽车沃尔沃予中国浙江省的吉利汽车，同时将于2010年第四季停止旗下中阶房车品牌所有业务
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 沃尔沃 <td> 0.771
<tr align="center"><td> 卡比 <td> 0.751
<tr align="center"><td> 永利 <td> 0.745
<tr align="center"><td> 天安 <td> 0.741
<tr align="center"><td> 仁和 <td> 0.741 
</table>

 
Target sentence: 主要演员有扎克·布拉夫、萨拉·朝克、唐纳德·费森、尼尔·弗林、肯·詹金斯、约翰·麦吉利、朱迪·雷耶斯、迈克尔·莫斯利等。
<table>
<tr align="center"><td> Target word: 吉利 <td> 
<tr align="center"><td> 玛利 <td> 0.791
<tr align="center"><td> 米格 <td> 0.768
<tr align="center"><td> 韦利 <td> 0.767
<tr align="center"><td> 马力 <td> 0.764
<tr align="center"><td> 安吉 <td> 0.761 
</table>

### Quantitative evaluation
We use a range of Chinese datasets to evaluate the performance of UER-py. Douban book review, ChnSentiCorp, Shopping, and Tencentnews are sentence-level small-scale sentiment classification datasets. MSRA-NER is a sequence labeling dataset. These datasets are included in this project. Dianping, JDfull, JDbinary, Ifeng, and Chinanews are large-scale classification datasets. They are collected in [glyph](https://arxiv.org/pdf/1708.02657.pdf) and can be downloaded at [glyph's github project](https://github.com/zhangxiangxiao/glyph). These five datasets don't contain validation set. We use 10% instances in trainset for validation.

Most pre-training models consist of 2 stages: pre-training on general-domain corpus and fine-tuning on downstream dataset. We recommend 3-stage mode: 1)Pre-training on general-domain corpus; 2)Pre-training on downstream dataset; 3)Fine-tuning on downstream dataset. Stage 2 enables models to get familiar with distributions of downstream tasks. It is sometimes known as semi-supervised fune-tuning.

Hyper-parameter settings are as follows:
- Stage 1: We train with batch size of 256 sequences and each sequence contains 256 tokens. We load Google's pretrained models and train upon it for 500,000 steps. The learning rate is 2e-5 and other optimizer settings are identical with Google BERT. BERT tokenizer is used.
- Stage 2: We train with batch size of 256 sequences. For classification datasets, the sequence length is 128. For sequence labeling datasets, the sequence length is 256. We train upon Google's pretrained model for 20,000 steps. Optimizer settings and tokenizer are identical with stage 1.
- Stage 3: For classification datasets, the training batch size and epochs are 64 and 3. For sequence labeling datasets, the training batch size and epochs are 32 and 5. Optimizer settings and tokenizer are identical with stage 1.

We provide the pre-trained models (using BERT target) on different downstream datasets: [book_review_model.bin](https://share.weiyun.com/52BEFs2); [chnsenticorp_model.bin](https://share.weiyun.com/53WDBeJ); [shopping_model.bin](https://share.weiyun.com/5HaxwAf); [msra_model.bin](https://share.weiyun.com/5E6XpEt). Tencentnews dataset and its pretrained model will be publicly available after data desensitization. 

<table>
<tr align="center"><th> Model/Dataset              <th> Douban book review <th> ChnSentiCorp <th> Shopping <th> MSRA-NER <th> Tencentnews review
<tr align="center"><td> BERT                       <td> 87.5               <td> 94.3         <td> 96.3     <td> 93.0/92.4/92.7  <td> 84.2
<tr align="center"><td> BERT+semi_BertTarget       <td> 88.1               <td> 95.6         <td> 97.0     <td> 94.3/92.6/93.4  <td> 85.1
<tr align="center"><td> BERT+semi_MlmTarget        <td> 87.9               <td> 95.5         <td> 97.1     <td>   <td> 85.1
</table>

Pre-training is also important for other encoders and targets. We pre-train a 2-layer LSTM on 1.9G review corpus with language model target. Embedding size and hidden size are 512. The model is much more efficient than BERT in pre-training and fine-tuning stages. We show that pre-training brings significant improvements and achieves competitive results (the differences are not big compared with the results of BERT).

<table>
<tr align="center"><th> Model/Dataset              <th> Douban book review <th> ChnSentiCorp <th> Shopping 
<tr align="center"><td> BERT                       <td> 87.5               <td> 94.3         <td> 96.3     
<tr align="center"><td> LSTM                       <td> 80.2               <td> 88.3         <td> 94.4     
<tr align="center"><td> LSTM+pre-training          <td> 86.6(+6.4)         <td> 94.5(+6.2)   <td> 96.5(+2.1)     
</table>


It requires tremendous computional resources to fine-tune on large-scale datasets. For Ifeng, Chinanews, Dianping, JDbinary, and JDfull datasets, we provide their classification models (see Chinese model zoo). Classification models on large-scale datasets allow users to reproduce the results without training. Besides that, classification models could be used for improving other related tasks. More experimental results will come soon. 

Ifeng and Chinanews datasets contain news' titles and abstracts. In stage 2, we use title to predict abstract. 

<table>
<tr align="center"><th> Model/Dataset              <th> Ifeng     <th> Chinanews <th> Dianping <th> JDbinary <th> JDfull
<tr align="center"><td> pre-SOTA (Glyph & Glyce)   <td> 85.76     <td> 91.88     <td> 78.46    <td> 91.76    <td> 54.24 
<tr align="center"><td> BERT                       <td> 87.50     <td> 93.37     <td>          <td> 92.37    <td> 54.79
<tr align="center"><td> BERT+semi+BertTarget       <td> 87.65     <td>           <td>          <td>          <td> 
</table>

We also provide the pre-trained models on different corpora, encoders, and targets (see Chinese model zoo). Selecting proper pre-training models is beneficial to the performance of downstream tasks.

<table>
<tr align="center"><th> Model/Dataset                     <th> MSRA-NER
<tr align="center"><td> Wikizh corpus (Google)            <td> 93.0/92.4/92.7
<tr align="center"><td> Renminribao corpus                <td> 94.4/94.4/94.4
</table>


<br/>

## Chinese_model_zoo
With the help of UER, we are pre-training models with different corpora, encoders, and targets.
<table>
<tr align="center"><th> pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh+BertEncoder+BertTarget <td> https://share.weiyun.com/5s9AsfQ <td> The training corpus is Wiki_zh, trained by Google
<tr align="center"><td> Wikizh(word-based)+BertEncoder+BertTarget <td> Model: https://share.weiyun.com/5s4HVMi Vocab: https://share.weiyun.com/5NWYbYn <td> Word-based BERT model trained on Wikizh. Training steps: 500,000
<tr align="center"><td> RenMinRiBao+BertEncoder+BertTarget <td> https://share.weiyun.com/5JWVjSE <td> The training corpus is news data from People's Daily (1946-2017). It is suitable for datasets related with news, e.g. F1 is improved on MSRA-NER from 92.6 to 94.4 (compared with Google BERT). Training steps: 500,000
<tr align="center"><td> Webqa2019+BertEncoder+BertTarget <td> https://share.weiyun.com/5HYbmBh <td> The training corpus is WebQA, which is suitable for datasets related with social media, e.g. Accuracy (dev/test) on LCQMC is improved from 88.8/87.0 to 89.6/87.4; Accuracy (dev/test) on XNLI is improved from 78.1/77.2 to 79.0/78.8 (compared with Google BERT). Training steps: 500,000
<tr align="center"><td> Google-BERT-en-uncased-base/large <td> Base: https://share.weiyun.com/5hWivED  Large: https://share.weiyun.com/5ghgDjT Vocab: https://share.weiyun.com/5MJZAfa <td> Provided by Google.
<tr align="center"><td> Reviews+LstmEncoder+LmTarget <td> https://share.weiyun.com/57dZhqo  <td> The training corpus is amazon reviews + JDbinary reviews + dainping reviews (11.4M reviews in total). Language model target is used. It is suitable for datasets related with reviews. It achieves over 5 percent improvements on some review datasets compared with random initialization. Training steps: 200,000; Sequence length: 128
<tr align="center"><td> (Mixedlarge corpus & Amazon reviews)+LstmEncoder+(LmTarget & ClsTarget) <td> https://share.weiyun.com/5B671Ik  <td> Mixedlarge corpus contains baidubaike + wiki + webqa + RenMinRiBao. The model is trained on it with language model target. And then the model is trained on Amazon reviews with language model and classification targets. It is suitable for datasets related with reviews. It can achieve comparable results with BERT on some review datasets. Training steps: 500,000 + 100,000; Sequence length: 128

<tr align="center"><td> IfengNews+BertEncoder+BertTarget <td> https://share.weiyun.com/5HVcUWO <td> The training corpus is news data from Ifeng website. We use news titles to predict news abstracts. Training steps: 100,000; Sequence length: 128
<tr align="center"><td> jdbinary+BertEncoder+ClsTarget <td> https://share.weiyun.com/596k2bu <td> The training corpus is review data from JD (jingdong). Classification target is used for pre-training. It is suitable for datasets related with shopping reviews, e.g. accuracy is improved on shopping datasets from 96.3 to 97.2 (compared with Google BERT). Training steps: 50,000; Sequence length: 128
<tr align="center"><td> jdfull+BertEncoder+MlmTarget <td> https://share.weiyun.com/5L6EkUF <td> The training corpus is review data from JD (jingdong). Masked LM target is used for pre-training. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> Amazonreview+BertEncoder+ClsTarget <td> https://share.weiyun.com/5XuxtFA <td> The training corpus is review data from Amazon (including book reviews, movie reviews, and etc.). Classification target is used for pre-training. It is suitable for datasets related with reviews, e.g. accuracy is improved on Douban book review datasets from 87.6 to 88.5 (compared with Google BERT). Training steps: 20,000; Sequence length: 128
<tr align="center"><td> XNLI+BertEncoder+ClsTarget <td> https://share.weiyun.com/5oXPugA <td> Infersent with BertEncoder
<tr align="center"><td> <td> <td> 
</table>

We release the classification models on 5 large-scale datasets, i.e. Ifeng, Chinanews, Dianping, JDbinary, and
JDfull. Users can use these models to reproduce results, or regard them as pre-training models for other datasets.
<table>
<tr align="center"><th> Datasets <th> Link  
<tr align="center"><td> Ifeng <td> https://share.weiyun.com/5ZCp4wU 
<tr align="center"><td> Chinanews <td> https://share.weiyun.com/5bSfeQ7
<tr align="center"><td> Dianping <td> https://share.weiyun.com/5Ls8R02
<tr align="center"><td> JDbinary <td> https://share.weiyun.com/5QNu4QF
<tr align="center"><td> JDfull <td> https://share.weiyun.com/5bqchN1
</table>

<br/>

## Contact information
For communication related to this project, please contact Zhe Zhao (helloworld@ruc.edu.cn; nlpzhezhao@tencent.com) or Xin Zhao (2014201975@ruc.edu.cn).


