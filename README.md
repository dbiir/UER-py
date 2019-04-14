# UER-py

<img src="uer-logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks and has led to remarkable improvements. UER-py is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of different pre-training models (such as BERT), and provides interfaces for users to further extend upon. UER-py also incorporates many mechanisms for better performance and efficiency. It has been tested on serveal Chinese datasets and should match or even outperform Google's TF implementation.

<br>

Table of Contents
=================
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Instructions](#instructions)
  * [Modules](#modules)
  * [Scripts](#scripts)
  * [Experiments](#experiments)
  * [Chinese_model_zoo](#chinese_model_zoo)
<br/>

## Features
UER-py has the following features:
- __Reliable implementation.__ UER-py is able to reproduce the results of existing pre-training models (such as [Google BERT](https://github.com/google-research/bert)).
- __Multi-GPU.__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity.__ UER-py is divided into four components: subencoder, encoder, target, and fine-tuning. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.
- __Efficiency.__ UER-py incorporates many mechanisms in pre-processing, pre-training, and fine-tuning stages, which largely boosts the efficiency in speed and memory.
- __SOTA results.__ Our works further improve the results upon Google BERT, providing new strong baselines for a range of datasets.
- __Chinese model zoo.__ We are pre-training models with different corpora, encoders, and targets.


<br/>

## Quickstart
We use BERT model and book review classification dataset to demonstrate how to UER-py. There are three input files: book review corpus, book review dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

The book review corpus is obtained by book review dataset with labels removed. The format of the corpus for BERT is as follows：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```

The format of the classification dataset is as follows (label and instance are separated by \t):
```
1 instance1
0 instance2
1 instance3
```

We use Google's Chinese vocabulary file, which contains 21128 Chinese characters. The format of the vocabulary is as follows:
```
word-1
word-2
...
word-n
```

Suppose we have a machine with 8 GPUs.
First of all, we preprocess the book review corpus (from downstream dataset):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt --dataset_path dataset \
                      --dataset_split_num 8 --target bert
```
Since we have 8 GPUs, we split the *dataset* into 8 parts. Each GPU processes one part.
We download [Google's pre-trained Chinese model](https://share.weiyun.com/51tMpcr), and put it into *models* file.
Then we load Google's pre-trained model and train on book review corpus.
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 1000 --target bert
```
Finally, we do classification. We can use google_model.bin:
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
    --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt --test_path datasets/book_review/test.txt \
    --epochs_num 3 --batch_size 64
```
or use our book_review_model.bin：
```
python3 classifier.py --pretrained_model_path models/review_model.bin --vocab_path models/google_vocab.txt \
    --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt --test_path datasets/book_review/test.txt \
    --epochs_num 3 --batch_size 64
```
It turns out that the result of Google's model is 87.5; The result of book_review_model.bin is 88.1.


<br/>

## Instructions
### UER-py's framework
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/: contains encoders such as RNN, CNN, Attention, BERT
    |    |--layers/: contains common NN layers, such as embedding layer, normalization layer
    |    |--models/: contains model.py, which combines subencoder, embedding, encoder, and target modules
    |    |--subencoders/: contains subencoders such as RNN, CNN and different pooling strategies
    |    |--targets/: contains targets such as language model, masked language model, sentence prediction
    |    |--utils/: contains common utilities
    |    |--model_builder.py 
    |    |--model_saver.py
    |    |--trainer.py
    |
    |--corpora/: contains corpora for pre-training
    |--datasets/: contains downstream tasks
    |--models/: contains pre-trained models, vocabularies, and config files
    |--scripts/
    |
    |--preprocess.py
    |--pretrain.py
    |--classifier.py
    |--cloze.py
    |--tagger.py
    |--feature_extractor.py
    |--README.md
```

Next, we provide detailed instructions of UER-py. UER-py uses BERT encoder and BERT target by default.

### Preprocess the data
```
usage: preprocess.py [-h] --corpus_path CORPUS_PATH --vocab_path VOCAB_PATH
                     [--dataset_path DATASET_PATH]
                     [--tokenizer {bert,char,word,space}]
                     [--dataset_split_num DATASET_SPLIT_NUM]
                     [--docs_buffer_size DOCS_BUFFER_SIZE]
                     [--seq_length SEQ_LENGTH] [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--seed SEED]
```
Example of using CPU and single GPU for training. UER-py uses BERT target by default. Here we specify the target explicitly：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt \
                      --dataset_path dataset --target bert
```
Example of using distributed mode for training (single machine). *--dataset_split_num n* represents the corpus is divided into n parts. During the pre-training stage, each process handles one part. Suppose we have 8 GPUs, the n is set to 8：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt --dataset_path dataset --dataset_split_num 8
```
Example of using distributed mode for training (multiple machines). Suppose we have two machines, each has 8 GPUs (16 GPUs in total):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt --dataset_path dataset --dataset_split_num 16
```
We can obtain 16 datasets: from dataset-0.pt to dataset-15.pt. We need to copy dataset-8.pt~dataset-15.pt to the second machine.


### Pretrain the model
There two strategies for pre-training: 1）random initialization 2）loading a pre-trained model.
```
usage: pretrain.py [-h] [--dataset_path DATASET_PATH] --vocab_path VOCAB_PATH
                   [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   --output_model_path OUTPUT_MODEL_PATH
                   [--config_path CONFIG_PATH] [--total_steps TOTAL_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--report_steps REPORT_STEPS] [--batch_size BATCH_SIZE]
                   [--instances_buffer_size INSTANCES_BUFFER_SIZE]
                   [--emb_size EMB_SIZE] [--hidden_size HIDDEN_SIZE]
                   [--feedforward_size FEEDFORWARD_SIZE]
                   [--heads_num HEADS_NUM] [--layers_num LAYERS_NUM]
                   [--dropout DROPOUT] [--seed SEED]
                   [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                   [--world_size WORLD_SIZE]
                   [--gpu_ranks GPU_RANKS [GPU_RANKS ...]]
                   [--master_ip MASTER_IP] [--backend {nccl,gloo}]
```
#### Random initialization
Pre-training on CPU. UER-py uses BERT encoder and BERT target by default. Here we specify the encoder and the target explicitly：
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt --output_model_path models/model.bin --encoder bert --target bert
```
Pre-training on single GPU. The id of GPU is 3：
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt --output_model_path models/model.bin --gpu_ranks 3
```
Pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
                    --output_model_path models/model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
Pre-training on two nachines, each has 8 GPUs (16 GPUs in total): 
```
Node-0 : python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
            --output_model_path models/model.bin --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
            --master_ip tcp://node-0-addr:port
Node-1 : python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
            --output_model_path models/model.bin --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
            --master_ip tcp://node-0-addr:port            
```

#### Loading a pre-trained model
We recommend to load a pre-trained model. We can specify the pre-trained model by *--pretrained_model_path* .
Pre-training on CPU and single GPU:
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
                    --pretrained_model_path models/google_model.bin --output_model_path models/model.bin \
                    --encoder bert --target bert
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
                    --pretrained_model_path models/google_model.bin --output_model_path models/model.bin \
                    --gpu_ranks 3 --encoder bert --target bert
```
Pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin \
                    --output_model_path models/model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
Pre-training on two nachines, each has 8 GPUs (16 GPUs in total): 
```
Node-0 : python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
            --output_model_path models/model.bin --pretrained_model_path models/google_model.bin\
            --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 --master_ip tcp://node-0-addr:port
Node-1 : python3 pretrain.py --dataset_path dataset --vocab_path models/google_vocab.txt \
            --output_model_path models/model.bin --pretrained_model_path models/google_model.bin \
            --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 --master_ip tcp://node-0-addr:port
```

### Fine-tune on downstream tasks
Currently, UER-py consists of 4 downstream tasks, i.e. classification, sequence labeling, cloze test, feature extractor.

#### Classification
classifier.py adds two feedforward layers upon encoder layer.
```
usage: classifier.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                     [--output_model_path OUTPUT_MODEL_PATH]
                     [--vocab_path VOCAB_PATH] --train_path TRAIN_PATH
                     --dev_path DEV_PATH --test_path TEST_PATH
                     [--config_path CONFIG_PATH] [--batch_size BATCH_SIZE]
                     [--seq_length SEQ_LENGTH]
                     [--tokenizer {bert,char,word,space}]
                     [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                     [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                     [--report_steps REPORT_STEPS] [--seed SEED]
```
Example of using classifier.py：
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt \
                      --test_path datasets/book_review/test.txt --epochs_num 3 --batch_size 64

```
#### Sequence labeling
tagger.py adds a feedforward layer upon encoder layer.
```
usage: tagger.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                 [--output_model_path OUTPUT_MODEL_PATH]
                 [--vocab_path VOCAB_PATH] [--train_path TRAIN_PATH]
                 [--dev_path DEV_PATH] [--test_path TEST_PATH]
                 [--config_path CONFIG_PATH] [--batch_size BATCH_SIZE]
                 [--seq_length SEQ_LENGTH] [--learning_rate LEARNING_RATE]
                 [--warmup WARMUP] [--dropout DROPOUT]
                 [--epochs_num EPOCHS_NUM] [--report_steps REPORT_STEPS]
                 [--seed SEED]
```
Example of using tagger.py：
```
python3 tagger.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                  --train_path datasets/msra/train.txt --dev_path datasets/msra/dev.txt --test_path datasets/msra/test.txt \
                  --epochs_num 5 --batch_size 32
```

#### Cloze test
cloze.py predicts masked words. Top n words are returned.
```
usage: cloze.py [-h] [--model_path MODEL_PATH] [--vocab_path VOCAB_PATH]
                [--input_path INPUT_PATH] [--output_path OUTPUT_PATH]
                [--config_path CONFIG_PATH] [--batch_size BATCH_SIZE]
                [--seq_length SEQ_LENGTH] [--tokenizer {bert,char,word,space}]
                [--topn TOPN]
```
Example of using cloze.py：
```
python3 cloze.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                 --input_path ./datasets/cloze_input.txt --output_path output.txt

```

#### Feature extractor
feature_extractor.py extracts sentence embeddings.
```
usage: feature_extractor.py [-h] --input_path INPUT_PATH --model_path
                            MODEL_PATH --vocab_path VOCAB_PATH --output_path
                            OUTPUT_PATH [--seq_length SEQ_LENGTH]
                            [--batch_size BATCH_SIZE]
                            [--config_path CONFIG_PATH]
                            [--tokenizer {bert,char,word,space}]
```
Using example of feature_extractor.py：
```
python3 feature_extractor.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_model.bin \
                             --vocab_path models/google_vocab.txt --output_path output.npy
```

<br/>

## Modules
### Encoder

UER-py incorporates ample encoder modules:
- rnn_encoder.py: contains (bi-)LSTM and (bi-)GRU.
- cnn_encoder.py: contains CNN and gatedCNN
- attn_encoder.py: contains attention neural network
- gpt_encoder.py: contains GPT encoder
- bert_encoder.py: contains BERT encoder, a 12 transformer layers
- mixed_encoder.py: combined basic encoders, such as RCNN (RNN+CNN), CRNN (CNN+RNN)

### Target

UER-py incorporates ample target modules:
- lm_target.py: language model
- mlm_target.py: masked language model (cloze test)
- nsp_target.py: next sentence prediction
- cls_target.py: classification
- s2s_target.py: supports autoencoder and machine translation target
- bert_target.py: masked language model + next sentence prediction

<br/>

## Scripts
<table>
<tr align="center"><th> Scripts <th> Function description
<tr align="center"><td> average_model.py <td> Take the average of pre-trained model. A common ensemble strategy for deep learning models 
<tr align="center"><td> build_vocab.py <td> Build vocabulary (multi-processing supported)
<tr align="center"><td> check_model.py <td> Check the model (single GPU or multiple GPUs)
<tr align="center"><td> diff_vocab.py <td> Compare two vocabularies
<tr align="center"><td> dynamic_vocab_adapter.py <td> Change the pre-trained model according to the vocabulary
<tr align="center"><td> multi_single_convert.py <td> convert the model (single GPU or multiple GPUs)
</table>


<br/>

## Experiments
### Speed
```
GPU：Tesla P40

CPU：Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

```

<table>
<tr align="center"><th> #(machine) <th> #(GPU)/machine <th> tokens/second
<tr align="center"><td> 1 <td> 0 <td> 276
<tr align="center"><td> 1 <td> 1 <td> 7050
<tr align="center"><td> 1 <td> 2 <td> 13071
<tr align="center"><td> 1 <td> 4 <td> 24695
<tr align="center"><td> 1 <td> 8 <td> 44300
<tr align="center"><td> 3 <td> 8 <td> 84386
</table>

### Performance
We use a range of Chinese datasets to evaluate the performance of UER-py. These datasets are included in this project. One can reproduce our results with little efforts.
Training model on the corpus of downstream task can boost the performance. It is sometimes known as semi-supervised fune-tuning. More improvements will be added soon for better performance.
<table>
<tr align="center"><th> Model/Dataset              <th> Douban book review <th> ChnSentiCorp <th> Shopping <th> MSRA-NER
<tr align="center"><td> BERT                       <td> 87.5               <td> 94.3         <td> 96.3     <td> 93.0/92.4/92.7
<tr align="center"><td> BERT+semi-supervision      <td> 88.1               <td> 95.6         <td> 97.0     <td> 94.3/92.6/93.4
</table>


<br/>

## Chinese_model_zoo
With the help of UER, we are pre-training models with different corpora, encoders, and targets.
<table>
<tr align="center"><th> pre-trained model <th> Link <th> Description 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td>
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
<tr align="center"><td> <td> <td> 
</table>

<br/>

## Organization
Renmin University of China
<br/>
Tencent, Beijing Research
<br/>
Peking University

<br/>

## References
1. Devlin J, Chang M W, Lee K, et al. Bert: Pre-training of deep bidirectional transformers for language understanding[J]. arXiv preprint arXiv:1810.04805, 2018.

