# UER-py

<img src="uer-logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks and has led to remarkable improvements. UER-py is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of different pre-training models (e.g. BERT), and provides interfaces for users to further extend upon. UER-py also incorporates many mechanisms for better performance and efficiency. It has been tested on several Chinese datasets and should match or even outperform Google's TF implementation.

<br>

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
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
Python3.6, PyTorch-1.0.0, CUDA Version 9.0.176, CUDNN 7.0.5


<br/>

## Quickstart
We use BERT model and [Douban book review classification dataset](http://www.cips-cl.org/static/anthology/CCL-2018/CCL-18-086.pdf) to demonstrate how to use UER-py. We firstly pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

The format of the corpus for BERT is as follows：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained by book review dataset. We remove labels and split a review into two parts from the middle. Each review constitutes a document with two sentences. 

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

First of all, we preprocess the book review corpus. We need to specify the model's target in pre-processing stage:
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Pre-processing is time-consuming. Multi-process can largely accelerate the pre-processing speed.
Then  we download [Google's pre-trained Chinese model](https://share.weiyun.com/5DJasRk), and put it into *models* folder.
We load Google's pre-trained model and train on book review corpus. Suppose we have a machine with 8 GPUs. We explicitly specify model's encoder and target:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 20000 --save_checkpoint_steps 1000 --encoder bert --target bert
```
Finally, we do classification. We can use google_model.bin:
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
    --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt --test_path datasets/book_review/test.txt \
    --epochs_num 3 --batch_size 64 --encoder bert
```
or use our [book_review_model.bin](https://share.weiyun.com/52BEFs2), which is the output of pretrain.py：
```
python3 classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_vocab.txt \
    --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt --test_path datasets/book_review/test.txt \
    --epochs_num 3 --batch_size 64 --encoder bert
```
It turns out that the result of Google's model is 87.5; The result of *book_review_model.bin* is 88.1. It is also noticable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

We could search proper pre-trained models in [Chinese model zoo](#chinese_model_zoo) for further improvements. For example, we could download [models pre-trained on Amazon corpus (over 4 million reviews) with BERT encoder and classification target](https://share.weiyun.com/5XuxtFA). It achieves 88.5 accuracy on book review dataset.

<br/>

## Instructions
### UER-py's framework
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/: contains encoders such as RNN, CNN, Attention, CNN-RNN, BERT
    |    |--targets/: contains targets such as language model, masked language model, sentence prediction
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
                     [--seq_length SEQ_LENGTH] [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--seed SEED]
```
*--docs_buffer_size* could be used to control memory consumption in pre-processing stage. *--preprocesses_num n* denotes that n processes are used for pre-processing. The example of pre-processing on a single machine is as follows：
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
                   [--instances_buffer_size INSTANCES_BUFFER_SIZE]
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
*--instances_buffer_size* could be used to control memory consumption in pre-training stage.

Notice that it is recommended to explicitly specify model's encoder and target. UER-py consists of the following encoder modules:
- rnn_encoder.py: contains (bi-)LSTM and (bi-)GRU
- cnn_encoder.py: contains CNN and gatedCNN
- attn_encoder.py: contains attentionNN
- gpt_encoder.py: contains GPT encoder
- bert_encoder.py: contains BERT encoder
- mixed_encoder.py: contains combinations of basic encoders, such as RCNN (RNN+CNN), CRNN (CNN+RNN)

The target should be coincident with the target in pre-processing stage. Users can try different combinations of encoders and targets by *--encoder* and *--target*.

There two strategies for pre-training: 1）random initialization 2）loading a pre-trained model.
#### Random initialization
The example of pre-training on CPU：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert
```
The example of pre-training on single GPU (the id of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert --gpu_ranks 3
```
Pre-training on single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt \
                    --output_model_path models/output_model.bin --encoder bert --target bert --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
Pre-training on two nachines, each has 8 GPUs (16 GPUs in total): 
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

#### Try different pre-training models
UER-py allows users to combine different components (e.g. subencoders, encoders, and targets). Here is an example of trying different targets:

In fact, NSP target and sentence-level reviews are incompatible to some extent. We could replace BERT target with MLM target on book review dataset:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_vocab.txt --dataset_path dataset.pt --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_vocab.txt --pretrained_model_path models/google_model.bin --output_model_path models/book_review_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 1000 --encoder bert --target mlm
```
Notice that different targets correspond to different corpus formats. It is important to select proper format for a target. We provide the examples (of different formats) for different targets in corpora folder. Changing encoders is even simplier than changing targets. Only thing we need to do is specify --encoder in pretrain.py.



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
                     [--bidirectional] [--subword_type {none,char}]
                     [--sub_vocab_path SUB_VOCAB_PATH]
                     [--subencoder_type {avg,lstm,gru,cnn}]
                     [--tokenizer {bert,char,word,space}]
                     [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                     [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                     [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using classifier.py：
```
python3 classifier.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                      --train_path datasets/book_review/train.txt --dev_path datasets/book_review/dev.txt --test_path datasets/book_review/test.txt \
                      --epochs_num 3 --batch_size 64 --encoder bert
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
                 [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                 [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                 [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                 [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using tagger.py：
```
python3 tagger.py --pretrained_model_path models/google_model.bin --vocab_path models/google_vocab.txt \
                  --train_path datasets/msra/train.txt --dev_path datasets/msra/dev.txt --test_path datasets/msra/test.txt \
                  --epochs_num 5 --batch_size 32 --encoder bert
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
The example of using cloze.py：
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
The example of using feature_extractor.py：
```
python3 feature_extractor.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_model.bin \
                             --vocab_path models/google_vocab.txt --output_path output.npy
```

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

### Performance
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
<tr align="center"><td> BERT+semi+BertTarget       <td> 88.1               <td> 95.6         <td> 97.0     <td> 94.3/92.6/93.4  <td> 85.1
<tr align="center"><td> BERT+semi+MlmTarget        <td> 87.9               <td> 95.5         <td> 97.2     <td>   <td> 85.1
</table>


It requires tremendous computional resources to fine-tune on large-scale datasets. For Ifeng, Chinanews, Dianping, JDbinary, and JDfull datasets, we provide their classification models (see Chinese model zoo). Classification models on large-scale datasets allow users to reproduce the results without training. Besides that, classification models could be used for improving other related tasks. More experimental results will come soon. 

Ifeng and Chinanews datasets contain news' titles and abstracts. In stage 2, we use title to predict abstract. 

<table>
<tr align="center"><th> Model/Dataset              <th> Ifeng     <th> Chinanews <th> Dianping <th> JDbinary <th> JDfull
<tr align="center"><td> pre-SOTA (Glyph & Glyce)   <td> 85.76     <td> 91.88     <td> 78.46    <td> 91.76    <td> 54.24 
<tr align="center"><td> BERT                       <td> 87.50     <td> 93.37     <td>          <td>          <td> 
<tr align="center"><td> BERT+semi+BertTarget       <td> 87.65     <td>           <td>          <td>          <td> 
</table>

We also provide the pre-trained models on different corpora, encoders, and targets (see Chinese model zoo). Selecting proper pre-training models is beneficial to the performance of downstream tasks.

<table>
<tr align="center"><th> Model/Dataset                     <th> MSRA-NER
<tr align="center"><td> Wikizh corpus (Google)            <td> 93.0/92.4/92.7
<tr align="center"><td> Renminribao corpus                <td> 94.3/94.1/94.2
</table>


<br/>

## Chinese_model_zoo
With the help of UER, we are pre-training models with different corpora, encoders, and targets.
<table>
<tr align="center"><th> pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh+BertEncoder+BertTarget <td> https://share.weiyun.com/5DJasRk <td> The training corpus Wiki_zh, trained by Google
<tr align="center"><td> RenMinRiBao+BertEncoder+BertTarget <td> https://share.weiyun.com/5HKnsxq <td> The training corpus is news data from People's Daily (1946-2017). It is suitable for datasets related with news, e.g. F1 is improved on MSRA-NER from 92.7 to 94.2 (compared with Google BERT)
<tr align="center"><td> Webqa2019+BertEncoder+BertTarget <td> https://share.weiyun.com/5HYbmBh <td> The training corpus is WebQA, which is suitable for datasets related with social media
<tr align="center"><td> IfengNews+BertEncoder+BertTarget <td> https://share.weiyun.com/5HVcUWO <td> The training corpus is news data from Ifeng website. We use news titles to predict news abstracts. Training steps: 100,000; Sequence length: 128. 
<tr align="center"><td> jdbinary+BertEncoder+ClsTarget <td> https://share.weiyun.com/596k2bu <td> The training corpus is review data from JD (jingdong). Classification target is used for pre-training. It is suitable for datasets related with shopping reviews, e.g. accuracy is improved on shopping datasets from 96.3 to 97.2 (compared with Google BERT)
<tr align="center"><td> Amazonreview+BertEncoder+ClsTarget <td> https://share.weiyun.com/5XuxtFA <td> The training corpus is review data from Amazon (including book reviews, movie reviews, and etc.). Classification target is used for pre-training. It is suitable for datasets related with reviews, e.g. accuracy is improved on Douban book review datasets from 87.6 to 88.7 (compared with Google BERT)
<tr align="center"><td> Wikizh+LstmEncoder+LmTarget <td>  <td> 
<tr align="center"><td> <td> <td> 
</table>

We release the classification models on 5 large-scale datasets, i.e. Ifeng, Chinanews, Dianping, JDbinary, and
JDfull. Users can use these models to reproduce results, or regard them as pre-training models for other datasets.
<table>
<tr align="center"><th> Datasets <th> Link  
<tr align="center"><td> Ifeng <td>  
<tr align="center"><td> Chinanews <td>
<tr align="center"><td> Dianping <td>
<tr align="center"><td> JDbinary <td>
<tr align="center"><td> JDfull <td>
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

