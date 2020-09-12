# UER-py
[![Build Status](https://travis-ci.org/dbiir/UER-py.svg?branch=master)](https://travis-ci.org/dbiir/UER-py)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)

<img src="logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks and has led to remarkable improvements. UER-py (Universal Encoder Representations) is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of different pre-training models (e.g. BERT, GPT, ELMO), and provides interfaces for users to further extend upon. With UER-py, we build a model zoo which contains pre-trained models based on different corpora, encoders, and targets. 

<br/>

#### We have a paper one can cite for UER-py:
```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```

<br/>

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Datasets](#datasets)
  * [Modelzoo](#modelzoo)
  * [Instructions](#instructions)
  * [Scripts](#scripts)
  * [Experiments](#experiments)


<br/>

## Features
UER-py has the following features:
- __Reproducibility.__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations.
- __Multi-GPU.__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity.__ UER-py is divided into multiple components: embedding, encoder, target, and downstream task fine-tuning. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.
- __Efficiency.__ UER-py refines its pre-processing, pre-training, and fine-tuning stages, which largely improves speed and needs less memory.
- __Model zoo.__ With the help of UER-py, we pre-trained models with different corpora, encoders, and targets. Proper selection of pre-trained models is important to the downstream task performances.
- __SOTA results.__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and has been used in winning solutions of many NLP competitions.


<br/>

## Requirements
* Python 3.6
* torch >= 1.0
* six
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need SentencePiece


<br/>

## Quickstart
We use BERT model and [Douban book review classification dataset](https://embedding.github.io/evaluation/) to demonstrate how to use UER-py. We firstly pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review classification dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

The format of the corpus for BERT is as follows：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained by book review classification dataset. We remove labels and split a review into two parts from the middle (See *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows:
```
label    text_a
1        instance1
0        instance2
1        instance3
```
Label and instance are separated by \t . The first row is a list of column names. The label ID should be an integer between (and including) 0 and n-1 for n-way classification.

We use Google's Chinese vocabulary file, which contains 21128 Chinese characters. The format of the vocabulary is as follows:
```
word-1
word-2
...
word-n
```

First of all, we preprocess the book review corpus. We need to specify the model's target in pre-processing stage (*--target*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download [Google's pre-trained Chinese model](https://share.weiyun.com/A1C49VPb), and put it in *models* folder. We load Google's pre-trained Chinese model and train it on book review corpus. We should explicitly specify model's encoder (*--encoder*) and target (*--target*). Suppose we have a machine with 8 GPUs.:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --encoder bert --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step. We could remove the suffix for ease of use.

Then we fine-tune pre-trained models on downstream classification dataset. We can use *google_zh_model.bin*:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --encoder bert
```
or use our [*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA), which is the output of *pretrain.py*：
```
python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --encoder bert
``` 
It turns out that the result of Google's model is 87.5; The result of *book_review_model.bin* is 88.2. It is also noticeable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

The default path of the fine-tuned classifier model is *./models/classifier_model.bin* . Then we do inference with the classifier model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 --encoder bert
```
*--test_path* specifies the path of the file to be predicted.
*--prediction_path* specifies the path of the file with prediction results.
We need to explicitly specify the number of labels by *--labels_num*. Douban book review is a two-way classification dataset.

We recommend to use *CUDA_VISIBLE_DEVICES* to specify which GPUs are visible (all GPUs are used in default) :
```
CUDA_VISIBLE_DEVICES=0 python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                                                 --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                 --epochs_num 3 --batch_size 32 --encoder bert
```
```
CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                                                 --test_path datasets/douban_book_review/test_nolabel.tsv \
                                                                 --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 --encoder bert
```

BERT consists of next sentence prediction (NSP) target. However, NSP target is not suitable for sentence-level reviews since we have to split a sentence into multiple parts. UER-py facilitates the use of different targets. Using masked language modeling (MLM) as target could be a properer choice for pre-training of reviews:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_mlm_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --encoder bert --target mlm

mv models/book_review_mlm_model.bin-5000 models/book_review_mlm_model.bin

CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier.py --pretrained_model_path models/book_review_mlm_model.bin --vocab_path models/google_zh_vocab.txt \
                                                   --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                   --epochs_num 3 --batch_size 32 --encoder bert
```
It turns out that the result of [*book_review_mlm_model.bin*](https://share.weiyun.com/V0XidqrV) is around 88.3.

BERT is slow. It could be great if we can speed up the model and still achieve competitive performance. To achieve this goal, we select a 2-layers LSTM encoder to substitute 12-layers Transformer encoder. We firstly download [pre-trained model](https://share.weiyun.com/5B671Ik) for 2-layers LSTM encoder. Then we fine-tune it on downstream classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/reviews_lstm_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3  --batch_size 64 --learning_rate 1e-3 --embedding word --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder lstm --pooling mean
```
We can achieve over 86 accuracy on testset, which is a competitive result. Using the same LSTM encoder without pre-training can only achieve around 81 accuracy.

UER-py also provides many other encoders and corresponding pre-trained models.
The example of pre-training and fine-tuning ELMo on Chnsenticorp dataset:
```
python3 preprocess.py --corpus_path corpora/chnsenticorp.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --seq_length 192 --target bilm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/mixed_corpus_elmo_model.bin \
                    --config_path models/birnn_config.json \
                    --output_model_path models/chnsenticorp_elmo_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --learning_rate 5e-4 \
                    --embedding word --encoder bilstm --target bilm

mv models/chnsenticorp_elmo_model.bin-5000 models/chnsenticorp_elmo_model.bin

python3 run_classifier.py --pretrained_model_path models/chnsenticorp_elmo_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/birnn_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --seq_length 192 --learning_rate 5e-4 \
                          --embedding word --encoder bilstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/birnn_config.json --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder bilstm --pooling mean
```
Users can download mixed_corpus_elmo_model.bin from [here](https://share.weiyun.com/5Qihztq).

The example of fine-tuning GatedCNN on Chnsenticorp dataset:
```
CUDA_VISIBLE_DEVICES=0 python3 run_classifier.py --pretrained_model_path models/wikizh_gatedcnn_model.bin --vocab_path models/google_zh_vocab.txt \
                                                 --config_path models/gatedcnn_9_config.json \
                                                 --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                                                 --epochs_num 5  --batch_size 64 --learning_rate 5e-5 \
                                                 --embedding word --encoder gatedcnn --pooling max

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gatedcnn_9_config.json \
                                          --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder gatedcnn --pooling max
```
Users can download wikizh_gatedcnn_model.bin from [here](https://share.weiyun.com/W2gmPPeA).

Besides classification, UER-py also provides scripts for other downstream tasks. We could use *run_ner.py* for named entity recognition:
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 --encoder bert
```
*--label2id_path* specifies the path of label2id file for named entity recognition.
The default path of the fine-tuned ner model is *./models/ner_model.bin* . Then we do inference with the ner model:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json --encoder bert
```

We could use *run_cmrc.py* for machine reading comprehension:
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --epochs_num 2 --batch_size 8 --seq_length 512 --encoder bert
```
We don't specify the *--test_path* because CMRC2018 dataset doesn't provide labels for testset. 
Then we do inference with the cmrc model:
```
python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path datasets/cmrc2018/test.json  \
                                    --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 --encoder bert
```

<br/>

## Datasets
This project includes a range of Chinese datasets: XNLI, LCQMC, MSRA-NER, ChnSentiCorp, and NLPCC-DBQA are from [Baidu ERNIE](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE); Douban book review is from [BNU](https://embedding.github.io/evaluation/); Online shopping review is annotated by ourself; THUCNews is from [text-classification-cnn-rnn project](https://github.com/gaussic/text-classification-cnn-rnn); Sina Weibo review is from [ChineseNlpCorpus project](https://github.com/SophonPlus/ChineseNlpCorpus); CMRC2018 is from [HIT CMRC2018 project](https://github.com/ymcui/cmrc2018) and C3 is from [CLUE](https://www.cluebenchmarks.com/). More Large-scale datasets can be found in [glyph's github project](https://github.com/zhangxiangxiao/glyph).

<table>
<tr align="center"><td> Dataset <td> Link
<tr align="center"><td> ChnSentiCorp <td> in the project
<tr align="center"><td> Douban book review <td> in the project
<tr align="center"><td> CMRC2018 <td> in the project
<tr align="center"><td> C3 <td> in the project
<tr align="center"><td> Online shopping review <td> https://share.weiyun.com/5xxYiig
<tr align="center"><td> LCQMC <td> https://share.weiyun.com/5Fmf2SZ
<tr align="center"><td> XNLI <td> https://share.weiyun.com/5hQUfx8
<tr align="center"><td> MSRA-NER <td> in the project
<tr align="center"><td> NLPCC-DBQA <td> https://share.weiyun.com/5HJMbih
<tr align="center"><td> Sina Weibo <td> https://share.weiyun.com/5lEsv0w
<tr align="center"><td> THUCNews <td> https://share.weiyun.com/5jPpgBr
</table>

<br/>

## Modelzoo
With the help of UER, we pre-trained models with different corpora, encoders, and targets. All pre-trained models can be loaded by UER directly. More pre-trained models will be released in the near future. Unless otherwise noted, Chinese pre-trained models use *models/google_zh_vocab.txt* as vocabulary, which is used in original BERT project. *models/bert_base_config.json* is used as configuration file in default. Commonly-used vocabulary and configuration files are included in *models* folder and users do not need to download them.

Pre-trained Chinese models from Google (in UER format):
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh+BertEncoder+BertTarget <td> https://share.weiyun.com/A1C49VPb <td> Google's pre-trained Chinese model from https://github.com/google-research/bert
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(base)+AlbertTarget <td> https://share.weiyun.com/UnKHNKRG <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_base_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(large)+AlbertTarget <td> https://share.weiyun.com/9tTUwALd <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_large_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xlarge)+AlbertTarget <td> https://share.weiyun.com/mUamRQFR <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xlarge_config.json
<tr align="center"><td> CLUECorpus+<br>AlbertEncoder(xxlarge)+AlbertTarget <td> https://share.weiyun.com/0i2lX62b <td> Google's pre-trained Chinese model from https://github.com/google-research/albert . <br>The configuration file is albert_xxlarge_config.json
</table>

Models pre-trained by UER:
<table>
<tr align="center"><th> Pre-trained model <th> Link <th> Description 
<tr align="center"><td> Wikizh(word-based)+BertEncoder+BertTarget <td> Model: https://share.weiyun.com/5s4HVMi Vocab: https://share.weiyun.com/5NWYbYn <td> Word-based BERT model pre-trained on Wikizh. Training steps: 500,000
<tr align="center"><td> RenMinRiBao+BertEncoder+BertTarget <td> https://share.weiyun.com/5JWVjSE <td> The training corpus is news data from People's Daily (1946-2017).
<tr align="center"><td> Webqa2019+BertEncoder+BertTarget <td> https://share.weiyun.com/5HYbmBh <td> The training corpus is WebQA, which is suitable for datasets related with social media, e.g. LCQMC and XNLI. Training steps: 500,000
<tr align="center"><td> Weibo+BertEncoder+BertTarget <td> https://share.weiyun.com/5ZDZi4A <td> The training corpus is Weibo.
<tr align="center"><td> Weibo+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/CFKyMkp3 <td> The training corpus is Weibo. The configuration file is bert_large_config.json
<tr align="center"><td> Reviews+BertEncoder+MlmTarget <td> https://share.weiyun.com/tBgaSx77 <td> The training corpus is reviews.
<tr align="center"><td> Reviews+BertEncoder(large)+MlmTarget <td> https://share.weiyun.com/hn7kp9bs <td> The training corpus is reviews. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(large)+BertTarget <td> https://share.weiyun.com/5G90sMJ <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_large_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(base)+BertTarget <td> https://share.weiyun.com/5QOzPqq <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_base_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(small)+BertTarget <td> https://share.weiyun.com/fhcUanfy <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_small_config.json
<tr align="center"><td> MixedCorpus+BertEncoder(tiny)+BertTarget <td> https://share.weiyun.com/yXx0lfUg <td> Pre-trained on mixed large Chinese corpus. The configuration file is bert_tiny_config.json
<tr align="center"><td> MixedCorpus+GptEncoder+LmTarget <td> https://share.weiyun.com/51nTP8V <td> Pre-trained on mixed large Chinese corpus. Training steps: 500,000 (with sequence lenght of 128) + 100,000 (with sequence length of 512)
<tr align="center"><td> Reviews+LstmEncoder+LmTarget <td> https://share.weiyun.com/57dZhqo  <td> The training corpus is amazon reviews + JDbinary reviews + dainping reviews (11.4M reviews in total). Language model target is used. It is suitable for datasets related with reviews. It achieves over 5 percent improvements on some review datasets compared with random initialization. Set hidden_size in models/rnn_config.json to 512 before using it. Training steps: 200,000; Sequence length: 128;
<tr align="center"><td> (MixedCorpus & Amazon reviews)+LstmEncoder+(LmTarget & ClsTarget) <td> https://share.weiyun.com/5B671Ik  <td> Firstly pre-trained on mixed large Chinese corpus with LM target. And then is pre-trained on Amazon reviews with lm target and cls target. It is suitable for datasets related with reviews. It can achieve comparable results with BERT on some review datasets. Training steps: 500,000 + 100,000; Sequence length: 128
<tr align="center"><td> IfengNews+BertEncoder+BertTarget <td> https://share.weiyun.com/5HVcUWO <td> The training corpus is news data from Ifeng website. We use news title to predict news abstract. Training steps: 100,000; Sequence length: 128
<tr align="center"><td> jdbinary+BertEncoder+ClsTarget <td> https://share.weiyun.com/596k2bu <td> The training corpus is review data from JD (jingdong). CLS target is used for pre-training. It is suitable for datasets related with shopping reviews. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> jdfull+BertEncoder+MlmTarget <td> https://share.weiyun.com/5L6EkUF <td> The training corpus is review data from JD (jingdong). MLM target is used for pre-training. Training steps: 50,000; Sequence length: 128
<tr align="center"><td> Amazonreview+BertEncoder+ClsTarget <td> https://share.weiyun.com/5XuxtFA <td> The training corpus is review data from Amazon (including book reviews, movie reviews, and etc.). Classification target is used for pre-training. It is suitable for datasets related with reviews, e.g. accuracy is improved on Douban book review datasets from 87.6 to 88.5 (compared with Google BERT). Training steps: 20,000; Sequence length: 128
<tr align="center"><td> XNLI+BertEncoder+ClsTarget <td> https://share.weiyun.com/5oXPugA <td> Infersent with BertEncoder
</table>
MixedCorpus contains baidubaike, Wikizh, WebQA, RenMinRiBao, literature, and reviews.

<br/>

## Instructions
### UER-py's framework
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/: contains encoders such as RNN, CNN, Attention, CNN-RNN, BERT
    |    |--targets/: contains targets such as language modeling, masked language modeling, sentence prediction
    |    |--layers/: contains frequently-used NN layers, such as embedding layer, normalization layer
    |    |--models/: contains model.py, which combines embedding, encoder, and target modules
    |    |--utils/: contains frequently-used utilities
    |    |--model_builder.py
    |    |--model_loader.py
    |    |--model_saver.py
    |    |--trainer.py
    |
    |--corpora/: contains corpora for pre-training
    |--datasets/: contains downstream tasks
    |--models/: contains pre-trained models, vocabularies, and config files
    |--scripts/: contains some useful scripts for pre-training models
    |--inference/：contains inference scripts for downstream tasks
    |
    |--preprocess.py
    |--pretrain.py
    |--run_classifier.py
    |--run_cmrc.py
    |--run_ner.py
    |--run_dbqa.py
    |--run_c3.py
    |--run_mt_classifier.py
    |--README.md
```

The code is well-organized. Users can use and extend upon it with little efforts.

### Preprocess the data
```
usage: preprocess.py [-h] --corpus_path CORPUS_PATH [--vocab_path VOCAB_PATH]
                     [--spm_model_path SPM_MODEL_PATH]
                     [--dataset_path DATASET_PATH]
                     [--tokenizer {bert,char,space}]
                     [--processes_num PROCESSES_NUM]
                     [--target {bert,lm,cls,mlm,bilm,albert}]
                     [--docs_buffer_size DOCS_BUFFER_SIZE]
                     [--seq_length SEQ_LENGTH] [--dup_factor DUP_FACTOR]
                     [--short_seq_prob SHORT_SEQ_PROB] [--full_sentences]
                     [--seed SEED] [--dynamic_masking] [--span_masking]
                     [--span_geo_prob SPAN_GEO_PROB]
                     [--span_max_length SPAN_MAX_LENGTH]
```
The example of pre-processing on a single machine is as follows：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt\
                      --processes_num 8 --target bert
```
If multiple machines are available, users can execute preprocess.py on one machine and copy the dataset.pt to other machines. 

We need to specify the model's target in pre-processing stage since different targets require different data formats. Currently, UER-py consists of the following target modules:
- lm_target.py: language model
- mlm_target.py: masked language model (cloze test)
- cls_target.py: classification
- bilm_target.py: bi-directional language model
- bert_target.py: masked language model + next sentence prediction
- albert_target.py: masked language model + sentence order prediction

*--preprocesses_num n* denotes that n processes are used for pre-processing. More processes can speed up the preprocess but lead to more memory consumption.

*--dynamic_masking* denotes that the words are masked during the pre-train stage, which is used in RoBERTa. 
*--full_sentences* allows a sample to include contents from multiple documents, which is used in RoBERTa.
*--span_masking* denotes that masking consecutive words in mlm target, which is used in SpanBERT. If dynamic masking is used, we should specify *--span_masking* in pre-training stage, otherwise we should specify *--span_masking* in pre-process stage.
*--docs_buffer_size* specifies the buffer size in the memory in pre-processing stage.

### Pretrain the model
```
usage: pretrain.py [-h] [--dataset_path DATASET_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   --output_model_path OUTPUT_MODEL_PATH
                   [--config_path CONFIG_PATH] [--total_steps TOTAL_STEPS]
                   [--save_checkpoint_steps SAVE_CHECKPOINT_STEPS]
                   [--report_steps REPORT_STEPS]
                   [--accumulation_steps ACCUMULATION_STEPS]
                   [--batch_size BATCH_SIZE]
                   [--instances_buffer_size INSTANCES_BUFFER_SIZE]
                   [--dropout DROPOUT] [--seed SEED] [--embedding {bert,word}]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                   [--bidirectional] [--target {bert,lm,cls,mlm,bilm}]
                   [--tie_weights] [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--span_masking]
                   [--span_geo_prob SPAN_GEO_PROB]
                   [--span_max_length SPAN_MAX_LENGTH]
                   [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                   [--beta1 BETA1] [--beta2 BETA2] [--fp16]
                   [--fp16_opt_level {O0,O1,O2,O3}] [--world_size WORLD_SIZE]
                   [--gpu_ranks GPU_RANKS [GPU_RANKS ...]]
                   [--master_ip MASTER_IP] [--backend {nccl,gloo}]
```

*--instances_buffer_size* specifies the buffer size in the memory in pre-training stage.
*--tie_weights* denotes the word embedding and softmax weights are tied.
It is recommended to explicitly specify model's encoder and target. UER-py consists of the following encoder modules:
- rnn_encoder.py: contains (bi-)LSTM and (bi-)GRU
- birnn_encoder.py: contains bi-LSTM and bi-GRU (different from rnn_encoder.py with --bidirectional, see [the issue](https://github.com/pytorch/pytorch/issues/4930) for more details)
- cnn_encoder.py: contains CNN and gatedCNN
- gpt_encoder.py: contains GPT encoder
- bert_encoder.py: contains BERT encoder

The target should be coincident with the target in pre-processing stage. Users can try different combinations of encoders and targets by *--encoder* and *--target*.
*--config_path* denotes the path of the configuration file, which specifies the hyper-parameters of the pre-training model. We have put the commonly-used configuration files in *models* folder. Users should choose the proper one according to encoder they use.

There are two strategies for pre-training: 1）random initialization 2）loading a pre-trained model.
#### Random initialization
The example of pre-training on CPU：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert
```
The example of pre-training on single GPU (the id of GPU is 3)：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin --encoder bert --target bert --gpu_ranks 3
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --output_model_path models/output_model.bin --encoder bert --target bert --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 
```
*--world_size* specifies the number of processes (and GPUs) used for pre-training.
*--gpu_ranks* specifies the ID for each process and GPU.
Users could use CUDA_VISIBLE_DEVICES if they want to use part of GPUs:
```
CUDA_VISIBLE_DEVICES=1,2,3,5 python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                                                 --output_model_path models/output_model.bin --encoder bert --target bert --world_size 4 --gpu_ranks 0 1 2 3
```

The example of pre-training on two machines, each has 8 GPUs (16 GPUs in total): 
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 \
                             --total_steps 100000 --save_checkpoint_steps 10000 --report_steps 100 \
                             --master_ip tcp://9.73.138.133:12345
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --output_model_path models/output_model.bin --encoder bert --target bert --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 \
                             --total_steps 100000 \
                             --master_ip tcp://9.73.138.133:12345          
```
The IP of Node-0 is 9.73.138.133 .
*--total_steps* specifies the training steps.
*--save_checkpoint_steps* specifies how often to save the model checkpoint. We don't need to specify the *--save_checkpoint_steps* in Node-1 since only master node saves the pre-trained model.
*--report_steps* specifies how often to report the pre-training information. We don't need to specify the *--report_steps* in Node-1 since the information only appears in the master node.
Notice that when specifying *--master_ip* one can not select the port that occupied by other programs

#### Load a pre-trained model
We recommend to load a pre-trained model. We can specify the pre-trained model by *--pretrained_model_path* .
The example of pre-training on CPU and single GPU:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin --output_model_path models/output_model.bin \
                    --encoder bert --target bert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin --output_model_path models/output_model.bin \
                    --gpu_ranks 3 --encoder bert --target bert
```
The example of pre-training on a single machine with 8 GPUs：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target bert 
```
The example of pre-training on two machines, each has 8 GPUs (16 GPUs in total):
```
Node-0 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin --output_model_path models/output_model.bin \
                             --encoder bert --target bert --world_size 16 --gpu_ranks 0 1 2 3 4 5 6 7 --master_ip tcp://9.73.138.133:12345
Node-1 : python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                             --pretrained_model_path models/google_zh_model.bin --output_model_path models/output_model.bin \
                             --encoder bert --target bert --world_size 16 --gpu_ranks 8 9 10 11 12 13 14 15 --master_ip tcp://9.73.138.133:12345
```
The example of pre-training on three machines, each has 8 GPUs (24 GPUs in total):
```
Node-0: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 0 1 2 3 4 5 6 7 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
Node-1: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 8 9 10 11 12 13 14 15 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
Node-2: python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                            --pretrained_model_path models/google_zh_model.bin \
                            --output_model_path models/output_model.bin --world_size 24 --gpu_ranks 16 17 18 19 20 21 22 23 \
                            --master_ip tcp://9.73.138.133:12345 --encoder bert --target bert
```

#### Pre-training model size
In general, large model can achieve better results but lead to more resource consumption. We can specify the pre-trained model size by *--config_path*. Commonly-used configuration files are included in *models* folder. For example, we provide 4 configuration files for BERT model. They are bert_large_config.json, bert_base_config.json, bert_small_config.json, bert_tiny_config.json. We provide different pre-trained models according to different configuration files. See Chinese model zoo for more details.
The example of doing incremental pre-training upon BERT-large model:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/mixed_corpus_bert_large_model.bin --config_path models/bert_large_config.json \
                    --output_model_path models/output_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target bert
```

### Try pre-training models with different targets and encoders
UER-py allows users to combine different components (e.g. embeddings, encoders, and targets). Here are some examples of trying different combinations.

#### RoBERTa
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 12 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target mlm
```
RoBERTa uses dynamic masking, mlm target, and allows a sample to contain contents from multiple documents.
We don't recommend to use *--full_sentences* when the document is short (e.g. reviews).
Notice that RoBERTa removes NSP target. The corpus for RoBERTa stores one document per line, which is different from corpus used by BERT.
The example of doing incremental pre-training upon existing BERT model:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 12 \
                      --dynamic_masking --target mlm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target mlm
```
Nsp target is not suitable when document is short. We could replace BERT target with MLM target when handling corpus such as reviews.

#### ALBERT
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 12 --target albert
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/albert_base_config.json \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder bert --target albert \
                    --factorized_embedding_parameterization --parameter_sharing
```
The corpus format of ALBERT is the identical with BERT.
*--target albert* denotes that using ALBERT target, which consists of mlm and sop targets.
*--factorized_embedding_parameterization* denotes that using factorized embedding parameterization to untie the embedding size from the hidden layer size.
*--parameter_sharing* denotes that sharing all parameters (including feed-forward and attention parameters) across layers.
we provide 4 configuration files for ALBERT model in *models* folder, albert_base_config.json, albert_large_config.json, albert_xlarge_config.json, albert_xxlarge_config.json.

#### GPT
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 12 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --config_path models/bert_base_config.json --learning_rate 1e-4 \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --encoder gpt --target lm
```
The corpus format of GPT is the identical with RoBERTa. We can use GPT through *--encoder gpt* and *--target lm*.
GPT can use the configuration file of BERT.

#### More combinations
The example of using LSTM encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \ 
                    --embedding word --encoder lstm --target lm --learning_rate 1e-3 --config_path models/rnn_config.json
```
*--embedding word* denotes using traditional word embedding. LSTM does not require position embedding.
We use the *models/rnn_config.json* as configuration file.

The example of using GatedCNN encoder and LM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target lm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --encoder gatedcnn --target lm --learning_rate 1e-4 --config_path models/gcnn_9_config.json
```

The example of using Bi-LSTM encoder and BiLM target for pre-training:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt --processes_num 8 --target bilm
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt  --output_model_path models/output_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 20000 --save_checkpoint_steps 5000 \
                    --embedding word --encoder bilstm --target bilm --learning_rate 5e-4 --config_path models/birnn_config.json
```

### Fine-tune on downstream tasks
Currently, UER-py consists of the following downstream tasks: text classification, pair classification, document-based question answering, sequence labeling, and machine reading comprehension. The encoder used for downstream task should be coincident with the pre-trained model.

#### Classification
run_classifier.py adds two feedforward layers upon encoder layer.
```
usage: run_classifier.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                         [--output_model_path OUTPUT_MODEL_PATH]
                         [--vocab_path VOCAB_PATH]
                         [--spm_model_path SPM_MODEL_PATH] --train_path
                         TRAIN_PATH --dev_path DEV_PATH
                         [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                         [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                         [--embedding {bert,word}]
                         [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                         [--bidirectional] [--pooling {mean,max,first,last}]
                         [--factorized_embedding_parameterization]
                         [--parameter_sharing] [--tokenizer {bert,char,space}]
                         [--soft_targets] [--soft_alpha SOFT_ALPHA]
                         [--learning_rate LEARNING_RATE] [--warmup WARMUP]
                         [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                         [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                         [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_classifier.py*：
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using *run_classifier.py* for pair classification:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/lcqmc/train.tsv --dev_path datasets/lcqmc/dev.tsv --test_path datasets/lcqmc/test.tsv \
                          --epochs_num 3 --batch_size 64 --encoder bert
```
The example of using *inference/run_classifier_infer.py* to do inference:
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --seq_length 128 --output_logits --output_prob --encoder bert
```
*--labels_num* specifies the number of labels.
*--output_logits* denotes the predicted logits are outputted，whose column name is logits.
*--output_prob* denotes the predicted probability are outputted，whose column name is prob.
*--seq_length* specifies the sequence length, which should be the same with training.

The example of using ALBERT for classification:
```
python3 run_classifier.py --pretrained_model_path models/albert_base_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/albert_base_config.json \
                          --train_path datasets/douban_book_review/train.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \ 
                          --learning_rate 4e-5
                          --epochs_num 5 --batch_size 32 --parameter_sharing --factorized_embedding_parameterization --encoder bert
```
The performance of ALBERT is sensitive to hyper-parameter settings.
The example of doing inference for ALBERT:
```
python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert_base_config.json \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --parameter_sharing --factorized_embedding_parameterization --encoder bert
```

UER-py supports multi-task learning. Embedding and encoder layers are shared by different tasks.
The example of training two sentiment analysis datasets:
```
python3 run_mt_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                             --dataset_path_list datasets/douban_book_review/ datasets/chnsenticorp/ \
                             --epochs_num 1 --batch_size 64 --encoder bert
```
--dataset_path_list specifies folder path list of different tasks. Each folder should contains train set *train.tsv* and development set *dev.tsv* .


UER-py supports distillation for classification tasks. 
First of all, we fine-tune upon a Chinese BERT-large model (provided in Chinese model zoo):
```
python3 run_classifier.py --pretrained_model_path models/bert_large_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/bert_large_config.json \
                          --output_model_path models/teacher_classifier_model.bin \
                          --train_path datasets/douban_book_review/train.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --encoder bert
```
Then we use the teacher model to do inference. The pesudo labels and logits are generated:
```
python3 inference/run_classifier_infer.py --load_model_path models/teacher_classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/bert_large_config.json --test_path text.tsv \
                                          --prediction_path label_logits.tsv --labels_num 2 --output_logits --encoder bert
```
The input file *text.tsv* contains text to be predicted (see *datasets/douban_book_review/test_nolabel.tsv*). *text.tsv* could be trainset and related external data. Larger transfer set often leads to better performance.
The output file *label_logits.tsv* contains label column and logits column. Then we obtain *text_label_logits.tsv* by combining *text.tsv* and *label_logits.tsv* . *text_label_logits.tsv* contains text_a column (text_a column and text_b column for pair classification), label column (hard label), and logits column (soft label).
Student model is a 3-layers BERT-tiny model. The pre-trained model is provided in Chinese model zoo.
Then the student model learns the outputs (hard and soft labels) of the teacher model:
```
python3 run_classifier.py --pretrained_model_path bert_tiny_model.bin --vocab_path models/google_zh_vocab.txt \
                          --config_path models/bert_tiny_config.json \
                          --train_path text_label_logits.tsv \
                          --dev_path datasets/douban_book_review/dev.tsv \
                          --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 64 --soft_targets --soft_alpha 0.5 --encoder bert
```
*--soft_targets* denotes that the model uses logits (soft label) for training. Mean-squared-error(MSE) is used as loss function.
*--soft_alpha* specifies the weight of the soft label loss. The loss function is weighted average of cross-entropy loss (for hard label) and mean-squared-error loss (for soft label).

#### Sequence labeling
run_ner.py adds two feedforward layers upon encoder layer.
```
usage: run_ner.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                  [--output_model_path OUTPUT_MODEL_PATH]
                  [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                  --train_path TRAIN_PATH --dev_path DEV_PATH
                  [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                  --label2id_path LABEL2ID_PATH [--batch_size BATCH_SIZE]
                  [--seq_length SEQ_LENGTH] [--embedding {bert,word}]
                  [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                  [--bidirectional] [--factorized_embedding_parameterization]
                  [--parameter_sharing] [--learning_rate LEARNING_RATE]
                  [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                  [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                  [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_ner.py*:
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 --encoder bert
```
The example of doing inference:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/msra_ner/test_nolabel.tsv \
                                          --prediction_path datasets/msra_ner/prediction.tsv \
                                          --label2id_path datasets/msra_ner/label2id.json --encoder bert
```
The example of using ALBERT for NER:
```
python3 run_ner.py --pretrained_model_path models/albert_base_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --config_path models/albert_base_config.json \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 \
                   --parameter_sharing --factorized_embedding_parameterization --encoder bert
```
The example of doing inference for ALBERT:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/albert_base_config.json \
                                          --test_path datasets/msra_ner/test_nolabel.tsv \
                                          --prediction_path datasets/msra_ner/prediction.tsv \
                                          --label2id_path datasets/msra_ner/label2id.json \
                                          --parameter_sharing --factorized_embedding_parameterization --encoder bert
```

#### Machine reading comprehension
run_cmrc.py adds two feedforward layers upon encoder layer.
```
usage: run_cmrc.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                   [--output_model_path OUTPUT_MODEL_PATH]
                   [--vocab_path VOCAB_PATH] [--spm_model_path SPM_MODEL_PATH]
                   --train_path TRAIN_PATH --dev_path DEV_PATH
                   [--test_path TEST_PATH] [--config_path CONFIG_PATH]
                   [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                   [--doc_stride DOC_STRIDE] [--embedding {bert,word}]
                   [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,synt,rcnn,crnn,gpt,bilstm}]
                   [--bidirectional] [--factorized_embedding_parameterization]
                   [--parameter_sharing] [--learning_rate LEARNING_RATE]
                   [--warmup WARMUP] [--fp16] [--fp16_opt_level {O0,O1,O2,O3}]
                   [--dropout DROPOUT] [--epochs_num EPOCHS_NUM]
                   [--report_steps REPORT_STEPS] [--seed SEED]
```
The example of using *run_cmrc.py*:
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt 
                   --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json 
                   --epochs_num 2 --batch_size 8 --seq_length 512 --encoder bert
```
The train.json and dev.json are of squad-style. Trainset and devset are available [here](https://github.com/ymcui/cmrc2018). --test_path option is not specified since testset is not publicly available.

The example of doing inference:
```
python3  inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json --encoder bert
```
The example of using ALBERT for CMRC:
```
python3 run_cmrc.py --pretrained_model_path models/albert_base_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --config_path models/bert_large_config.json \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --epochs_num 2 --batch_size 8 --seq_length 512 \
                    --parameter_sharing --factorized_embedding_parameterization --encoder bert
```
The example of doing inference for ALBERT:
```
python3  inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                     --config_path models/albert_base_config.json \
                                     --test_path datasets/cmrc2018/test.json \
                                     --prediction_path datasets/cmrc2018/prediction.json \
                                     --parameter_sharing --factorized_embedding_parameterization --encoder bert
```

#### ChineseGLUE
The BERT-large model trained upon mixed large corpus achieves SOTA results on [ChineseGLUE](http://106.13.187.75:8003/leaderBorder). The detailed scripts are listed as follows:
```
CUDA_VISIBLE_DEVICES=0 /dockerdata/anaconda3/bin/python run_classifier.py --pretrained_model_path models/mixed_large_24_model.bin --vocab_path models/google_zh_vocab.txt --train_path datasets/tnews/train.tsv --dev_path datasets/tnews/dev.tsv --test_path datasets/tnews/test.tsv --epochs_num 3 --batch_size 32 --encoder bert --config_path models/bert_large_config.json --output_model_path tnews_classifier_16_64_1gpu_model.bin --seq_length 64
```

```
CUDA_VISIBLE_DEVICES=0,1 /dockerdata/anaconda3/bin/python run_classifier.py --pretrained_model_path models/mixed_large_24_model.bin --vocab_path models/google_zh_vocab.txt --train_path datasets/inews/train.tsv --dev_path datasets/inews/dev.tsv --test_path datasets/inews/test.tsv --epochs_num 3 --batch_size 16 --seq_length 512 --encoder bert --config_path models/bert_large_config.json --output_model_path inews_classifier_16_512_1gpu_model.bin
```

```
CUDA_VISIBLE_DEVICES=0 /dockerdata/anaconda3/bin/python run_classifier.py --pretrained_model_path models/mixed_large_24_model.bin --vocab_path models/google_zh_vocab.txt --train_path datasets/lcqmc/train.tsv --dev_path datasets/lcqmc/dev.tsv --test_path datasets/lcqmc/test.tsv --epochs_num 3 --batch_size 32 --encoder bert --config_path models/bert_large_config.json --output_model_path lcqmc_classifier_32_1gpu_model.bin
```

```
CUDA_VISIBLE_DEVICES=0 /dockerdata/anaconda3/bin/python run_classifier.py --pretrained_model_path models/mixed_large_24_model.bin --vocab_path models/google_zh_vocab.txt --train_path datasets/xnli/train.tsv --dev_path datasets/xnli/dev.tsv --test_path datasets/xnli/test.tsv --epochs_num 3 --batch_size 32 --encoder bert --config_path models/bert_large_config.json --output_model_path xnli_classifier_32_1gpu_model.bin
```

<br/>

## Scripts

UER-py provides abundant tool scripts for pre-training models.
This section firstly summarizes tool scripts and their functions, and then provides using examples of some scripts.

### Scripts overview

<table>
<tr align="center"><th> Script <th> Function description
<tr align="center"><td> average_model.py <td> Take the average of pre-trained models. A frequently-used ensemble strategy for deep learning models 
<tr align="center"><td> build_vocab.py <td> Build vocabulary (multi-processing supported)
<tr align="center"><td> check_model.py <td> Check the model (single GPU or multiple GPUs)

<tr align="center"><td> cloze_test.py <td> Randomly mask a word and predict it, top n words are returned
<tr align="center"><td> convert_bert_from_uer_to_google.py <td> convert the BERT of UER format to Google format (TF)
<tr align="center"><td> convert_bert_from_uer_to_huggingface.py <td> convert the BERT of UER format to Huggingface format (PyTorch)
<tr align="center"><td> convert_bert_from_google_to_uer.py <td> convert the BERT of Google format (TF) to UER format
<tr align="center"><td> convert_bert_from_huggingface_to_uer.py <td> convert the BERT of Huggingface format (PyTorch) to UER format
<tr align="center"><td> diff_vocab.py <td> Compare two vocabularies
<tr align="center"><td> dynamic_vocab_adapter.py <td> Change the pre-trained model according to the vocabulary. It can save memory in fine-tuning stage since task-specific vocabulary is much smaller than general-domain vocabulary 
<tr align="center"><td> extract_embedding.py <td> extract the embedding of the pre-trained model
<tr align="center"><td> extract_feature.py <td> extract the hidden states of the last of the pre-trained model
<tr align="center"><td> topn_words_indep.py <td> Finding nearest neighbours with context-independent word embedding
<tr align="center"><td> topn_words_dep.py <td> Finding nearest neighbours with context-dependent word embedding
</table>


### Cloze test
cloze_test.py predicts masked words. Top n words are returned.
```
usage: cloze_test.py [-h] [--pretrained_model_path PRETRAINED_MODEL_PATH]
                [--vocab_path VOCAB_PATH] [--input_path INPUT_PATH]
                [--output_path OUTPUT_PATH] [--config_path CONFIG_PATH]
                [--batch_size BATCH_SIZE] [--seq_length SEQ_LENGTH]
                [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                [--bidirectional] [--target {bert,lm,cls,mlm,nsp,s2s}]
                [--subword_type {none,char}] [--sub_vocab_path SUB_VOCAB_PATH]
                [--subencoder_type {avg,lstm,gru,cnn}]
                [--tokenizer {bert,char,word,space}] [--topn TOPN]
```
The example of using cloze_test.py：
```
python3 scripts/cloze_test.py --input_path datasets/cloze_input.txt --pretrained_model_path models/google_zh_model.bin \
                              --vocab_path models/google_zh_vocab.txt --output_path output.txt

```

### Feature extractor
extract_feature.py extracts hidden states of the last encoder layer.
```
usage: extract_feature.py [-h] --input_path INPUT_PATH --pretrained_model_path
                          PRETRAINED_MODEL_PATH --vocab_path VOCAB_PATH
                          --output_path OUTPUT_PATH [--seq_length SEQ_LENGTH]
                          [--batch_size BATCH_SIZE]
                          [--config_path CONFIG_PATH]
                          [--embedding {bert,word}]
                          [--encoder {bert,lstm,gru,cnn,gatedcnn,attn,rcnn,crnn,gpt}]
                          [--bidirectional] [--subword_type {none,char}]
                          [--sub_vocab_path SUB_VOCAB_PATH]
                          [--subencoder {avg,lstm,gru,cnn}]
                          [--sub_layers_num SUB_LAYERS_NUM]
                          [--tokenizer {bert,char,space}]
```
The example of using extract_feature.py：
```
python3 scripts/extract_feature.py --input_path datasets/cloze_input.txt --vocab_path models/google_zh_vocab.txt \
                                   --pretrained_model_path models/google_zh_model.bin --output_path feature_output.pt
```

### Finding nearest neighbours
Pre-trained models can learn high-quality word embeddings. Traditional word embeddings such as word2vec and GloVe assign each word a fixed vector (context-independent word embedding). However, polysemy is a pervasive phenomenon in human language, and the meanings of a polysemous word depend on the context. To this end, we use a the hidden state in pre-trained models to represent a word. It is noticeable that Google BERT is a character-based model. To obtain real word embedding (not character embedding), Users should download our [word-based BERT model](https://share.weiyun.com/5s4HVMi) and [vocabulary](https://share.weiyun.com/5NWYbYn).
The example of using scripts/topn_words_indep.py to find nearest neighbours for context-independent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_indep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --cand_vocab_path models/google_zh_vocab.txt --target_words_path target_words.txt
python3 scripts/topn_words_indep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                    --cand_vocab_path models/wiki_word_vocab.txt --target_words_path target_words.txt
```
Context-independent word embedding is obtained by model's embedding layer.
The format of the target_words.txt is as follows:
```
word-1
word-2
...
word-n
```
The example of using scripts/topn_words_dep.py to find nearest neighbours for context-dependent word embedding (character-based and word-based models)：
```
python3 scripts/topn_words_dep.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                                  --cand_vocab_path models/google_zh_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer bert
python3 scripts/topn_words_dep.py --pretrained_model_path models/bert_wiki_word_model.bin --vocab_path models/wiki_word_vocab.txt \
                                  --cand_vocab_path models/wiki_word_vocab.txt --sent_path target_words_with_sentences.txt --config_path models/bert_base_config.json \
                                  --batch_size 256 --seq_length 32 --tokenizer space
```
We substitute the target word with other words in the vocabulary and feed the sentences into the pretrained model. Hidden state is used as the context-dependent embedding of a word. Users should do word segmentation manually and use space tokenizer if word-based model is used. The format of 
target_words_with_sentences.txt is as follows:
```
sent1 word1
sent1 word1
...
sentn wordn
```
Sentence and word are splitted by \t. 

### Text generator
We could use *generate.py* to generate text. Given a few words or sentences, *generate.py* can continue writing. The example of using *generate.py*:
```
python3 scripts/generate.py --pretrained_model_path models/gpt_model.bin --vocab_path models/google_zh_vocab.txt 
                            --input_path story_beginning.txt --output_path story_full.txt --config_path models/bert_base_config.json 
                            --encoder gpt --target lm --seq_length 128  
```
where *story_beginning* contains the beginning of a text. One can use any models pre-trained with LM target, such as [GPT trained on mixed large corpus](https://share.weiyun.com/51nTP8V). By now we only provide a vanilla version of generator. More mechanisms will be added for better performance and efficiency.


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


Target sentence: 苹果削减了台式Mac产品线上的众多产品。
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

## Contact information
For communication related to this project, please contact Zhe Zhao (helloworld@ruc.edu.cn; nlpzhezhao@tencent.com) or Xin Zhao (zhaoxinruc@ruc.edu.cn).

This work is instructed by my enterprise mentors __Qi Ju__, __Haotang Deng__ and school mentors __Tao Liu__, __Xiaoyong Du__.

I also got a lot of help from my Tencent colleagues Yudong Li, Hui Chen, Jinbin Zhang, Zhiruo Wang, Weijie Liu, Peng Zhou, Haixiao Liu, and Weijian Wu. 


