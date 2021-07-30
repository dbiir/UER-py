[**English**](https://github.com/dbiir/UER-py) | [**中文**](https://github.com/dbiir/UER-py/blob/master/README_ZH.md)

[![Build Status](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml/badge.svg)](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1909.05658-<color>.svg)](https://arxiv.org/abs/1909.05658)

<img src="logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks. UER-py (Universal Encoder Representations) is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of existing pre-training models, and provides interfaces for users to further extend upon. With UER-py, we build a model zoo which contains pre-trained models based on different corpora, encoders, and targets. **See the Wiki for [Full Documentation](https://github.com/dbiir/UER-py/wiki)**.


<br/>

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Datasets](#datasets)
  * [Modelzoo](#modelzoo)
  * [Instructions](#instructions)
  * [Competition solutions](#competition-solutions)
  * [Citation](#citation)
  * [Contact information](#contact-information)


<br/>

## Features
UER-py has the following features:
- __Reproducibility__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations such as BERT, GPT-2, ELMo, and T5.
- __Multi-GPU__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity__ UER-py is divided into multiple components: embedding, encoder, and target. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules to construct pre-training models with as few restrictions as possible.
- __Efficiency__ UER-py refines its pre-processing, pre-training, fine-tuning, and inference stages, which largely improves speed while requires less memory and disk space.
- __Model zoo__ With the help of UER-py, we pre-trained models with different corpora, encoders, and targets. Proper selection of pre-trained models is important to the performances of downstream tasks.
- __SOTA results__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and provides winning solutions of many NLP competitions.
- __Abundant functions__ UER-py provides abundant functions related with pre-training, such as feature extractor and mixed precision training.


<br/>

## Requirements
* Python >= 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need [SentencePiece](https://github.com/google/sentencepiece)
* For developing a stacking model you will need LightGBM and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
* For the pre-training with whole word masking you will need word segmentation tool such as [jieba](https://github.com/fxsjy/jieba)
* For the use of CRF in sequence labeling downstream task you will need [pytorch-crf](https://github.com/kmkurn/pytorch-crf)


<br/>

## Quickstart
This section uses several commonly-used examples to demonstrate how to use UER-py. More details are discussed in Instructions section. We firstly use BERT model on [Douban book review classification dataset](https://embedding.github.io/evaluation/). We pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review classification dataset, and vocabulary. All files are encoded in UTF-8 and included in this project.

The format of the corpus for BERT is as follows (one sentence per line and documents are delimited by empty lines)：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained from book review classification dataset. We remove labels and split a review into two parts from the middle (see *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows:
```
label    text_a
1        instance1
0        instance2
1        instance3
```
Label and instance are separated by \t . The first row is a list of column names. The label ID should be an integer between (and including) 0 and n-1 for n-way classification.

We use Google's Chinese vocabulary file *models/google_zh_vocab.txt*, which contains 21128 Chinese characters.

We firstly pre-process the book review corpus. We need to specify the model's target in pre-processing stage (*--target*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bert
```
Notice that *six>=1.12.0* is required.


Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). BERT tokenizer is used in default (*--tokenizer bert*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download Google's pre-trained Chinese BERT model [*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb) (in UER format and the original model is from [here](https://github.com/google-research/bert)), and put it in *models* folder. We load the pre-trained Chinese BERT model and further pre-train it on book review corpus. Pre-training model is composed of embedding, encoder, and target layers. To build a pre-training model, we should explicitly specify model's embedding (*--embedding*), encoder (*--encoder* and *--mask*), and target (*--target*). Suppose we have a machine with 8 GPUs:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* specifies the attention mask types. BERT uses bidirectional LM. The word token can attend to all tokens and therefore we use *fully_visible* mask type. The embedding layer of BERT is the sum of word (token), position, and segment embeddings and therefore *--embedding word_pos_seg* is specified. By default, *models/bert/base_config.json* is used as configuration file, which specifies the model hyper-parameters. 
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step (*--total_steps*). We could remove the suffix for ease of use.


Then we fine-tune the pre-trained model on downstream classification dataset. We use [*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA), which is the output of *pretrain.py*:
```
python3 finetune/run_classifier.py --pretrained_model_path models/book_review_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 32 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
It is noticeable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

The default path of the fine-tuned classifier model is *models/finetuned_model.bin* . Then we do inference with the fine-tuned model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* specifies the path of the file to be predicted. The file should contain text_a column. <br>
*--prediction_path* specifies the path of the file with prediction results. <br>
We need to explicitly specify the number of labels by *--labels_num*. Douban book review is a two-way classification dataset.

<br>

The above content provides basic ways of using UER-py to pre-process, pre-train, fine-tune, and do inference. More use cases can be found in complete :arrow_right: [__quickstart__](https://github.com/dbiir/UER-py/wiki/Quickstart) :arrow_left: . The complete quickstart contains comprehensive use cases, covering most of the pre-training related application scenarios. It is recommended that users read the complete quickstart in order to use the project reasonably.


<br/>

## Datasets
We collected a range of :arrow_right: [__downstream datasets__](https://github.com/dbiir/UER-py/wiki/Datasets) :arrow_left: and converted them into the format that UER can load directly.

<br/>

## Modelzoo
With the help of UER, we pre-trained models with different corpora, encoders, and targets. Detailed introduction of pre-trained models and their download links can be found in :arrow_right: [__modelzoo__](https://github.com/dbiir/UER-py/wiki/Modelzoo) :arrow_left: . All pre-trained models can be loaded by UER directly. More pre-trained models will be released in the future.

<br/>

## Instructions
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/ # contains encoders such as RNN, CNN, Transformer
    |    |--targets/ # contains targets such as language modeling, masked language modeling
    |    |--layers/ # contains frequently-used NN layers, such as embedding layer, normalization layer
    |    |--models/ # contains model.py, which combines embedding, encoder, and target modules
    |    |--utils/ # contains frequently-used utilities
    |    |--model_builder.py
    |    |--model_loader.py
    |    |--model_saver.py
    |    |--trainer.py
    |
    |--corpora/ # contains corpora for pre-training
    |--datasets/ # contains downstream tasks
    |--models/ # contains pre-trained models, vocabularies, and configuration files
    |--scripts/ # contains useful scripts for pre-training models
    |--finetune/ # contains fine-tuning scripts for downstream tasks
    |--inference/ # contains inference scripts for downstream tasks
    |
    |--preprocess.py
    |--pretrain.py
    |--README.md
    |--README_ZH.md
    |--requirements.txt
    |--logo.jpg

```

The code is well-organized. Users can use and extend upon it with little efforts.

More examples of using UER can be found in :arrow_right: [__instructions__](https://github.com/dbiir/UER-py/wiki/Instructions) :arrow_left: , which help users quickly implement pre-training models such as BERT, GPT-2, ELMo, T5 and fine-tune pre-trained models on a range of downstream tasks.

<br/>

## Competition solutions
UER-py has been used in winning solutions of many NLP competitions. In this section, we provide some examples of using UER-py to achieve SOTA results on NLP competitions, such as CLUE. See :arrow_right: [__competition solutions__](https://github.com/dbiir/UER-py/wiki/Competition-solutions) :arrow_left: for more detailed information.

<br/>

## Citation
#### If you are using the work (e.g. pre-trained model) in UER-py for academic work, please cite the [system paper](https://arxiv.org/pdf/1909.05658.pdf) published in EMNLP 2019:
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

## Contact information
For communication related to this project, please contact Zhe Zhao (helloworld@ruc.edu.cn; nlpzhezhao@tencent.com) or Yudong Li (liyudong123@hotmail.com) or Cheng Hou (chenghoubupt@bupt.edu.cn) or Xin Zhao (zhaoxinruc@ruc.edu.cn).

This work is instructed by my enterprise mentors __Qi Ju__, __Xuefeng Yang__, __Haotang Deng__ and school mentors __Tao Liu__, __Xiaoyong Du__.

We also got a lot of help from Weijie Liu, Lusheng Zhang, Jianwei Cui, Xiayu Li, Weiquan Mao, Hui Chen, Jinbin Zhang, Zhiruo Wang, Peng Zhou, Haixiao Liu, and Weijian Wu. 
