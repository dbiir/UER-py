[**English**](https://github.com/dbiir/UER-py) | [**中文**](https://github.com/dbiir/UER-py/blob/master/README_ZH.md)

[![Build Status](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml/badge.svg)](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1909.05658-<color>.svg)](https://arxiv.org/abs/1909.05658)

<img src="logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks. UER-py (Universal Encoder Representations) is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of existing pre-training models, and provides interfaces for users to further extend upon. With UER-py, we build a model zoo which contains pre-trained models of different properties. **See the [UER-py project Wiki](https://github.com/dbiir/UER-py/wiki) for full documentation**.

<br/>
<br/>


**🚀** We have open-sourced the [TencentPretrain](https://github.com/Tencent/TencentPretrain), a refactored new version of UER-py. TencentPretrain supports multi-modal models and enables training of large models. If you are interested in text models of medium size (with parameter sizes of less than one billion), we recommend continuing to use the UER-py project.


<br/>

Table of Contents
=================
  * [Features](#features)
  * [Requirements](#requirements)
  * [Quickstart](#quickstart)
  * [Pre-training data](#pre-training-data)
  * [Downstream datasets](#downstream-datasets)
  * [Modelzoo](#modelzoo)
  * [Instructions](#instructions)
  * [Competition solutions](#competition-solutions)
  * [Citation](#citation)
  * [Contact information](#contact-information)


<br/>

## Features
UER-py has the following features:
- __Reproducibility__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations such as BERT, GPT-2, ELMo, and T5.
- __Model modularity__ UER-py is divided into the following components: embedding, encoder, target embedding (optional), decoder (optional), and target. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules to construct pre-training models with as few restrictions as possible.
- __Model training__ UER-py supports CPU mode, single GPU mode and distributed training mode.
- __Model zoo__ With the help of UER-py, we pre-train and release models of different properties. Proper selection of pre-trained models is important to the performances of downstream tasks.
- __SOTA results__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and provides winning solutions of many NLP competitions.
- __Abundant functions__ UER-py provides abundant functions related with pre-training, such as feature extractor and text generation.


<br/>

## Requirements
* Python >= 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* regex
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need [SentencePiece](https://github.com/google/sentencepiece)
* For developing a stacking model you will need LightGBM and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
* For the pre-training with whole word masking you will need word segmentation tool such as [jieba](https://github.com/fxsjy/jieba)
* For the use of CRF in sequence labeling downstream task you will need [pytorch-crf](https://github.com/kmkurn/pytorch-crf)


<br/>

## Quickstart
This section uses several commonly-used examples to demonstrate how to use UER-py. More details are discussed in Instructions section. We firstly use BERT (a text pre-training model) on book review sentiment classification dataset. We pre-train model on book review corpus and then fine-tune it on book review sentiment classification dataset. There are three input files: book review corpus, book review sentiment classification dataset, and vocabulary. All files are encoded in UTF-8 and included in this project.

The format of the corpus for BERT is as follows (one sentence per line and documents are delimited by empty lines)：
```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
The book review corpus is obtained from book review classification dataset. We remove labels and split a review into two parts from the middle to construct a document with two sentences (see *book_review_bert.txt* in *corpora* folder). 

The format of the classification dataset is as follows:
```
label    text_a
1        instance1
0        instance2
1        instance3
```
Label and instance are separated by \t . The first row is a list of column names. The label ID should be an integer between (and including) 0 and n-1 for n-way classification.

We use Google's Chinese vocabulary file *models/google_zh_vocab.txt*, which contains 21128 Chinese characters.

We firstly pre-process the book review corpus. In the pre-processing stage, the corpus needs to be processed into the format required by the specified pre-training model (*--data_processor*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --data_processor bert
```
Notice that *six>=1.12.0* is required.


Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). BERT tokenizer is used in default (*--tokenizer bert*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download Google's pre-trained Chinese BERT model [*google_zh_model.bin*](https://share.weiyun.com/FR4rPxc4) (in UER format and the original model is from [here](https://github.com/google-research/bert)), and put it in *models* folder. We load the pre-trained Chinese BERT model and further pre-train it on book review corpus. Pre-training model is usually composed of embedding, encoder, and target layers. To build a pre-training model, we should provide related information. Configuration file (*--config_path*) specifies the modules and hyper-parameters used by pre-training models. More details can be found in *models/bert/base_config.json*. Suppose we have a machine with 8 GPUs:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --config_path models/bert/base_config.json \
                    --output_model_path models/book_review_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step (*--total_steps*). We could remove the suffix for ease of use.

Then we fine-tune the pre-trained model on downstream classification dataset. We use embedding and encoder layers of [*book_review_model.bin*](https://share.weiyun.com/PnxMrRwZ), which is the output of *pretrain.py*:
```
python3 finetune/run_classifier.py --pretrained_model_path models/book_review_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --config_path models/bert/base_config.json \
                                   --train_path datasets/book_review/train.tsv \
                                   --dev_path datasets/book_review/dev.tsv \
                                   --test_path datasets/book_review/test.tsv \
                                   --epochs_num 3 --batch_size 32
``` 
The default path of the fine-tuned classifier model is *models/finetuned_model.bin* . It is noticeable that the actual batch size of pre-training is *--batch_size* times *--world_size* ; The actual batch size of downstream task (e.g. classification) is *--batch_size* . 
Then we do inference with the fine-tuned model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/bert/base_config.json \
                                          --test_path datasets/book_review/test_nolabel.tsv \
                                          --prediction_path datasets/book_review/prediction.tsv \
                                          --labels_num 2
```
*--test_path* specifies the path of the file to be predicted. The file should contain text_a column.
*--prediction_path* specifies the path of the file with prediction results.
We need to explicitly specify the number of labels by *--labels_num*. The above dataset is a two-way classification dataset.

<br>

The above content provides basic ways of using UER-py to pre-process, pre-train, fine-tune, and do inference. More use cases can be found in complete :arrow_right: [__quickstart__](https://github.com/dbiir/UER-py/wiki/Quickstart) :arrow_left: . The complete quickstart contains abundant use cases, covering most of the pre-training related application scenarios. It is recommended that users read the complete quickstart in order to use the project reasonably.

<br/>

## Pre-training data
This section provides links to a range of :arrow_right: [__pre-training data__](https://github.com/dbiir/UER-py/wiki/Pretraining-data) :arrow_left: . UER can load these pre-training data directly.

<br/>

## Downstream datasets
This section provides links to a range of :arrow_right: [__downstream datasets__](https://github.com/dbiir/UER-py/wiki/Downstream-datasets) :arrow_left: . UER can load these datasets directly.

<br/>

## Modelzoo
With the help of UER, we pre-trained models of different properties (e.g. models based on different corpora, encoders, and targets). Detailed introduction of pre-trained models and their download links can be found in :arrow_right: [__modelzoo__](https://github.com/dbiir/UER-py/wiki/Modelzoo) :arrow_left: . All pre-trained models can be loaded by UER directly.

<br/>

## Instructions
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--embeddings/ # contains modules of embedding component
    |    |--encoders/ # contains modules of encoder component such as RNN, CNN, Transformer
    |    |--decoders/ # contains modules of decoder component
    |    |--targets/ # contains modules of target component such as language modeling, masked language modeling
    |    |--layers/ # contains frequently-used NN layers
    |    |--models/ # contains model.py, which combines modules of different components
    |    |--utils/ # contains frequently-used utilities
    |    |--model_builder.py
    |    |--model_loader.py
    |    |--model_saver.py
    |    |--opts.py
    |    |--trainer.py
    |
    |--corpora/ # contains pre-training data
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
    |--LICENSE

```

The code is organized based on components (e.g. embeddings, encoders). Users can use and extend upon it with little efforts.

Comprehensive examples of using UER can be found in :arrow_right: [__instructions__](https://github.com/dbiir/UER-py/wiki/Instructions) :arrow_left: , which help users quickly implement pre-training models such as BERT, GPT-2, ELMo, T5 and fine-tune pre-trained models on a range of downstream tasks.

<br/>

## Competition solutions
UER-py has been used in winning solutions of many NLP competitions. In this section, we provide some examples of using UER-py to achieve SOTA results on NLP competitions, such as CLUE. See :arrow_right: [__competition solutions__](https://github.com/dbiir/UER-py/wiki/Competition-solutions) :arrow_left: for more detailed information.

<br/>

## Citation
#### If you are using the work (e.g. pre-trained models) in UER-py for academic work, please cite the [system paper](https://arxiv.org/pdf/1909.05658.pdf) published in EMNLP 2019:
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
For communication related to this project, please contact Zhe Zhao (helloworld@alu.ruc.edu.cn; nlpzhezhao@tencent.com) or Yudong Li (liyudong123@hotmail.com) or Cheng Hou (chenghoubupt@bupt.edu.cn) or Wenhang Shi (wenhangshi@ruc.edu.cn).

This work is instructed by my enterprise mentors __Qi Ju__, __Xuefeng Yang__, __Haotang Deng__ and school mentors __Tao Liu__, __Xiaoyong Du__.

We also got a lot of help from Weijie Liu, Lusheng Zhang, Jianwei Cui, Xiayu Li, Weiquan Mao, Xin Zhao, Hui Chen, Jinbin Zhang, Zhiruo Wang, Peng Zhou, Haixiao Liu, and Weijian Wu. 
