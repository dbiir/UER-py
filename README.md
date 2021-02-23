[**English**](https://github.com/dbiir/UER-py) | [**中文**](https://github.com/dbiir/UER-py/blob/master/README_ZH.md)

[![Build Status](https://travis-ci.org/dbiir/UER-py.svg?branch=master)](https://travis-ci.org/dbiir/UER-py)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)

<img src="logo.jpg" width="390" hegiht="390" align=left />

Pre-training has become an essential part for NLP tasks and has led to remarkable improvements. UER-py (Universal Encoder Representations) is a toolkit for pre-training on general-domain corpus and fine-tuning on downstream task. UER-py maintains model modularity and supports research extensibility. It facilitates the use of pre-training models, and provides interfaces for users to further extend upon. With UER-py, we build a model zoo which contains pre-trained models based on different corpora, encoders, and targets. 


<br/>
Recently we received [Chinese 36-layers RoBERTa](https://share.weiyun.com/J9rj9WRB) and [Chinese 36-layers RoBERTa-WWM](https://share.weiyun.com/UsI0OSeR) models pre-trained with Tencent Cloud TI-ONE (see models/bert_xlarge_config.json for detailed configuration). We were told that the 36-layers RoBERTa achieves improvements on some downstream tasks such as XNLI and CMRC2018 and models with more corpora and training steps will be released in the near future. We look forward to user feedback on this model. We recommend to use small learning rates (e.g. 1e-5) for large models.
<br>

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
- __Reproducibility.__ UER-py has been tested on many datasets and should match the performances of the original pre-training model implementations such as BERT, GPT, ELMo, and T5.
- __Multi-GPU.__ UER-py supports CPU mode, single GPU mode, and distributed training mode. 
- __Model modularity.__ UER-py is divided into multiple components: embedding, encoder, target, and downstream task fine-tuning. Ample modules are implemented in each component. Clear and robust interface allows users to combine modules with as few restrictions as possible.
- __Efficiency.__ UER-py refines its pre-processing, pre-training, and fine-tuning stages, which largely improves speed and needs less memory.
- __Model zoo.__ With the help of UER-py, we pre-trained models with different corpora, encoders, and targets. Proper selection of pre-trained models is important to the downstream task performances.
- __SOTA results.__ UER-py supports comprehensive downstream tasks (e.g. classification and machine reading comprehension) and has been used in winning solutions of many NLP competitions.


<br/>

## Requirements
* Python 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* For the mixed precision training you will need apex from NVIDIA
* For the pre-trained model conversion (related with TensorFlow) you will need TensorFlow
* For the tokenization with sentencepiece model you will need SentencePiece
* For developing a stacking model you will need LightGBM and [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

<br/>

## Quickstart
We firstly use BERT model and [Douban book review classification dataset](https://embedding.github.io/evaluation/) to demonstrate how to use UER-py. We pre-train model on book review corpus and then fine-tune it on classification dataset. There are three input files: book review corpus, book review classification dataset, and vocabulary. All files are encoded in UTF-8 and are included in this project.

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

We use Google's Chinese vocabulary file *models/google_zh_vocab.txt*, which contains 21128 Chinese characters.

We firstly preprocess the book review corpus. We need to specify the model's target in pre-processing stage (*--target*):
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
Notice that ``six>=1.12.0`` is required.

Pre-processing is time-consuming. Using multiple processes can largely accelerate the pre-processing speed (*--processes_num*). After pre-processing, the raw text is converted to *dataset.pt*, which is the input of *pretrain.py*. Then we download Google's original pre-trained Chinese BERT model [*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb) (in UER's format), and put it in *models* folder. We load the pre-trained Chinese BERT model and train it on book review corpus. Pre-training model is composed of embedding, encoder, and target. To build a pre-training model, we should explicitly specify model's embedding (*--embedding*), encoder (*--encoder* and *--mask*), and target (*--target*). Suppose we have a machine with 8 GPUs.:
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* specifies the attention mask types. BERT uses bidirectional LM. The word token can attend to all tokens and therefore we use *fully_visible* mask type. By default, *models/bert/base_config.json* is used as configuration file, which specifies the model hyper-parameters. 
Notice that the model trained by *pretrain.py* is attacted with the suffix which records the training step. We could remove the suffix for ease of use.


Then we fine-tune pre-trained models on downstream classification dataset. We can use *google_zh_model.bin*:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
or use our [*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA), which is the output of *pretrain.py*：
```
python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
It turns out that the result of Google's model is 87.5; The result of *book_review_model.bin* is 88.2. It is also noticeable that we don't need to specify the target in fine-tuning stage. Pre-training target is replaced with task-specific target.

The default path of the fine-tuned classifier model is *models/finetuned_model.bin* . Then we do inference with the fine-tuned model. 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* specifies the path of the file to be predicted. <br>
*--prediction_path* specifies the path of the file with prediction results. <br>
We need to explicitly specify the number of labels by *--labels_num*. Douban book review is a two-way classification dataset.

We recommend to use *CUDA_VISIBLE_DEVICES* to specify which GPUs are visible (all GPUs are used in default) :
```
CUDA_VISIBLE_DEVICES=0 python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                                                 --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                 --output_model_path models/classifier_model.bin
                                                 --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible

CUDA_VISIBLE_DEVICES=0 python3 inference/run_classifier_infer.py --load_model_path models/classifier_model.bin --vocab_path models/google_zh_vocab.txt \
                                                                 --test_path datasets/douban_book_review/test_nolabel.tsv \
                                                                 --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                                                 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
Notice that we explicitly specify the fine-tuned model path by *--output_model_path* in fine-tuning stage.
<br>

BERT consists of next sentence prediction (NSP) target. However, NSP target is not suitable for sentence-level reviews since we have to split a sentence into multiple parts. UER-py facilitates the use of different targets. Using masked language modeling (MLM) as target could be a properer choice for pre-training of reviews:
```
python3 preprocess.py --corpus_path corpora/book_review.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target mlm

python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_mlm_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 2500 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible --target mlm

mv models/book_review_mlm_model.bin-5000 models/book_review_mlm_model.bin

CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier.py --pretrained_model_path models/book_review_mlm_model.bin --vocab_path models/google_zh_vocab.txt \
                                                   --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                                                   --epochs_num 3 --batch_size 64 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The actual batch size of pre-training is *--batch_size* times *--world_size* . <br>
It turns out that the result of [*book_review_mlm_model.bin*](https://share.weiyun.com/V0XidqrV) is around 88.5.
<br>

BERT is slow. It could be great if we can speed up the model and still achieve competitive performance. To achieve this goal, we select a 2-layers LSTM encoder to substitute 12-layers Transformer encoder. We firstly download [*reviews_lstm_lm_model.bin*](https://share.weiyun.com/57dZhqo) for 2-layers LSTM encoder. Then we fine-tune it on downstream classification dataset:
```
python3 run_classifier.py --pretrained_model_path models/reviews_lstm_lm_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 1e-3 --embedding word --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder lstm --pooling mean
```
We can achieve over 85.4 accuracy on testset, which is a competitive result. Using the same LSTM encoder without pre-training can only achieve around 81 accuracy.
<br>

UER-py also provides many other encoders and corresponding pre-trained models. <br>
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
```
Users can download *mixed_corpus_elmo_model.bin* from [here](https://share.weiyun.com/5Qihztq).

The example of fine-tuning GatedCNN on Chnsenticorp dataset:
```
python3 run_classifier.py --pretrained_model_path models/wikizh_gatedcnn_lm_model.bin \
                          --vocab_path models/google_zh_vocab.txt \
                          --config_path models/gatedcnn_9_config.json \
                          --train_path datasets/chnsenticorp/train.tsv --dev_path datasets/chnsenticorp/dev.tsv --test_path datasets/chnsenticorp/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 5e-5 \
                          --embedding word --encoder gatedcnn --pooling max

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/gatedcnn_9_config.json \
                                          --test_path datasets/chnsenticorp/test_nolabel.tsv \
                                          --prediction_path datasets/chnsenticorp/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder gatedcnn --pooling max
```
Users can download *wikizh_gatedcnn_lm_model.bin* from [here](https://share.weiyun.com/W2gmPPeA).
<br>

UER-py supports cross validation for classification. The example of using cross validation on [SMP2020-EWECT](http://39.97.118.137/), a competition dataset:
```
CUDA_VISIBLE_DEVICES=0 python3 run_classifier_cv.py --pretrained_model_path models/google_zh_model.bin \
                                                    --vocab_path models/google_zh_vocab.txt \
                                                    --config_path models/bert/base_config.json \
                                                    --output_model_path models/classifier_model.bin \
                                                    --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                    --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                    --epochs_num 3 --batch_size 32 --folds_num 5 \
                                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results of *google_zh_model.bin* are 79.1/63.8 (Accuracy/Marco F1). <br>
*--folds_num* specifies the number of rounds of cross-validation. <br>
*--output_path* specifies the path of the fine-tuned model. *--folds_num* models are saved and the *fold ID* suffix is added to the model's name. <br>
*--train_features_path* specifies the path of out-of-fold (OOF) predictions. *run_classifier_cv.py* generates probabilities over classes on each fold by training a model on the other folds in the dataset. *train_features.npy* can be used as features for stacking. More details are introduced in *Competition solutions* section. <br>

We can further try different pre-trained models. For example, we download [*RoBERTa-wwm-ext-large from HIT*](https://github.com/ymcui/Chinese-BERT-wwm) and convert it into UER's format:
```
python3 scripts/convert_bert_from_huggingface_to_uer.py --input_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model.bin \
                                                        --output_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model_uer.bin \
                                                        --layers_num 24

CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/chinese_roberta_wwm_large_ext_pytorch/pytorch_model_uer.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                      --epochs_num 3 --batch_size 64 --folds_num 5 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results of *RoBERTa-wwm-ext-large* are 80.3/66.8 (Accuracy/Marco F1). <br>
The example of using our pre-trained model [*Reviews+BertEncoder(large)+MlmTarget*](https://share.weiyun.com/hn7kp9bs) (see model zoo for more details):
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/reviews_bert_large_mlm_model.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                      --folds_num 5 --epochs_num 3 --batch_size 64 --seed 17 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
The results are 81.3/68.4 (Accuracy/Marco F1), which has very competitive advantage compared with other open-source pre-trained weights. <br>
Sometimes large model does not converge. We need to try different random seeds by specifying *--seed*. 
<br>

Besides classification, UER-py also provides scripts for other downstream tasks. We could use *run_ner.py* for named entity recognition:
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --output_model_path models/ner_model.bin \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 \
                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--label2id_path* specifies the path of label2id file for named entity recognition.
Then we do inference with the fine-tuned ner model:
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
<br>

We could use *run_cmrc.py* for machine reading comprehension:
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --output_model_path models/cmrc_model.bin \
                    --epochs_num 2 --batch_size 8 --seq_length 512 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
We don't specify the *--test_path* because CMRC2018 dataset doesn't provide labels for testset. 
Then we do inference with the fine-tuned cmrc model:
```
python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path datasets/cmrc2018/test.json  \
                                    --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 \
                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```

<br/>

## Datasets
We collected a range of [__downstream datasets__](https://github.com/dbiir/UER-py/wiki/Datasets) and converted them into format that UER can load directly.

<br/>

## Modelzoo
With the help of UER, we pre-trained models with different corpora, encoders, and targets. All pre-trained models can be loaded by UER directly. More pre-trained models will be released in the future. Detailed introduction of pre-trained models and download links can be found in [__modelzoo__](https://github.com/dbiir/UER-py/wiki/Modelzoo).

<br/>

## Instructions
### UER-py's framework
UER-py is organized as follows：
```
UER-py/
    |--uer/
    |    |--encoders/: contains encoders such as RNN, CNN, BERT
    |    |--targets/: contains targets such as language modeling, masked language modeling
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
    |--models/: contains pre-trained models, vocabularies, and configuration files
    |--scripts/: contains useful scripts for pre-training models
    |--inference/：contains inference scripts for downstream tasks
    |
    |--preprocess.py
    |--pretrain.py
    |--run_classifier.py
    |--run_classifier_cv.py
    |--run_mt_classifier.py
    |--run_cmrc.py
    |--run_ner.py
    |--run_dbqa.py
    |--run_c3.py
    |--run_chid.py
    |--README.md

```

The code is well-organized. Users can use and extend upon it with little efforts.

More examples of using UER can be found in [__instructions__](https://github.com/dbiir/UER-py/wiki/Instructions), which help users quickly implement pre-training models such as BERT, GPT, ELMo, T5 and fine-tune pre-trained models on a range of downstream tasks.

<br/>

## Competition solutions
UER-py has been used in winning solutions of many NLP competitions. In this section, we provide some examples of using UER-py to achieve SOTA results on NLP competitions, such as CLUE. See [__competition solutions__](https://github.com/dbiir/UER-py/wiki/Competition-solutions) for more detailed information.

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
For communication related to this project, please contact Zhe Zhao (helloworld@ruc.edu.cn; nlpzhezhao@tencent.com) or Yudong Li (liyudong123@hotmail.com) or Xin Zhao (zhaoxinruc@ruc.edu.cn).

This work is instructed by my enterprise mentors __Qi Ju__, __Xuefeng Yang__, __Haotang Deng__ and school mentors __Tao Liu__, __Xiaoyong Du__.

I also got a lot of help from my Tencent colleagues Hui Chen, Jinbin Zhang, Zhiruo Wang, Weijie Liu, Peng Zhou, Haixiao Liu, and Weijian Wu. 
