[**English**](https://github.com/dbiir/UER-py) | [**中文**](https://github.com/dbiir/UER-py/blob/master/README_ZH.md)

[![Build Status](https://travis-ci.org/dbiir/UER-py.svg?branch=master)](https://travis-ci.org/dbiir/UER-py)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)

<img src="logo.jpg" width="390" hegiht="390" align=left />

预训练已经成为自然语言处理任务的重要组成部分，为大量自然语言处理任务带来了显著提升。 UER-py（Universal Encoder Representations）是一个用于对通用语料进行预训练并对下游任务进行微调的工具包。UER-py遵循模块化的设计原则。通过模块的组合，用户能迅速精准复现已有的预训练模型，并利用已有的接口进一步开发更多的预训练模型。通过UER-py，我们建立了一个模型仓库，其中包含基于不同语料，编码器和目标任务的预训练模型。用户可以根据具体任务的要求，从中选择合适的预训练模型使用。


<br>

目录
=================
  * [项目特色](#项目特色)
  * [依赖环境](#依赖环境)
  * [快速上手](#快速上手)
  * [数据集](#数据集)
  * [预训练模型仓库](#预训练模型仓库)
  * [使用说明](#使用说明)
  * [竞赛解决方案](#竞赛解决方案)
  * [引用](#引用)


<br/>

## 项目特色
UER-py有如下几方面优势:
- __可复现__ UER-py已在许多数据集上进行了测试，与原始预训练模型实现（例如BERT，GPT，ELMo，T5）的表现相匹配。
- __多GPU模式__ UER-py支持CPU、单机单GPU、单机多GPU、多机多GPU训练模式。BERT类模型计算量大。多GPU模式使得UER-py能够在大规模语料上进行预训练
- __模块化__ UER-py使用解耦的模块化设计框架。模型分成Embedding、Encoder、Target三个部分。每个部分提供清晰的接口，便于对模型的模块进行组合优化
- __高效__ UER-py优化了预处理，预训练和微调阶段的代码，从而大大提高了速度并减少了内存需求
- __模型仓库__ 我们维护并持续发布中文预训练模型。用户可以根据具体任务的要求，从中选择合适的预训练模型使用。
- __SOTA结果__ UER-py支持全面的下游任务，包括文本分类、文本对分类、序列标注、阅读理解等
- __相关功能__ UER-py提供了丰富的预训练相关的功能和优化，包括特征抽取、近义词检索、预训练模型转换、模型集成、混合精度训练等


<br/>

## 依赖环境
* Python >= 3.6
* torch >= 1.0
* six >= 1.12.0
* argparse
* packaging
* 如果使用混合精度，需要安装英伟达的apex
* 如果涉及到TensorFlow模型的转换，需要安装TensorFlow
* 如果在tokenizer中使用sentencepiece模型，需要安装sentencepiece
* 如果使用模型集成stacking，需要LightGBM和[BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

<br/>

## 快速上手
我们使用BERT模型和[豆瓣书评分类数据集](https://embedding.github.io/evaluation/)简要说明如何使用UER-py。 我们首先在书评语料上对模型进行预训练，然后在书评分类数据集上对其进行微调。这个过程有三个输入文件：书评语料，书评分类数据集和中文词典。这些文件均为UTF-8编码，并被包括在这个项目中。

BERT模型要求的预训练语料格式是一行一个句子，不同文档使用空行分隔，如下所示：

```
doc1-sent1
doc1-sent2
doc1-sent3

doc2-sent1

doc3-sent1
doc3-sent2
```
书评语料是由书评分类数据集去掉标签得到的。我们将一条评论从中间分开，从而形成一个两句话的文档，具体可见*corpora*文件夹中的*book_review_bert.txt*。

分类数据集的格式如下：
```
label    text_a
1        instance1
0        instance2
1        instance3
```
标签和文本之间用\t分隔，第一行是列名。对于n分类，标签应该是0到n-1之间（包括0和n-1）的整数。

词典文件的格式是一行一个单词，我们使用谷歌提供的包含21128个中文字符的词典文件*models/google_zh_vocab.txt*

我们首先对书评语料进行预处理。预处理阶段需要指定模型的目标任务（*--target*）：
```
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt --dataset_path dataset.pt \
                      --processes_num 8 --target bert
```
注意我们需要安装 ``six>=1.12.0``。

预处理非常耗时，使用多个进程可以大大加快预处理速度（*--processes_num*）。 原始文本在预处理之后被转换为*pretrain.py*的可以接受的输入，*dataset.pt*。然后下载Google中文预训练模型[*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb)（此文件为UER支持的格式），并将其放在 *models* 文件夹中。接着加载Google中文预训练模型，在书评语料上对其进行增量预训练。预训练模型由词向量层，编码层和目标任务层组成。因此要构建预训练模型，我们应明确指定模型的词向量层（*--embedding*），编码器层（*--encoder* 和 *--mask*）和目标任务层（*--target*）的类型。假设我们有一台带有8个GPU的机器：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin  --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* 指定注意力网络中使用的遮罩类型。BERT使用双向语言模型，句子中的任意一个词可以看到所有词的信息，因此我们使用 *fully_visible* 遮罩类型。默认情况下，配置文件为 *models/bert/base_config.json* 。配置文件指定了模型的超参数。
请注意，*pretrain.py*输出的模型会带有记录训练步数的后缀，这里我们可以删除后缀以方便使用。

然后，我们在下游分类数据集上微调预训练模型，我们可以用 *google_zh_model.bin*:
```
python3 run_classifier.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
```
或者使用 *pretrain.py* 的输出[*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA)
```
python3 run_classifier.py --pretrained_model_path models/book_review_model.bin --vocab_path models/google_zh_vocab.txt \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 3 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
实验结果显示，谷歌BERT模型在书评分类任务测试集上的结果为准确率87.5；*book_review_model.bin* 在其上的结果是88.2。值得注意的是，我们不需要在微调阶段指定目标任务。预训练模型的目标任务已被替换为特定下游任务需要的目标任务。

微调后的模型的默认路径是*models/finetuned_model.bin*, 然后我们利用微调后的分类器模型进行预测。 
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* 指定需要预测的文件；<br>
*--prediction_path* 指定预测结果的文件；<br>
注意到我们需要指定分类任务标签的个数 *--labels_num* ，这里是二分类任务。

一般下游任务规模不大，通常可以使用一个GPU进行微调和推理，推荐使用CUDA_VISIBLE_DEVICES指定程序可见的GPU（如果不指定，则使用所有的GPU）：
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
注意到这里我们在微调阶段使用 *--output_model_path* 指定微调后的模型的输出路径。
<br>
预测是否是下一个句子（NSP）是BERT的目标任务之一，但是，NSP任务不适合句子级别的评论，因为我们需要将句子切分为多个部分。 UER-py可以让用户自由选择不同的目标任务。这里我们选择使用遮罩语言模型（MLM）作为目标任务。MLM目标任务对书籍评论语料可能是更合适的选择：

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
预训练的实际批次大小是 *--batch_size* 乘以 *--world_size*。 <br>
实验证明[*book_review_mlm_model.bin*](https://share.weiyun.com/V0XidqrV)的结果为88.5。
<br>

BERT参数量大，计算较慢。我们希望加速模型运算的同时让模型仍然在下游任务上有好的表现。这里我们选择2层LSTM编码器来替代12层Transformer编码器。 我们首先下载2层LSTM编码器的预训练模型[*reviews_lstm_lm_model.bin*](https://share.weiyun.com/57dZhqo)。 然后在下游分类数据集上对其进行微调：
```
python3 run_classifier.py --pretrained_model_path models/reviews_lstm_lm_model.bin --vocab_path models/google_zh_vocab.txt --config_path models/rnn_config.json \
                          --train_path datasets/douban_book_review/train.tsv --dev_path datasets/douban_book_review/dev.tsv --test_path datasets/douban_book_review/test.tsv \
                          --epochs_num 5  --batch_size 64 --learning_rate 1e-3 --embedding word --encoder lstm --pooling mean

python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin --vocab_path models/google_zh_vocab.txt \
                                          --config_path models/rnn_config.json --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 --embedding word --encoder lstm --pooling mean
```
我们可以在书评分类任务测试集上获得超过85.4的准确率。相比之下，我们使用相同的LSTM编码器，但是不加载预训练模型，只能获得约81的准确率。
<br>

UER-py还支持更多的预训练模型。 <br>
在Chnsenticorp数据集上使用ELMo进行预训练和微调的示例：
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
用户可以从[这里](https://share.weiyun.com/5Qihztq)下载 *mixed_corpus_elmo_model.bin*。

在Chnsenticorp数据集上微调GatedCNN模型的示例：
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
用户可以从[这里](https://share.weiyun.com/W2gmPPeA)下载 *wikizh_gatedcnn_lm_model.bin*。
<br>

UER-py支持分类任务的交叉验证，在竞赛数据集[SMP2020-EWECT](http://39.97.118.137/)上使用交叉验证的示例：
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
*google_zh_model.bin* 的结果为79.1/63.8（准确率/F1值）；<br>
*--folds_num* 指定交叉验证的轮数；<br>
*--output_path* 指定微调模型的路径，共保存 *--folds_num* 个微调后的模型，并将 *fold ID* 后缀添加到模型名称中；<br>
*--train_features_path* 指定out-of-fold预测文件的路径；训练集被分成了 *--folds_num* 折。一折样本的预测概率是由其他折上的数据训练的模型预测得到的。*train_features.npy* 可用于stacking集成。*竞赛解决方案*部分中介绍了更多详细信息。<br>

我们可以进一步尝试不同的预训练模型。例如，可以下载[*RoBERTa-wwm-ext-large from HIT*](https://github.com/ymcui/Chinese-BERT-wwm)并将其转换为UER格式：
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
*RoBERTa-wwm-ext-large* 的结果是80.3/66.8（准确率/F1值）。 <br>
使用我们的预训练模型[*Reviews+BertEncoder(large)+MlmTarget*](https://share.weiyun.com/hn7kp9bs)的示例如下（更多详细信息，请参见模型仓库）：
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_classifier_cv.py --pretrained_model_path models/reviews_bert_large_mlm_model.bin \
                                                      --vocab_path models/google_zh_vocab.txt \
                                                      --config_path models/bert/large_config.json \
                                                      --train_path datasets/smp2020-ewect/virus/train.tsv \
                                                      --train_features_path datasets/smp2020-ewect/virus/train_features.npy \
                                                      --folds_num 5 --epochs_num 3 --batch_size 64 --seed 17 \
                                                      --embedding word_pos_seg --encoder transformer --mask fully_visible
```
结果为81.3/68.4（准确率/F1值），与其他开源预训练权重相比，这一结果很具有竞争力。 <br>
有时大模型无法收敛，我们需要通过指定 *--seed* 尝试不同的随机种子。
<br>

除了分类外，UER-py还提供其他下游任务的脚本。我们可以使用*run_ner.py*进行命名实体识别：
```
python3 run_ner.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                   --train_path datasets/msra_ner/train.tsv --dev_path datasets/msra_ner/dev.tsv --test_path datasets/msra_ner/test.tsv \
                   --output_model_path models/ner_model.bin \
                   --label2id_path datasets/msra_ner/label2id.json --epochs_num 5 --batch_size 16 \
                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--label2id_path* 指定用于命名实体识别的label2id文件的路径。

然后我们对ner模型进行推断：
```
python3 inference/run_ner_infer.py --load_model_path models/ner_model.bin --vocab_path models/google_zh_vocab.txt \
                                   --test_path datasets/msra_ner/test_nolabel.tsv \
                                   --prediction_path datasets/msra_ner/prediction.tsv \
                                   --label2id_path datasets/msra_ner/label2id.json \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
```
<br>

我们可以使用 *run_cmrc.py* 来进行机器阅读理解：
```
python3 run_cmrc.py --pretrained_model_path models/google_zh_model.bin --vocab_path models/google_zh_vocab.txt \
                    --train_path datasets/cmrc2018/train.json --dev_path datasets/cmrc2018/dev.json \
                    --output_model_path models/cmrc_model.bin \
                    --epochs_num 2 --batch_size 8 --seq_length 512 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```
我们不指定 *-test_path*，因为CMRC2018数据集不提供测试集的标签。
然后，我们使用cmrc模型进行推断：
```
python3 inference/run_cmrc_infer.py --load_model_path models/cmrc_model.bin --vocab_path models/google_zh_vocab.txt \
                                    --test_path datasets/cmrc2018/test.json  \
                                    --prediction_path datasets/cmrc2018/prediction.json --seq_length 512 \
                                    --embedding word_pos_seg --encoder transformer --mask fully_visible
```

<br/>

## 数据集
我们收集了一系列[__下游任务数据集__](https://github.com/dbiir/UER-py/wiki/下游任务数据集)并将其转换为UER可以直接加载的格式。

<br/>

## 预训练模型仓库
借助UER-py，我们使用不同的语料，编码器和目标任务进行了预训练，所有预训练模型都可以由UER-py直接加载。将来我们会发布更多的预训练模型。用户可以在[__预训练模型仓库__](https://github.com/dbiir/UER-py/wiki/模型仓库)中找到预训练模型和对应描述和下载链接。

<br/>

## 使用说明
### 整体框架
UER-py使用解耦的设计框架，方便用户使用和扩展，项目组织如下：
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

更多使用示例在[__使用说明__](https://github.com/dbiir/UER-py/wiki/使用说明)中可以找到。这些示例可以帮助用户快速完成BERT、GPT、ELMO等预训练模型以及使用这些预训练模型在一系列下游任务上微调的实验。

<br/>

## 竞赛解决方案
UER-py已用于许多NLP竞赛的获奖解决方案中。在本节中，我们提供了一些使用UER-py在NLP比赛中获得SOTA成绩的示例，例如CLUE。更多详细信息参见[__竞赛解决方案__](https://github.com/dbiir/UER-py/wiki/竞赛解决方案)。

<br/>

## 引用
#### 如果您在您的学术工作中使用我们的工作（比如预训练权重），可以引用我们的[论文](https://arxiv.org/pdf/1909.05658.pdf)：
```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```
