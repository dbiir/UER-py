[**English**](https://github.com/dbiir/UER-py) | [**中文**](https://github.com/dbiir/UER-py/blob/master/README_ZH.md) 

[![Build Status](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml/badge.svg)](https://github.com/dbiir/UER-py/actions/workflows/github-actions.yml)
[![codebeat badge](https://codebeat.co/badges/f75fab90-6d00-44b4-bb42-d19067400243)](https://codebeat.co/projects/github-com-dbiir-uer-py-master)
![](https://img.shields.io/badge/license-MIT-000000.svg)
[![arXiv](https://img.shields.io/badge/arXiv-1909.05658-<color>.svg)](https://arxiv.org/abs/1909.05658)

<img src="logo.jpg" width="390" hegiht="390" align=left />

预训练已经成为自然语言处理任务的重要组成部分，为大量自然语言处理任务带来了显著提升。 UER-py（Universal Encoder Representations）是一个用于对通用语料进行预训练并对下游任务进行微调的工具包。UER-py遵循模块化的设计原则。通过模块的组合，用户能迅速精准的复现已有的预训练模型，并利用已有的接口进一步开发更多的预训练模型。通过UER-py，我们建立了一个模型仓库，其中包含基于不同语料，编码器和目标任务的预训练模型。用户可以根据具体任务的要求，从中选择合适的预训练模型使用。**[完整文档](https://github.com/dbiir/UER-py/wiki/主页)请参见本项目Wiki**。


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
- __可复现__ UER-py已在许多数据集上进行了测试，与原始预训练模型实现（例如BERT、GPT-2、ELMo、T5）的表现相匹配
- __多GPU模式__ UER-py支持CPU、单机单GPU、单机多GPU、多机多GPU训练模式。多GPU模式让UER-py能够在大规模语料上进行预训练
- __模块化__ UER-py使用解耦的模块化设计框架。框架分成Embedding、Encoder、Target三个部分。各个部分之间有着清晰的接口并且每个部分包括了丰富的模块。可以对不同模块进行组合，构建出性质不同的预训练模型
- __高效__ UER-py优化了预处理，预训练，微调，推理阶段的代码，从而大大提高了速度并减少了对内存和磁盘空间的需求
- __模型仓库__ 我们维护并持续发布中文预训练模型。用户可以根据具体任务的要求，从中选择合适的预训练模型使用
- __SOTA结果__ UER-py支持全面的下游任务，包括文本分类、文本对分类、序列标注、阅读理解等，并提供了多个竞赛获胜解决方案
- __预训练相关功能__ UER-py提供了丰富的预训练相关的功能和优化，包括特征抽取、近义词检索、预训练模型转换、模型集成、混合精度训练等


<br/>

## 依赖环境
* Python >= 3.6
* torch >= 1.1
* six >= 1.12.0
* argparse
* packaging
* 如果使用混合精度，需要安装英伟达的apex
* 如果涉及到TensorFlow模型的转换，需要安装TensorFlow
* 如果在tokenizer中使用sentencepiece模型，需要安装[SentencePiece](https://github.com/google/sentencepiece)
* 如果使用模型集成stacking，需要安装LightGBM和[BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
* 如果使用全词遮罩（whole word masking）预训练，需要安装分词工具，例如[jieba](https://github.com/fxsjy/jieba)
* 如果在序列标注下游任务中使用CRF，需要安装[pytorch-crf](https://github.com/kmkurn/pytorch-crf)


<br/>

## 快速上手
这里我们通过常用的例子来简要说明如何使用UER-py，更多的细节请参考使用说明章节。我们首先使用BERT模型和[豆瓣书评分类数据集](https://embedding.github.io/evaluation/)。我们在书评语料上对模型进行预训练，然后在书评分类数据集上对其进行微调。这个过程有三个输入文件：书评语料，书评分类数据集和中文词典。这些文件均为UTF-8编码，并被包括在这个项目中。

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
python3 preprocess.py --corpus_path corpora/book_review_bert.txt --vocab_path models/google_zh_vocab.txt \
                      --dataset_path dataset.pt --processes_num 8 --target bert
```
注意我们需要安装 *six>=1.12.0*。

预处理非常耗时，使用多个进程可以大大加快预处理速度（*--processes_num*）。默认的分词器为 *--tokenizer bert* 。原始文本在预处理之后被转换为*pretrain.py*的可以接收的输入，*dataset.pt*。然后下载Google中文预训练模型[*google_zh_model.bin*](https://share.weiyun.com/A1C49VPb)（此文件为UER支持的格式，原始模型来自于[这里](https://github.com/google-research/bert)），并将其放在 *models* 文件夹中。接着加载Google中文预训练模型，在书评语料上对其进行增量预训练。预训练模型由词向量层，编码层和目标任务层组成。因此要构建预训练模型，我们应明确指定模型的词向量层（*--embedding*），编码器层（*--encoder* 和 *--mask*）和目标任务层（*--target*）的类型。假设我们有一台带有8个GPU的机器：
```
python3 pretrain.py --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --pretrained_model_path models/google_zh_model.bin \
                    --output_model_path models/book_review_model.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 5000 --save_checkpoint_steps 1000 --batch_size 32 \
                    --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert

mv models/book_review_model.bin-5000 models/book_review_model.bin
```
*--mask* 指定注意力网络中使用的遮罩类型。BERT使用双向语言模型，句子中的任意一个词可以看到所有词的信息，因此我们使用 *fully_visible* 遮罩类型。BERT模型的词向量层是word（token）、position、segment向量的求和，因此我们使用 *--embedding word_pos_seg* 。默认情况下，配置文件为 *models/bert/base_config.json* 。配置文件指定了模型的超参数。
请注意，*pretrain.py*输出的模型会带有记录训练步数的后缀（*--total_steps*），这里我们可以删除后缀以方便使用。

然后，我们在下游分类数据集上微调预训练模型，我们使用 *pretrain.py* 的输出[*book_review_model.bin*](https://share.weiyun.com/xOFsYxZA)：
```
python3 finetune/run_classifier.py --pretrained_model_path models/book_review_model.bin \
                                   --vocab_path models/google_zh_vocab.txt \
                                   --train_path datasets/douban_book_review/train.tsv \
                                   --dev_path datasets/douban_book_review/dev.tsv \
                                   --test_path datasets/douban_book_review/test.tsv \
                                   --epochs_num 3 --batch_size 32 \
                                   --embedding word_pos_seg --encoder transformer --mask fully_visible
``` 
值得注意的是，我们不需要在微调阶段指定目标任务。预训练模型的目标任务已被替换为特定下游任务需要的目标任务。

微调后的模型的默认路径是*models/finetuned_model.bin*, 然后我们利用微调后的分类器模型进行预测：
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/google_zh_vocab.txt \
                                          --test_path datasets/douban_book_review/test_nolabel.tsv \
                                          --prediction_path datasets/douban_book_review/prediction.tsv \
                                          --labels_num 2 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
*--test_path* 指定需要预测的文件，文件需要包括text_a列；<br>
*--prediction_path* 指定预测结果的文件；<br>
注意到我们需要指定分类任务标签的个数 *--labels_num* ，这里是二分类任务。

<br>

以上我们给出了用UER-py进行预处理、预训练、微调、推理的基本使用方式。更多的例子参见完整的 :arrow_right: [__快速上手__](https://github.com/dbiir/UER-py/wiki/快速上手) :arrow_left: 。完整的快速上手包含了全面的例子，覆盖了大多数预训练相关的使用场景。推荐用户完整阅读快速上手章节，以便能合理的使用本项目。

<br/>

## 数据集
我们收集了一系列 :arrow_right: [__下游任务数据集__](https://github.com/dbiir/UER-py/wiki/下游任务数据集) :arrow_left: 并将其转换为UER可以直接加载的格式。

<br/>

## 预训练模型仓库
借助UER-py，我们使用不同的语料，编码器和目标任务进行预训练。用户可以在 :arrow_right: [__预训练模型仓库__](https://github.com/dbiir/UER-py/wiki/预训练模型仓库) :arrow_left: 中找到各种性质的预训练模型以及它们对应的描述和下载链接。所有预训练模型都可以由UER-py直接加载。将来我们会发布更多的预训练模型。

<br/>

## 使用说明
UER-py使用解耦的设计框架，方便用户使用和扩展，项目组织如下：
```
UER-py/
    |--uer/
    |    |--encoders/ # 包括编码器模块，例如RNN, CNN, Transformer
    |    |--targets/ # 包括目标任务模块，例如语言模型, 遮罩语言模型
    |    |--layers/ # 包括常用的神经网络层
    |    |--models/ # 包括 model.py，用于组合词向量（embedding）、编码器（encoder）、目标任务（target）模块
    |    |--utils/ # 包括常用的功能模块
    |    |--model_builder.py
    |    |--model_loader.py
    |    |--model_saver.py
    |    |--trainer.py
    |
    |--corpora/ # 预训练语料存放文件夹
    |--datasets/ # 下游任务数据集存放文件夹
    |--models/ # 模型、词典、配置文件存放文件夹
    |--scripts/ # 实用脚本存放文件夹
    |--finetune/ # 微调脚本存放文件夹
    |--inference/ # 前向推理脚本存放文件夹
    |
    |--preprocess.py
    |--pretrain.py
    |--README.md
    |--README_ZH.md
    |--requirements.txt
    |--logo.jpg

```

更多使用示例在 :arrow_right: [__使用说明__](https://github.com/dbiir/UER-py/wiki/使用说明) :arrow_left: 中可以找到。这些示例可以帮助用户快速实现BERT、GPT-2、ELMo、T5等预训练模型以及使用这些预训练模型在一系列下游任务上进行微调。

<br/>

## 竞赛解决方案
UER-py已用于许多NLP竞赛的获奖解决方案中。在本章节中，我们提供了一些使用UER-py在NLP竞赛中获得SOTA成绩的示例，例如CLUE。更多详细信息参见 :arrow_right: [__竞赛解决方案__](https://github.com/dbiir/UER-py/wiki/竞赛解决方案) :arrow_left: 。

<br/>

## 引用
#### 如果您在您的学术工作中使用我们的工作（比如预训练模型权重），可以引用我们的[论文](https://arxiv.org/pdf/1909.05658.pdf)：
```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```
