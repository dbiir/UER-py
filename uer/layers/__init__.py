from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import DualEmbedding
from uer.layers.embeddings import WordPosSegPinyinBushouEmbedding

str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "dual": DualEmbedding,
                 'word_pos_seg_pinyin_bushou': WordPosSegPinyinBushouEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "WordSinusoidalposEmbedding",
           "WordPosSegPinyinBushouEmbedding", "DualEmbedding", "str2embedding"]
