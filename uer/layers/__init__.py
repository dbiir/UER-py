from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import PatchEmbedding


str2embedding = {"patch": PatchEmbedding ,"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding}

__all__ = ["PatchEmbedding", "WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding",
           "WordSinusoidalposEmbedding", "str2embedding"]
