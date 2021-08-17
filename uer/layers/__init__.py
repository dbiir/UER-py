from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import DualEmbedding

str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "dual": DualEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "WordSinusoidalposEmbedding",
           "DualEmbedding", "str2embedding"]
