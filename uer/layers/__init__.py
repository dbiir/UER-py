from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "str2embedding"]

