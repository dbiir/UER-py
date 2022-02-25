from uer.embeddings.dual_embedding import DualEmbedding
from uer.embeddings.word_embedding import WordEmbedding
from uer.embeddings.wordpos_embedding import WordPosEmbedding
from uer.embeddings.wordposseg_embedding import WordPosSegEmbedding
from uer.embeddings.wordsinusoidalpos_embedding import WordSinusoidalposEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding, "dual": DualEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding", "WordSinusoidalposEmbedding",
           "DualEmbedding", "str2embedding"]
