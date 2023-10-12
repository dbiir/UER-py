from uer.embeddings.embedding import Embedding
from uer.embeddings.dual_embedding import DualEmbedding
from uer.embeddings.word_embedding import WordEmbedding
from uer.embeddings.pos_embedding import PosEmbedding
from uer.embeddings.seg_embedding import SegEmbedding
from uer.embeddings.sinusoidalpos_embedding import SinusoidalposEmbedding


str2embedding = {"word": WordEmbedding, "pos": PosEmbedding, "seg": SegEmbedding,
                 "sinusoidalpos": SinusoidalposEmbedding, "dual": DualEmbedding,}

__all__ = ["Embedding", "WordEmbedding", "PosEmbedding", "SegEmbedding", "SinusoidalposEmbedding",
           "DualEmbedding", "str2embedding"]
