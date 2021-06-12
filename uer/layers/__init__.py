from uer.layers.embeddings import WordEmbedding
from uer.layers.embeddings import WordPosEmbedding
from uer.layers.embeddings import WordPosSegEmbedding
from uer.layers.embeddings import WordSinusoidalposEmbedding
from uer.layers.embeddings import PatchEmbedding
from uer.layers.embeddings import PatchPosEmbedding
from uer.layers.embeddings import ViLEmbedding
from uer.layers.embeddings import ClipEmbedding

str2embedding = {"patch_pos": PatchPosEmbedding , "patch": PatchEmbedding , "clip": "ClipEmbedding", "vil": ViLEmbedding ,"word": WordEmbedding,
                 "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,"word_sinusoidalpos": WordSinusoidalposEmbedding}

__all__ = ["PatchPosEmbedding", "PatchEmbedding", "ViLEmbedding", "ClipEmbedding", "WordEmbedding", "WordPosEmbedding",
           "WordPosSegEmbedding","WordSinusoidalposEmbedding", "str2embedding"]
