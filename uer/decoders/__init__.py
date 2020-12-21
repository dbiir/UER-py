from uer.decoders.transformer_decoder import TransformerDecoder


str2decoder = {"transformer": TransformerDecoder}

__all__ = ["TransformerDecoder", "str2decoder"]

