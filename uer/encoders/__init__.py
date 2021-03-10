from uer.encoders.transformer_encoder import TransformerEncoder
from uer.encoders.rnn_encoder import RnnEncoder
from uer.encoders.rnn_encoder import LstmEncoder
from uer.encoders.rnn_encoder import GruEncoder
from uer.encoders.rnn_encoder import BirnnEncoder
from uer.encoders.rnn_encoder import BilstmEncoder
from uer.encoders.rnn_encoder import BigruEncoder
from uer.encoders.cnn_encoder import GatedcnnEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "str2encoder"]

