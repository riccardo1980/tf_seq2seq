import tensorflow as tf
from typing import Tuple


class Encoder(tf.keras.Model):
    """
        GRU based Encoder for sequences

        - Accepts symbols
        - initializes and uses internal embedding (symbol to dense repr)
        - initializes and uses a GRU layer

        TODO:
            - Work on dense vectors: move embedder outside

    """
    def __init__(self, vocab_size: int,
                 embedding_dim: int, units: int, batch_size: int):
        """
        Initialize the encoder

        :param vocab_size:
            size of the vocabulary, i.e. maximum integer index + 1.
        :param embedding_dim: dimension of the dense embedding
        :param units: dimensionality of the output space.
        :param batch_size: batch size

        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.units = units

        # initialize embedding layer (from symbols to dense vectors)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # initialize recurrent cells
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self,
             x: tf.Tensor,
             hidden: tf.Tensor
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass

        :param x:
            input tensor of shape (batch_size, max_input_length)
        :param hidden:
            hidden state initialization tensor of shape (batch_size, units)

        :return:
            output tensor of shape (batch_size, sequence_length, units)
            state tensor of shape (batch_size, units)
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)

        return output, state

    def initialize_hidden_state(self) -> tf.Tensor:
        """
        Facility for having a zeroed hidden state tensor of correct size

        :return: a tensor of shape (batch_size, units)
        """
        return tf.zeros((self.batch_size, self.units))


class Decoder(tf.keras.Model):
    """
        GRU based Decoder for sequences

        - Accepts symbols
        - initializes and uses internal embedding (symbol to dense repr)
        - initializes and uses a GRU layer

        TODO:
            - work on dense vectors: move embedder outside
            - what the want to return?
                - GRU output (embedding)
                - distribution on symbols
                - [current] output of FC layer of size (batch_size, vocab_size)

    """
    def __init__(self, vocab_size: int,
                 embedding_dim: int, units: int, batch_size: int):
        """
        Initialize the decoder

        :param vocab_size:
            size of the vocabulary, i.e. maximum integer index + 1.
        :param embedding_dim: dimension of the dense embedding
        :param units: dimensionality of the output space.
        :param batch_size: batch size

        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.units = units
        self.vocab_size = vocab_size

        # initialize embedding layer (from symbols to dense vectors)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # initialize recurrent cells
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self,
             x: tf.Tensor,
             hidden: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass

        :param x: input tensor of shape (batch_size, max_input_length)
        :param hidden:
            hidden state initialization tensor of shape (batch_size, units)

        :return:
            output tensor of shape (batch_size, sequence_length, units)
            state tensor of shape (batch_size, vocab_size)
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state
