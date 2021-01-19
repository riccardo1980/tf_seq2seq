import sys
import os
import pytest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tf_seq2seq.base import Encoder, Decoder  # noqa: E402


def mock_input_batch(batch_size, max_input_length, vocab_input_size):
    """
        mock input batch
    """
    data = np.random.randint(0, high=vocab_input_size,
                             size=(batch_size, max_input_length))

    return data


@pytest.mark.parametrize("batch_size, vocab_input_size, embedding_dim, units",
                         [
                          (128, 200, 256, 1024),
                          (64, 200, 256, 1024)
                         ])
def test_encoder_initialize_hidden_state_size(batch_size, vocab_input_size,
                                              embedding_dim, units):

    encoder = Encoder(vocab_input_size, embedding_dim, units, batch_size)
    sample_hidden = encoder.initialize_hidden_state()

    assert sample_hidden.shape == (batch_size, units)


@pytest.mark.parametrize(
    """
    vocab_input_size, embedding_input_dim, batch_size,
    max_target_length, vocab_target_size, embedding_dim, units
    """,
    [
        (600, 128, 64, 1, 200, 256, 1024),
        (600, 256, 128, 1, 200, 256, 1024)
    ])
def test_decoder(
    vocab_input_size, embedding_input_dim,
    batch_size, max_target_length, vocab_target_size, embedding_dim, units
):

    # mock input
    input_batch = mock_input_batch(batch_size,
                                   max_target_length,
                                   vocab_target_size)
    dataset = tf.data.Dataset.from_tensor_slices(input_batch).batch(batch_size)
    sample_input = next(iter(dataset))

    # init encoder
    encoder = Encoder(vocab_input_size, embedding_input_dim, units, batch_size)
    sample_hidden = encoder.initialize_hidden_state()
    decoder = Decoder(vocab_target_size, embedding_dim, units, batch_size)

    # call forward
    sample_output, sample_hidden = decoder(sample_input, sample_hidden)

    # compare
    assert sample_hidden.shape == (batch_size, units)
    assert sample_output.shape == (batch_size, vocab_target_size)
