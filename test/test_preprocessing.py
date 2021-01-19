import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tf_seq2seq.text_preprocessing import (  # noqa: E402
    unicode_to_ascii, preprocess_sentence
)


@pytest.mark.parametrize("test_input,expected", [
    ("éèàùò", "eeauo")
])
def test_unicode_to_ascii(test_input, expected):
    assert(unicode_to_ascii(test_input) == expected)


@pytest.mark.parametrize("test_input,expected", [
    (u"May I borrow this book?", "<start> may i borrow this book ? <end>"),
    (u"¿Puedo tomar prestado este libro?",
     "<start> ¿ puedo tomar prestado este libro ? <end>")
])
def test_preprocess_sentence(test_input, expected):
    assert(preprocess_sentence(test_input) == expected)
