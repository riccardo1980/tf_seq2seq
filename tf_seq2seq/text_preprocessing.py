import unicodedata
import re
import io
import tensorflow as tf
import numpy as np
from typing import Tuple, List


def unicode_to_ascii(s: str) -> str:
    """
        Converts unicode string to ascii

        Non-spacing marks (Mn category) are discarded,
        see https://www.fileformat.info/info/unicode/category/Mn/list.htm

        Applies Normalization Form C (NFC)

        :param s: unicode string
        :return: ascii string
    """

    return ''.join(c for c
                   in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w: str) -> str:
    """
        Convert single string sentence

        1. unicode to ascii
        2. adds a space between word and following punctuation
        3. removes all chars except a-Z, A-Z, , ".", "?", "!", ",","¿"
        4. removes leading/trailing blanks
        5. adds start/end tokens

        :param w: unicode string string
        :return: cleaned string
    """
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:
    #  https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except basic punctuation and alpha chars
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path: str, num_examples: int) -> List[List[str]]:
    """
        Create pairs of sentences

        1. Remove the accents
        2. Clean the sentences
        3. Return sentences grouped by language]

        :param path: path to input file
        :param num_examples: maximum number of examples
        :return: tuple containing two list of sequences,
                 one for each column in input file
    """
    # each line contains two columns separated by tab character
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    # split lines, preprocess phrases, get a tuple for each line
    sentence_pairs = [
                      [preprocess_sentence(w) for w in line.split('\t')]
                      for line in lines[:num_examples]]

    # rearrange to a tuple for each language
    return zip(*sentence_pairs)


def tokenize(lang: List[str]) -> Tuple[np.ndarray,
                                       tf.keras.preprocessing.text.Tokenizer]:
    """
        Fit a tokenizer on input list of sentences

        From words (string) to symbols (integer)

        :param lang: list of sentences of same language
        :return: a tuple of:
        a tensor:
            of size [NUMBER_OF_SENTENCES, SENTENCE_SIZE]
            contaning the vectorizations of the sentences
        a learned tokenizer
    """

    # create vanilla tokenizer
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')

    # learn tokenization procedure on given set of sentences
    lang_tokenizer.fit_on_texts(lang)

    # transforms sentences in sequences of integers
    # (sequences are of different lengths)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # pad sequences
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')

    return tensor, lang_tokenizer


def load_and_tokenize(
    path,
    num_examples=None) -> Tuple[np.ndarray,
                                np.ndarray,
                                tf.keras.preprocessing.text.Tokenizer,
                                tf.keras.preprocessing.text.Tokenizer]:
    """
        Load dataset, with preprocessing and tokenization

        :param path: path to input file
        :param num_examples: maximum number of examples
    """

    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)

    # tokenization
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
