import nltk
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from typing import Any, Dict, List, Set, Tuple, Union

from lib.constants import PROJECT_DIR

nltk.download('punkt', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

INDONESIAN_ROOT_WORDS = set()
INDONESIAN_INFORMAL_TO_FORMAL_MAPPER = dict()


def flatten_tokens(tokens: List[str]) -> List[str]:
    _tokens = []
    for token in tokens:
        if ' ' in token:
            _tokens += token.split()
        else:
            _tokens.append(token)

    return _tokens


class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str], **kwargs):
        return [word_tokenize(text) for text in X]


class WordsMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapper: Dict[str, str]) -> None:
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]], **kwargs):
        X_mapped = [
            [self.mapper.get(word, word) for word in words]
            for words in X
        ]
        return [flatten_tokens(words) for words in X_mapped]


class WordsFormalizer(WordsMapper):
    def __init__(self) -> None:
        global INDONESIAN_INFORMAL_TO_FORMAL_MAPPER

        if len(INDONESIAN_INFORMAL_TO_FORMAL_MAPPER) == 0:
            collex = pd.read_table(
                PROJECT_DIR
                / 'data' / 'indo-collex'
                / 'informal-to-formal-dictionary.tsv'
            )
            INDONESIAN_INFORMAL_TO_FORMAL_MAPPER = {
                row['informal']: row['formal']
                for _, row in collex.iterrows()
            }

        super(WordsFormalizer, self).__init__(INDONESIAN_INFORMAL_TO_FORMAL_MAPPER)


class WordsFilter(BaseEstimator, TransformerMixin):
    def __init__(self, words: Set[str], exclusive: bool = True) -> None:
        self.words = words
        self.exclusive = exclusive

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]], **kwargs):
        if self.exclusive:
            return [
                [word for word in words if word not in self.words]
                for words in X
            ]

        return [
            [word for word in words if word in self.words]
            for words in X
        ]


class StopWordsFilter(WordsFilter):
    def __init__(self) -> None:
        stop_words = set(stopwords.words('indonesian'))

        super(StopWordsFilter, self).__init__(stop_words)


class UnknownWordsFilter(WordsFilter):
    def __init__(self) -> None:
        global INDONESIAN_ROOT_WORDS

        if len(INDONESIAN_ROOT_WORDS) == 0:
            with open(PROJECT_DIR / 'data' / 'sastrawi' / 'kata-dasar.txt') as f:
                INDONESIAN_ROOT_WORDS = set(f.read().splitlines())

        super(UnknownWordsFilter, self).__init__(INDONESIAN_ROOT_WORDS, False)


class SpecialCharacterFilter(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.pattern = r'[\W_]+'

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]], **kwargs):
        return [
            [
                word for word in words
                if len(word) > 0 and re.match(self.pattern, word) is None
            ]
            for words in X
        ]


class WordsLemmatization(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.stemmer = StemmerFactory().create_stemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]], **kwargs):
        return [
            [self.stemmer.stem(word) for word in words]
            for words in X
        ]


class TokenToTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[List[str]], **kwargs):
        return [' '.join(words) for words in X]


class TokenSequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_words: int) -> None:
        from tensorflow.keras.preprocessing.text import Tokenizer

        self.max_words = max_words
        self.tokenizer = Tokenizer(num_words=max_words)

    def fit(self, X: List[str], y=None):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X: List[str], **kwargs):
        sequences = self.tokenizer.texts_to_sequences(X)
        return sequences


def extract_unknown_words(words: List[str]):
    known_words = UnknownWordsFilter().words
    unknown_words = [word for word in words if word not in known_words]
    return unknown_words


def split_sequences(
        X: List[List[Any]],
        max_len: int,
        y: np.array = None,
        min_len: int = None,
        padding: str = 'pre',
        truncating: str = 'pre',
        padding_value: float = 0
    ) -> Union[Tuple[np.array, np.array], np.array]:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    min_len = min_len if min_len is not None else max_len

    X_, y_ = [], []
    for i in range(len(X)):
        if len(X[i]) <= min_len:
            sequences = [X[i]]
        else:
            sequences = [
                X[i][np.max([0, j - max_len]):j]
                for j in range(min_len, len(X[i]))
            ]

        sequences_padded = pad_sequences(
            sequences,
            maxlen=max_len,
            padding=padding,
            truncating=truncating,
            value=padding_value
        )
        X_.extend(sequences_padded)
        if y is not None:
            y_.extend([y[i]] * len(sequences_padded))

    if y is not None:
        return np.array(X_), np.array(y_)

    return np.array(X_)
