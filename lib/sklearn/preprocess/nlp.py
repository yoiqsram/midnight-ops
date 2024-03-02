import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from lib.constants import PROJECT_DIR

nltk.download('punkt', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

INDONESIAN_ROOT_WORDS = set()
INDONESIAN_INFORMAL_TO_FORMAL_MAPPER = dict()


def flatten_tokens(tokens: list[str]) -> list[str]:
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

    def transform(self, X: list[str], **kwargs):
        return [word_tokenize(text) for text in X]


class WordsMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapper: dict[str, str]) -> None:
        self.mapper = mapper

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[list[str]], **kwargs):
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
    def __init__(self, words: set[str], exclusive: bool = True) -> None:
        self.words = words
        self.exclusive = exclusive

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[list[str]], **kwargs):
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

    def transform(self, X: list[list[str]], **kwargs):
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

    def transform(self, X: list[list[str]], **kwargs):
        return [
            [self.stemmer.stem(word) for word in words]
            for words in X
        ]


class TokenToTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: list[list[str]], **kwargs):
        return [' '.join(words) for words in X]


def extract_unknown_words(words: list[str]):
    known_words = UnknownWordsFilter().words
    unknown_words = [word for word in words if word not in known_words]
    return unknown_words
