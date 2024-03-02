import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from lib.constants import PROJECT_DIR


class TextTokenizer(TransformerMixin):
    def __init__(self, column: str) -> None:
        self.column = column

    def transform(self, X: pd.DataFrame, **kwargs):
        return X.assign(**{
            self.column: X[self.column].str.lower().map(word_tokenize)
        })


class WordsMapper(TransformerMixin):
    def __init__(self, column: str, mapper: dict[str, str]) -> None:
        self.column = column
        self.mapper = mapper

    def transform(self, X: pd.DataFrame, **kwargs):
        return X.assign(**{
            self.column: X[self.column].map(lambda words: [self.mapper.get(word, word) for word in words])
        })


class WordsFormalizer(WordsMapper):
    def __init__(self, column: str) -> None:
        collex = pd.read_table(
            PROJECT_DIR
            / 'data' / 'indo-collex'
            / 'informal-to-formal-dictionary.tsv'
        )
        mapper = {
            row['informal']: row['formal']
            for _, row in collex.iterrows()
        }

        super(WordsFormalizer, self).__init__(column, mapper)


class WordsFilter(TransformerMixin):
    def __init__(self, column: str, words: set[str], exclusive: bool = True) -> None:
        self.column = column
        self.words = words
        self.exclusive = exclusive

    def transform(self, X: pd.DataFrame, **kwargs):
        if self.exclusive:
            return X.assign(**{
                self.column: X[self.column].map(lambda words: [word for word in words if word not in self.words])
            })

        return X.assign(**{
            self.column: X[self.column].map(lambda words: [word for word in words if word in self.words])
        })


class StopWordsFilter(WordsFilter):
    def __init__(self, column: str) -> None:
        stop_words = set(stopwords.words('indonesian'))

        super(StopWordsFilter, self).__init__(column, stop_words)


class UnknownWordsFilter(WordsFilter):
    def __init__(self, column: str) -> None:
        with open(PROJECT_DIR / 'data' / 'sastrawi' / 'kata-dasar.txt') as f:
            known_words = set(f.read().splitlines())
        
        super(UnknownWordsFilter, self).__init__(column, known_words, False)


class WordsLemmatization(TransformerMixin):
    def __init__(self, column: str) -> None:
        self.column = column
        self.stemmer = StemmerFactory().create_stemmer()

    def transform(self, X: np.array, **kwargs):
        return X.assign(**{
            self.column: X[self.column].map(lambda words: [self.stemmer.stem(word) for word in words])
        })


def extract_unknown_words(words: np.array):
    known_words = UnknownWordsFilter('').words
    unknown_words = [word for word in words if word not in known_words]
    return unknown_words
