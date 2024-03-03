import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class RegressionExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, regressor: RegressorMixin):
        self.regressor = regressor

    def fit(self, X: list[list[str]], y: np.array = None):
        self.regressor.fit(X, y)
        return self

    def transform(self, X: np.array):
        float_predictions = self.regressor.predict(X)
        return float_predictions.reshape([-1, 1])
