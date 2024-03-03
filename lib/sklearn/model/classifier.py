import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RegressionClassifier(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            regressor: RegressorMixin,
            min: int = None,
            max: int = None
        ):
        self.regressor = regressor
        self.min = min
        self.max = max

    def fit(self, X: list[list[str]], y: np.array = None):
        self.regressor.fit(X, y)

        if self.min is None:
            self.min = np.min(X)

        if self.max is None:
            self.max = np.max(X)

        return self

    def predict(self, X: np.array):
        float_predictions = self.regressor.predict(X)
        label_predictions = np.clip(np.round(float_predictions), self.min, self.max).astype(np.int8)
        return label_predictions
