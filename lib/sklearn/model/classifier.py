import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import Callable, List, Union


class RegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            regressor: RegressorMixin,
            min: int = None,
            max: int = None
        ):
        self.regressor = regressor
        self.min = min
        self.max = max

    def fit(self, X: List[List[str]], y: np.array = None):
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


class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            model,
            optimizer: str = None,
            loss: Union[str, Callable] = None,
            batch_size: int = 4,
            epochs: int = 100,
            metrics: List[str] = None
        ) -> None:
        from tensorflow.keras.models import Sequential

        self.model: Sequential = model
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = metrics

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        self.n_label = self.model.output_shape[-1]
        self.labels = []

    def fit(self, X: np.array, y: np.array = None, verbose: int = True, **kwargs):
        self.labels = sorted(list(set(y)))
        assert len(self.labels) == self.n_label

        one_hot_labels = np.eye(self.n_label)
        y = np.array([
            one_hot_labels[self.labels.index(yi)]
            for yi in y
        ])
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=int(verbose),
            **kwargs
        )
        return self

    def predict_proba(self, X: np.array):
        return self.model.predict(X)

    def predict(self, X: np.array):
        y_pred_proba = self.predict_proba(X)
        if self.n_label == 1:
            return (y_pred_proba > 0.5).astype(int)

        y_pred = np.array([
            self.labels[i]
            for i in np.argmax(y_pred_proba, axis=1)
        ])
        return y_pred
