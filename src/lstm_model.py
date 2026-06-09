from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LSTMClassifier:
    """sklearn-compatible GRU wrapper for time-series binary classification.

    Internally builds a recurrent model via tensorflow.keras and converts
    tabular 2-D feature DataFrames into 3-D sequences ``(samples, lookback,
    features)`` so it slots directly into the existing pipeline.

    Uses a single GRU layer (fewer parameters than LSTM — important on small
    datasets).  The raw sigmoid probabilities are returned directly; the
    downstream position-policy layer handles scaling / ranking.

    Parameters
    ----------
    lookback : int
        Number of past days used as the input sequence window.
    gru_units : int
        Units in the GRU layer.
    dropout : float
        Dropout rate applied after the GRU layer.
    dense_units : int
        Units in the intermediate Dense layer before the output head.
    learning_rate : float
        Adam learning rate.
    batch_size : int
        Training batch size.
    epochs : int
        Maximum training epochs (early-stopping usually cuts this short).
    early_stopping_patience : int
        Epochs without validation-loss improvement before stopping.
    validation_split : float
        Fraction of training data held out for validation / early-stopping.
    random_state : int
        Seed for reproducibility.
    verbose : int
        Keras verbosity level (0 = silent, 1 = progress bar, 2 = one line per
        epoch).
    """

    def __init__(
        self,
        lookback: int = 20,
        gru_units: int = 48,
        dropout: float = 0.25,
        dense_units: int = 16,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.lookback = lookback
        self.gru_units = gru_units
        self.dropout = dropout
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------
    # Public sklearn-compatible API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Train the GRU on tabular features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) — binary labels (0 / 1).

        Returns
        -------
        self
        """
        X_arr = self._to_array(X)
        y_arr = self._to_array_1d(y)
        self._n_features = X_arr.shape[1]

        X_seq, y_seq = self._make_sequences(X_arr, y_arr)

        self._set_seeds()
        self._model = self._build_model()
        self._model.fit(
            X_seq,
            y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=self._build_callbacks(),
            verbose=self.verbose,
        )
        return self

    def predict_proba(self, X):
        """Return class probabilities for every sample.

        Samples without a full lookback window return ``[0.5, 0.5]``.
        """
        X_arr = self._to_array(X)
        n_samples = X_arr.shape[0]
        proba = np.full((n_samples, 2), 0.5, dtype=np.float64)

        if n_samples < self.lookback or self._model is None:
            return proba

        X_seq = self._make_sequence_windows(X_arr)
        prob_up = self._model.predict(X_seq, batch_size=self.batch_size, verbose=0).ravel()

        start = self.lookback - 1
        proba[start:, 1] = prob_up
        proba[start:, 0] = 1.0 - prob_up
        return proba

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(X):
        if hasattr(X, "values"):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _to_array_1d(y):
        if hasattr(y, "values"):
            return y.values.astype(np.float32).ravel()
        return np.asarray(y, dtype=np.float32).ravel()

    def _make_sequences(self, X_arr, y_arr):
        n_samples = X_arr.shape[0]
        n_features = X_arr.shape[1]
        seq_len = n_samples - self.lookback + 1

        X_seq = np.zeros((seq_len, self.lookback, n_features), dtype=np.float32)
        y_seq = np.zeros(seq_len, dtype=np.float32)

        for i in range(seq_len):
            X_seq[i] = X_arr[i : i + self.lookback]
            y_seq[i] = y_arr[i + self.lookback - 1]

        return X_seq, y_seq

    def _make_sequence_windows(self, X_arr):
        n_samples = X_arr.shape[0]
        n_features = X_arr.shape[1]
        seq_len = n_samples - self.lookback + 1

        X_seq = np.zeros((seq_len, self.lookback, n_features), dtype=np.float32)
        for i in range(seq_len):
            X_seq[i] = X_arr[i : i + self.lookback]
        return X_seq

    def _set_seeds(self) -> None:
        import random as _random

        _random.seed(self.random_state)
        np.random.seed(self.random_state)

        import tensorflow as tf

        tf.random.set_seed(self.random_state)

    def _build_model(self):
        from tensorflow.keras.layers import GRU, Dropout, Dense, Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        if self._n_features is None:
            raise RuntimeError("_n_features not set — call fit() first")

        model = Sequential(
            [
                Input(shape=(self.lookback, self._n_features)),
                GRU(self.gru_units),
                Dropout(self.dropout),
                Dense(self.dense_units, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
        )
        return model

    def _build_callbacks(self):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.verbose,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(3, self.early_stopping_patience // 3),
                min_lr=1e-6,
                verbose=self.verbose,
            ),
        ]
