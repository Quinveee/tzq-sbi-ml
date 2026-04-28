"""
Normalizer classes for both equivariant and non-equivariant architectures
"""

import numpy as np


class Normalizer:
    """Per-feature z-score normalizer; NaN entries are zero-filled."""

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return np.nan_to_num((X - self.mean) / self.std, nan=0.0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class EquivariantNormalizer:
    """
    Equivariant (wrt to Lorentz) normalizer
    matching LGATr implementations
    """

    def fit(self, X):
        self.std = np.nanstd(X)

    def transform(self, X):
        return np.nan_to_num(X / self.std, nan=0.0)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def get_normalizer(model_name):
    model2nom = {
        "lgatr": EquivariantNormalizer(),
        "LGATrCNF": EquivariantNormalizer(),
        "transformer_lloca": EquivariantNormalizer(),
    }
    return model2nom.get(model_name, Normalizer())
