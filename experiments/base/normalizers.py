"""
Normalizer classes for both equivariant and non-equivariant architectures
"""

import numpy as np


class Normalizer:
    """
    Per-feature z-score normalizer.

    If ``mask_nan`` is True, any columns that contain NaN in the fit data are
    treated as "optional": the NaN positions are zero-filled post-normalization
    and a presence indicator (1 = present, 0 = NaN) is appended for each such
    column. Output dim becomes ``n_input + (# columns with any NaN)``.
    """

    def __init__(self, mask_nan: bool = True):
        self._mask_nan = mask_nan
        self._nan_cols: np.ndarray | None = None

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std == 0] = 1
        self._nan_cols = (
            np.isnan(X).any(axis=0) if self._mask_nan
            else np.zeros(X.shape[1], dtype=bool)
        )

    def transform(self, X):
        norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)
        if self._nan_cols is not None and self._nan_cols.any():
            present = (~np.isnan(X[:, self._nan_cols])).astype(norm.dtype)
            norm = np.concatenate([norm, present], axis=1)
        return norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def n_mask_cols(self) -> int:
        """Number of presence-mask columns appended by ``transform``."""
        return int(self._nan_cols.sum()) if self._nan_cols is not None else 0


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
