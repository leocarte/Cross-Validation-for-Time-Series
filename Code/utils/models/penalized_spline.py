from dataclasses import dataclass, field
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import SplineTransformer
from scipy.linalg import solve

@dataclass
class PenalizedSplineRegressor(BaseEstimator, RegressorMixin):

    param_value: float = 1.0  
    n_knots: int = 30
    degree: int = 3
    penalty_order: int = 2
    
    coef_: np.ndarray = field(init=False, repr=False)
    transformer: SplineTransformer = field(init=False, repr=False)

    def __post_init__(self):
        self.param_name = "lambda"

    def set_param(self, value):
        self.param_value = float(value)

    def _make_penalty_matrix(self, n_features):
        D = np.eye(n_features)
        for _ in range(self.penalty_order):
            D = np.diff(D, axis=0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return D.T @ D

    def fit(self, x, y):
        if isinstance(x, list): x = np.array(x)
        if isinstance(y, list): y = np.array(y)
        
        if x.ndim == 1: x = x.reshape(-1, 1)
        

        self.transformer = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=True,
            extrapolation="linear" 
        )
        X_basis = self.transformer.fit_transform(x)

        X_basis = np.nan_to_num(X_basis, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_features = X_basis.shape[1]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            XtX = X_basis.T @ X_basis
            Xty = X_basis.T @ y
            S = self._make_penalty_matrix(n_features)
            lhs = XtX + self.param_value * S + 1e-10 * np.eye(n_features)
        
        try:
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                self.coef_ = solve(lhs, Xty, assume_a='sym')
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(lhs, Xty, rcond=None)[0]
            
        return self

    def predict(self, x):
        if x.ndim == 1: x = x.reshape(-1, 1)
        X_basis = self.transformer.transform(x)
        X_basis = np.nan_to_num(X_basis, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            out = X_basis @ self.coef_
        return out