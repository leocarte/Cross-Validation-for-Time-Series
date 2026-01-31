import numpy as np
from scipy.linalg import solve, cholesky, cho_solve, inv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import SplineTransformer

class FastNCVPenalizedSpline(BaseEstimator, RegressorMixin):

    
    def __init__(self, n_knots=20, degree=3, penalty_order=2):
        self.n_knots = n_knots
        self.degree = degree
        self.penalty_order = penalty_order
        self.spline_trans = None
        self.S = None 
        self.X = None 
        self.y = None
        self.coef_ = None
        self.H_inv = None 
        self.param_name = "lambda"
        self.param_value = None


    def _make_penalty_matrix(self, n_features):

        D = np.eye(n_features)
        for _ in range(self.penalty_order):
            D = np.diff(D, axis=0)
        return D.T @ D


    def set_param(self, value):
        self.param_value = float(value)


    def fit(self, X, y):

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        
        self.spline_trans = SplineTransformer(
            n_knots=self.n_knots, 
            degree=self.degree, 
            include_bias=True
        )
        self.X = self.spline_trans.fit_transform(X)
        
        self.S = self._make_penalty_matrix(self.X.shape[1])
        return self


    def _fit_for_lambda(self, lam):

        XtX = self.X.T @ self.X
        H = 2 * (XtX + lam * self.S)
        
        Xty = 2 * (self.X.T @ self.y)
        
        try:
            self.H_inv = inv(H) 
            self.coef_ = self.H_inv @ Xty
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(H, Xty, rcond=None)[0]
            self.H_inv = np.linalg.pinv(H)


    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_basis = self.spline_trans.transform(X)
        X_basis = np.nan_to_num(X_basis, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return X_basis @ self.coef_


    def compute_ncv_score(self, lam, l_buffer):

        self._fit_for_lambda(lam)
        
        n_samples, n_features = self.X.shape
        y_preds_ncv = np.zeros(n_samples)
        
        residuals = self.y - self.X @ self.coef_
        

        for i in range(n_samples):
            
            start = max(0, i - l_buffer)
            end = min(n_samples, i + l_buffer + 1)
            alpha_indices = np.arange(start, end)
            
            X_alpha = self.X[alpha_indices]
            r_alpha = residuals[alpha_indices]
            

            g_alpha = -2 * (X_alpha.T @ r_alpha)
            

            VAinv = X_alpha @ self.H_inv
            

            k = len(alpha_indices)
            inner_matrix = (0.5 * np.eye(k)) - (VAinv @ X_alpha.T)
            

            try:
                inner_inv = inv(inner_matrix)
            except np.linalg.LinAlgError:
                inner_inv = np.linalg.pinv(inner_matrix)
            

            update_term = (self.H_inv @ X_alpha.T) @ inner_inv @ VAinv
            
            H_rem_inv = self.H_inv + update_term 
            

            delta_beta = H_rem_inv @ g_alpha
            
            beta_ncv = self.coef_ + delta_beta
            

            y_preds_ncv[i] = self.X[i] @ beta_ncv
            
        return np.mean((self.y - y_preds_ncv)**2)