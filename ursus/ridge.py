import numpy as np


class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True, fit_offset=True):
        self.k = alpha
        self.fit_intercept = fit_intercept
        self.fit_offset = fit_offset
    
    def fit(self, X, y):
        k = self.k
        if self.fit_offset:
            self.intercept_ = np.min(y)
            _y = y - self.intercept_
        elif self.fit_intercept:
            inter = np.ones_like(y).reshape((-1, 1))
            X = np.hstack((X, inter))
            if isinstance(k, np.ndarray):
                k = np.append(k, 0)
            _y = y
        else: 
            _y = y
        I = np.eye(X.shape[1])
        self.coef_ = np.dot(np.linalg.inv(np.dot(X.T, X) + k * I), np.dot(X.T, _y))
        if self.fit_offset:
            pass
        elif self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0

    def predict(self, X):
        y_hat = np.sum(self.coef_ * X, axis=1) + self.intercept_
        return y_hat
