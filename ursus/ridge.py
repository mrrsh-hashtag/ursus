import numpy as np


class HNRidge:
    def __init__(self, alpha=1.0, fit_intercept=False, scale=True):
        self.do_scale = scale
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        if self.fit_intercept:
            inter = np.ones_like(y).reshape((-1, 1))
            X = np.hstack((X, inter))
            if isinstance(k, np.ndarray):
                k = np.append(k, 0)
            _y = y
        elif self.do_scale:
            self.x_mean = np.mean(X, axis=0)
            self.x_std = np.std(X, axis=0)
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)

            X = (X - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std
            _y = y
            # print("std X")
            # print(np.std(X, axis=0))
            # print("std y")
            # print(np.std(y))
        else: 
            _y = y
        I = np.eye(X.shape[1])
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ _y
        
        y_hat = np.sum(self.coef_ * X, axis=1)
        if self.fit_intercept:
            self.intercept_ = self.coef_[-1]
            self.coef_ = self.coef_[:-1]
        else:
            self.intercept_ = 0
        
        # sst = np.sum(np.power(y - np.mean(y), 2)) / _y.shape[0]
        # print("sst")
        # print(sst)
        s2 = np.sum(np.power(_y - y_hat, 2)) / _y.shape[0]
        # print("s2")
        # print(s2)
        # r2 = 1 - s2/sst
        # print("r2")
        # print(r2)

        # print("beta")
        # print(self.coef_)
        # print("intercept")
        # print(self.intercept_)

        ##### New Method
        LAMDA, GAMMA = np.linalg.eig(X.T @ X)
        # print("LAMDA eigenvalues")
        # print(LAMDA)
        # print("GAMMA eigenvectors")
        # print(GAMMA)

        X_GAMMA = X @ GAMMA
        # print("std X_GAMMA")
        # print(np.std(X_GAMMA, axis=0))
        # print(np.sum(np.std(X_GAMMA, axis=0)))

        
        LAMDA_diag = np.diag(LAMDA)
        LAMBDA_inv = np.linalg.inv(LAMDA_diag)
        b_star = LAMBDA_inv @ GAMMA.T @ X.T @ _y
        
        # Should end when b_star and b_star_new is close -> convergence.
        for _ in range(3):
            K = np.zeros_like(LAMDA_diag)
            for i in range(LAMDA.size):
                K[i, i] = s2 / (LAMDA[i] * np.power(b_star[i], 2))
            b_star = np.linalg.inv(LAMDA_diag + K) @ GAMMA.T @ X.T @ _y
            
        b_rr = GAMMA @ b_star

        if not self.do_scale:
            self.intercept_ = b_rr[-1]
            self.coef_ = b_rr[:-1]
        else:
            self.coef_ = b_rr * self.y_std / self.x_std
            self.intercept_ = self.y_mean - np.sum(self.coef_ * self.x_mean)
            # print(self.coef_)
            # print(self.intercept_)


    def predict(self, X):
        y_hat = np.sum(self.coef_ * X, axis=1) + self.intercept_
        return y_hat
