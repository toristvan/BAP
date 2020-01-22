import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

class BayesiaLinReg():
    '''
    Bayesian Linear Regression Model
    '''
    def __init__(self, degree=2, sigma2=25, c=None):
        self.degree = degree
        self.D = degree + 1
        self.sigma2 = sigma2
        if c is None:
            c = np.ones(self.D)
        self.C = np.diag(c)
        self.C_inv = np.linalg.inv(self.C)
        self.sigma2_C_inv = sigma2*self.C_inv
        self.y_hat = None
        self.y_pred = None
        self.F = 0
        
    def fit(self,x,y):
        self.N = len(y)
        self.x = x; self.y = y
        self.X = x[:,None] ** np.arange(0,self.D)[None]
        self.XtX = self.X.T @ self.X  
        self.Xty = self.X.T @ y
        
        self.Sigma_W = la.inv(self.XtX + self.sigma2_C_inv) 
        self.W = self.Sigma_W @ self.Xty
        
        self.y_hat = self.X @ self.W
        
        return self.y_hat

        
    def predict(self, x):
        self.m = x 
        self.T = x[:,None] ** np.arange(0,self.D)[None]
        
        y_pred = []
        y_sigma2_hat = []

        for x_star in self.T:
            Sw = la.inv(self.XtX + np.outer(x_star,x_star) + self.sigma2_C_inv)
            nom = x_star @ Sw @ self.Xty
            den = 1 - x_star @ Sw @ x_star
            y_pred.append(nom/den)
            y_sigma2_hat.append(self.sigma2/den)

        self.y_pred = np.array(y_pred)
        self.y_sigma2_hat = np.array(y_sigma2_hat).flatten()
        
        return self.y_pred, self.y_sigma2_hat
        
    def energy(self):
        F = self.N*np.log(2*np.pi*self.sigma2)
        log_det = np.prod(la.slogdet(self.C @ self.XtX + self.sigma2*np.eye(self.D)))
        F += log_det
        F += ( la.norm(self.y)**2 - self.Xty.T @  self.Sigma_W @ self.Xty)/self.sigma2
        self.F = F/2 
        return self.F
