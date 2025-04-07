import numpy as np 
import pandas as pd
from utils.dgp import MultiQuad
from utils.compute import apply_floor
from sklearn.linear_model import RidgeCV

class LinTS:

    '''
    Implement the linear TS algorithm 
    K: number of arms 
    p: feature dimension 
    floor_start: K-dimensional vector for lower bounded exploration floor_start * t^{-floor_dacay}
    floor_decay: K-dimensional vector for lower bounded exploration, can be None for no bounds
    num_mc: number of Monte Carlo samples
    DGP: data generating object
    '''

    def __init__(self, K, p, DGP, num_mc = 100, 
                 if_floor = False, floor_start = None, floor_decay = None):
        self.num_mc = num_mc
        self.K = K
        self.p = p
        self.floor_start = None
        self.floor_decay = None 
        self.mu = [np.zeros((p+1))] * K # sample mean 
        self.V = [np.zeros((p+1, p+1))] * K # sample covariance
        self.X = [[] for _ in range(K)] # observed features
        self.y = [[] for _ in range(K)] # observed outcomes 
        self.A = []
        self.DGP = DGP
        self.ps = None
        self.if_floor = if_floor
        self.ws = None 
        if if_floor:
            self.floor_start = floor_start
            self.floor_decay = floor_decay
    
    def initialize_ps(self, T):
        self.ps = np.empty((T, self.K))
    
    def initialize_w(self, T):
        self.ws = np.empty(T)
    
    def set_floor(self, if_floor, floor_start, floor_decay):
        self.if_floor = if_floor
        self.floor_start = floor_start
        self.floor_decay = floor_decay
         
    def _update_TS(self, x, y, k):
        self.X[k].extend(x)
        # print(len(self.X[k]))
        self.y[k].extend(y)
        # print(len(self.y[k]))
        regr = RidgeCV(alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]).fit(self.X[k], self.y[k])
        yhat = regr.predict(self.X[k]) 
        self.mu[k] = [regr.intercept_, *list(regr.coef_)]
        # print(self.mu[k])
        X = np.concatenate([np.ones((len(self.X[k]), 1)), self.X[k]], axis=1) 
        B = np.matmul(X.T, X) + regr.alpha_ * np.eye(self.p+1) 
        self.V[k] = np.mean((self.y[k] - yhat) ** 2) * np.linalg.inv(B) 

    # === update TS parameters based on new observations
    # xss: features for the new observations
    # wss: actions for the new obs
    # yss: rewards for the new obs
    def update_TS(self, xss, wss, yss):
        for k in range(self.K):
            # print("k:"+str(k)+", x:"+str(xss[wss==k].shape))
            self._update_TS(xss[wss==k], yss[wss==k], k)

    # add_floor is T*K matrix of additional floor 
    def draw_TS_one_batch(self, xs, start, end, current_t, add_floor = None): 
        T, p = xs.shape 
        xt = np.concatenate([np.ones((T, 1)), xs], axis=1) 

        # initialize propensity score matrix
        if self.ps is None:
            self.ps = np.empty((T, self.K))

        # Thompson sampling
        coeff = np.empty((self.K, self.num_mc, self.p+1))
        for k in range(self.K):
            coeff[k] = np.random.multivariate_normal(self.mu[k], self.V[k], size = self.num_mc) 
        draws = np.matmul(coeff, xt.T)  
        for s in np.arange(start, end):
            self.ps[s, :] = np.bincount(np.argmax(draws[:, :, s], axis = 0), minlength = self.K) / self.num_mc
            print(self.ps[s, :])
            # if specified to apply floor, compute sampling floors  
            if add_floor is not None:
                if self.if_floor:
                    psmin = np.maximum(self.floor_start / current_t ** self.floor_decay, add_floor[s,:])
                    self.ps[s, :] = apply_floor(self.ps[s, :], psmin) 
                else:
                    self.ps[s, :] = apply_floor(self.ps[s, :], add_floor[s,:])  
            else:
                if self.if_floor:
                    psmin = self.floor_start / current_t ** self.floor_decay 
                    self.ps[s, :] = apply_floor(self.ps[s, :], psmin)
            
        w = [np.random.choice(self.K, p = self.ps[t]) for t in range(start, end)]
        self.ws[range(start,end)] = w 
            
        return w, self.ps
 
        


