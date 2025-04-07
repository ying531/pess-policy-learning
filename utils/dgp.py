import numpy as np 
import pandas as pd


class TwoQuad:

    '''
    Y(k) = 1 - alpha_k + alpha_k * x1^2 + epsilon 
    K: number of arms 
    alpha: the vector of alpha_k's
    p: dimension of X where X_i ~ U[-2,2]
    sigma: noise N(0, sigma^2)
    '''
    def __init__(self, p, sigma = 0.1, alpha = None): 
        self.p = p  
        self.sigma = sigma
        if alpha is None:
            self.alpha = np.array([-1,1])
        else:
            self.alpha = alpha
        self.mu = 1 - 2*self.alpha/3


    def sample_single_arm(self, n, k):
        X = np.random.uniform(-2, 2, n)
        return 1 - self.alpha[k] + self.alpha[k] * X**2 + np.random.normal(n) * self.sigma

    def sample_all_arm(self, n): 
        X = []
        for k in range(2):
            X0 = np.random.uniform(-2, 2, n)
            X.append((1 - self.alpha[k] + self.alpha[k] * X0**2 + np.random.normal(n) * self.sigma).tolist())
        return X

    def sample_data(self, n):
        xs = np.random.uniform(-2, 2, size = (n, self.p))
        ys = np.zeros((n, 2))
        muxs = np.zeros((n, 2))
        for k in range(2):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2)
            ys[:,k] = muxs[:, k] + np.random.normal(size=n) * self.sigma
        data = dict(xs = xs, ys = ys, muxs = muxs)
        return data
    
    def sample_data_v2(self, n):
        xs = np.random.uniform(-2, 2, size = (n, self.p))
        ys = np.zeros((n, 2))
        muxs = np.zeros((n, 2))
        for k in range(2):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2) + xs[:,1]/2
            ys[:,k] = muxs[:, k] + np.random.normal(size=n) * self.sigma
        data = dict(xs = xs, ys = ys, muxs = muxs)
        return data
    
    
    def compute_optimal(self, xs):
        n, p = xs.shape
        muxs = np.zeros((n, 2))
        for k in range(2):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2)
        opt_arms = [np.argmax(muxs[i,:]) for i in range(n)]
        return opt_arms




class MultiQuad:

    '''
    Y(k) = 1 - alpha_k + alpha_k * x1^2 + epsilon 
    K: number of arms 
    alpha: the vector of alpha_k's
    p: dimension of X where X_i ~ U[-2,2]
    sigma: noise N(0, sigma^2)
    '''
    def __init__(self, p, K = 2, sigma = 1, alpha = None): 
        self.p = p  
        self.sigma = sigma
        self.K = K
        if alpha is None:
            self.alpha = np.linspace(-1, 1, num = K)
        else:
            self.alpha = alpha
        self.mu = 1 - 2*self.alpha/3


    def sample_single_arm(self, n, k):
        X = np.random.uniform(-2, 2, n)
        return 1 - self.alpha[k] + self.alpha[k] * X**2 + np.random.normal(n) * self.sigma

    def sample_all_arm(self, n): 
        X = []
        for k in range(self.K):
            X0 = np.random.uniform(-2, 2, n)
            X.append((1 - self.alpha[k] + self.alpha[k] * X0**2 + np.random.normal(n) * self.sigma).tolist())
        return X

    def sample_data(self, n):
        xs = np.random.uniform(-2, 2, size = (n, self.p))
        ys = np.zeros((n, self.K))
        muxs = np.zeros((n, self.K))
        for k in range(self.K):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2)
            ys[:,k] = muxs[:, k] + np.random.normal(size=n) * self.sigma
        data = dict(xs = xs, ys = ys, muxs = muxs)
        return data
    
    def sample_data_v2(self, n):
        xs = np.random.uniform(-2, 2, size = (n, self.p))
        ys = np.zeros((n, self.K))
        muxs = np.zeros((n, self.K))
        for k in range(self.K):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2) + xs[:,1]**2/8
            ys[:,k] = muxs[:, k] + np.random.normal(size=n) * self.sigma
        data = dict(xs = xs, ys = ys, muxs = muxs)
        return data
    
    
    def compute_optimal(self, xs):
        n, p = xs.shape
        muxs = np.zeros((n, self.K))
        for k in range(self.K):
            muxs[:, k] = 1 - self.alpha[k] + self.alpha[k] * (xs[:,0] ** 2)
        opt_arms = [np.argmax(muxs[i,:]) for i in range(n)]
        return opt_arms
     



class MultiLinear:

    '''
    Y(k) = 1 - alpha_k/2 + alpha_k * x1 /2 -  alpha_k * x2 + epsilon 
    K: number of arms 
    alpha: the vector of alpha_k's
    p: dimension of X where X_i ~ U[-2,2]
    sigma: noise N(0, sigma^2)
    '''
    def __init__(self, p, K = 2, sigma = 1, alpha = None): 
        self.p = p  
        self.sigma = sigma
        self.K = K
        if alpha is None:
            self.alpha = np.linspace(-1, 1, num = K)
        else:
            self.alpha = alpha
        self.mu = 1 -  self.alpha/2


    # def sample_single_arm(self, n, k):
    #     X = np.random.uniform(-2, 2, n)
    #     return 1 - self.alpha[k] + self.alpha[k] * X**2 + np.random.normal(n) * self.sigma

    def sample_all_arm(self, n): 
        X = []
        for k in range(self.K):
            X0 = np.random.uniform(-2, 2, n)
            X1 = np.random.uniform(-2, 2, n)
            X.append((1 - self.alpha[k]/2 + self.alpha[k] * X0/2 - self.alpha[k]* X1+ np.random.normal(n) * self.sigma).tolist())
        return X

    def sample_data(self, n):
        xs = np.random.uniform(-2, 2, size = (n, self.p))
        ys = np.zeros((n, self.K))
        muxs = np.zeros((n, self.K))
        for k in range(self.K):
            muxs[:, k] = 1 - self.alpha[k]/2 + self.alpha[k] * xs[:,0]/2 -  self.alpha[k] * xs[:,1]
            ys[:,k] = muxs[:, k] + np.random.normal(size=n) * self.sigma
        data = dict(xs = xs, ys = ys, muxs = muxs)
        return data 
    
    
    def compute_optimal(self, xs):
        n, p = xs.shape
        muxs = np.zeros((n, self.K))
        for k in range(self.K):
            muxs[:, k] = 1 - self.alpha[k]/2 + self.alpha[k] * xs[:,0] /2 -  self.alpha[k] * xs[:,1]
        opt_arms = [np.argmax(muxs[i,:]) for i in range(n)]
        return opt_arms





class MABandit:

    '''
    No covariates, multi-arm bandits
    '''

    def __init__(self, mu = None, sigma = 0.1, ps = None): 
        self.sigma = sigma
        self.ps = ps
        if mu is None:
            self.mu = np.linspace(-1, 1, num=5)
        else:
            self.mu = mu 


    def sample_data(self, n):
        ys = np.zeros((n, 5))
        for k in range(5):
            ys[:,k] = self.mu[k] + np.random.normal(size=n) * self.sigma
        data = dict(ys = ys)
        return data
 
    def set_ps(self, ps):
        self.ps = ps

    def sample_arms(self, n):
        ws = np.random.choice(range(5), size=n, p=self.ps)
        return ws 
    