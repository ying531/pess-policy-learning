import numpy as np 

# from autograd import elementwise_grad as grad
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
#from statsmodels.api import add_constant
from scipy.optimize import minimize
from copy import deepcopy 

numpy2ri.activate()
pt = importr("policytree")
 


""" compute AIPW score """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# Gamma_mat is the T*K score matrix for Gamma(t,a), a=1,..,K
# where Gamma(t,a) = \hat\mu(x_t,a) + (Y_t - \hat\mu(x_t,a)) * 1(A_t=a) /e_t(a)
def PL_aipw_score(xxs, yobs, wws, exs, muxs = None):
    T, K = exs.shape
     
    if muxs is None:
        muxs = np.zeros((T, K))
    
    Gamma_mat = deepcopy(muxs)
    for w in range(K):
        Gamma_mat[wws==w,w] = muxs[wws==w,w] + (yobs[wws==w] - muxs[wws==w,w]) / exs[wws==w,w]
    
    return Gamma_mat


""" apply policy tree to new data """
def predict_ptree(ptree, xtest):
    w = pt.predict_policy_tree(ptree, np.atleast_2d(xtest))
    w = np.array(w, dtype = np.int_)-1
    return w

def eval_ptree(ptree, eval_data):  
    eval_xs = eval_data['xs']
    eval_ys = eval_data['ys']
    T_eval, K = eval_ys.shape
    eval_w = predict_ptree(ptree, eval_xs)
    eval_reward = np.zeros(T_eval)
    for k in range(K):
        eval_reward[eval_w==k] = eval_ys[eval_w==k, k]
    rw_mean = np.mean(eval_reward) 
    
    return eval_w, eval_reward, rw_mean



""" compute greedy policy tree """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# depth (default=2) is the depth of policy tree
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# ptree: a learned policy tree from the R package 
def PL_greedy(xxs, yobs, wws, exs, depth=3, muxs = None):
    gamma_mat = PL_aipw_score(xxs, yobs, wws, exs, muxs)
    ptree = pt.hybrid_policy_tree(X = xxs, Gamma = gamma_mat, depth=depth)
    
    return ptree

""" empirically evaluate a learned policy tree """
def emp_eval_ptree(ptree, eval_xs, eval_yobs, eval_ws, eval_exs, eval_muxs = None):
    T_eval, K = eval_exs.shape
    if eval_muxs is None:
        eval_muxs = np.zeros(eval_exs.shape)
    pred_ws = predict_ptree(ptree, eval_xs)
    eval_score = PL_aipw_score(eval_xs, eval_yobs, eval_ws, eval_exs, muxs = eval_muxs)
    for k in range(K):
        eval_score[(eval_ws == k) & (pred_ws != k), k] = 0
    avg_score = np.sum(eval_score) / T_eval
    return avg_score
