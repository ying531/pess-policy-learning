from utils.experiment import run_experiment
from utils.thompson import *
from utils.dgp import *
from algs.ptree import *
from algs.pess import * 

import numpy as np 
from copy import deepcopy 
import argparse 
import os

parser = argparse.ArgumentParser() 
parser.add_argument('--setting', type = int, default = 1, help = 'setting number for decay rate')
parser.add_argument('--T', type = int, default = 1000, help = 'total sample size')   
parser.add_argument('--beta', type = int, default = 0, help = 'penalty parameter')   
parser.add_argument('--scenario', type = int, default = 1, help = 'setting in the paper') 

args = parser.parse_args() 
setting = args.setting
p = 2
T = args.T
beta_id = args.beta
scenario = args.scenario

depth = 5   

SAVE_PATH = "./results/synthetic_dt/"
 
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
 
  
# T = np.sum(batch_sizes)
T_eval = 100000
# BETA_LIST=  [0.1, 0.2, 0.5, 1, 5, 10]
BETA_LIST = [0.1, 0.2, 0.5, 1, 5, 10, 0.001, 0.01, 0.0001]
beta = BETA_LIST[beta_id-1]
Nrep = 200
K = 10
p = 2

if setting == 0: # pure exploration
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0) 
    if_floor = True
    floor_start = 0.001
    floor_decay = 0
    
if setting > 0:
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = setting/10
    

eval_data = dgp_eval.sample_data(T_eval)  


res_df  = pd.DataFrame()
for ii_seed in range(Nrep):
    np.random.seed(ii_seed)

    """ sample all x and y """
    data = dgp.sample_data(T)
    xs = data['xs']
    ys = data['ys']

    """ collect adaptive data with decaying probability """
    w_opt = dgp.compute_optimal(xs)
    ps = np.zeros((T, K))
    ws = np.zeros(T)
    yobs = np.zeros(T)
    
    if scenario == 1: # setting 1 in paper
        for tt in range(T):
            ps[tt, :] = floor_start / (1+tt)**floor_decay 
            ps[tt, w_opt[tt]] = 1 -  (K-1) * floor_start / (1+tt)**floor_decay 
            ws[tt] = np.random.choice(K, p = ps[tt,:])
            yobs[tt] = ys[tt, int(ws[tt])]
            
    if scenario == 3: # setting 3 in paper
        for tt in range(T):
            if w_opt[tt] == 9:
                ps[tt, :] = np.array([0.2 - floor_start / (1+tt) ** floor_decay] * 5 + [floor_start / (1+tt) ** floor_decay] * 5)
            else:
                ps[tt, :] = np.array([floor_start / (1+tt) ** floor_decay] * 5 + [0.2 - floor_start / (1+tt) ** floor_decay] * 5)
            ws[tt] = np.random.choice(K, p = ps[tt,:])
            yobs[tt] = ys[tt, int(ws[tt])]
            
    if scenario == 2: # setting 2 in paper
        if_opt = np.zeros(T)
        Nswap = 5
        for bb in range(Nswap):
            if bb % 2 == 0:
                if_opt[range(int(bb * T / Nswap), int((bb+1) * T / Nswap))] = 1

        for tt in range(T):
            if if_opt[tt] == 1:
                if w_opt[tt] == 0:
                    ps[tt, :] = np.array([0.2 - floor_start / (1+tt) ** floor_decay] * 5 + [floor_start / (1+tt) ** floor_decay] * 5)
                else:
                    ps[tt, :] = np.array([floor_start / (1+tt) ** floor_decay] * 5 + [0.2 - floor_start / (1+tt) ** floor_decay] * 5)
            
            else:
                if w_opt[tt] == 9:
                    ps[tt, :] = np.array([0.2 - floor_start / (1+tt) ** floor_decay] * 5 + [floor_start / (1+tt) ** floor_decay] * 5)
                else:
                    ps[tt, :] = np.array([floor_start / (1+tt) ** floor_decay] * 5 + [0.2 - floor_start / (1+tt) ** floor_decay] * 5)
            
            ws[tt] = np.random.choice(K, p = ps[tt,:])
            yobs[tt] = ys[tt, int(ws[tt])]

    
    """ greedy policy tree """
    # if ii_seed == 0:
    #     print("-- start running greedy ptree ...")
    greedy_ptree = PL_greedy(xs, yobs, ws, ps, depth=depth)

    # ====================================================
    """ # run pessimism with the given beta without CV """ 

    """ pessimistic policy tree """
    if ii_seed == 0:
        print("-- start running pess tree ...") 
    pess_ptree, _ = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                            beta = beta, depth = depth, 
                            lower_bound = 0.0001, muxs = None, verbose=False) 
    
    """  evaluation  """
    greedy_w, greedy_eval_reward, rw_greedy = eval_ptree(greedy_ptree, eval_data)
    pess_w, pess_eval_reward, rw_pess = eval_ptree(pess_ptree, eval_data)  
    
    
    # ====================================================
    """ run pesssimism with linear function approximation """ 
    hat_theta_list = []
    hat_ilambda_list = []
    for w in range(K):
        xs_w = xs[ws==w,:]
        ys_w = ys[ws==w,w]
        xs_w = np.hstack((np.ones((xs_w.shape[0],1)), xs_w))
        # covariance matrix 
        hat_Lambda_w = np.linalg.inv((xs_w.transpose() @ xs_w) + np.identity(xs_w.shape[1]))
        hat_ilambda_list.append(hat_Lambda_w)
        # ridge regression 
        hat_theta_w = hat_Lambda_w @ (xs_w.transpose () @ ys_w)
        hat_theta_list.append(hat_theta_w)

    """ evaluate """
    # compute pessimistic policy 
    xs_eval = eval_data['xs']
    ys_eval = eval_data['ys']
    val_eval = np.zeros((xs_eval.shape[0], K))

    xs_eval = np.hstack((np.ones((xs_eval.shape[0],1)), xs_eval))

    for w in range(K): 
        val_eval[:,w] = xs_eval @ hat_theta_list[w] - beta * np.sqrt(np.array([xs_eval[j,:] @ hat_ilambda_list[w] @ xs_eval[j,:].T for j in range(xs_eval.shape[0])]))

    linear_w = np.argmax(val_eval, axis=1)
    rw_linear = np.mean([ys_eval[j, linear_w[j]] for j in range(xs_eval.shape[0])])


    res_df = pd.concat([res_df, 
                        pd.DataFrame({
                            "eval_reward": [rw_greedy, rw_pess, rw_linear], 
                            "method_param": ['greedy', 'pess_'+str(beta), 'lin'+str(beta)],
                            "method": ['greedy', 'pessimism', 'pess_linear'],  
                            "T": T,  
                            "beta": [beta]*3,  
                            "seed": ii_seed})], axis = 0)
     

res_df.to_csv(SAVE_PATH + "setting_" + str(setting) + "_scenario_" + str(scenario) + "_T_" + str(T) + "_beta_" + str(beta_id) + ".csv")
 
