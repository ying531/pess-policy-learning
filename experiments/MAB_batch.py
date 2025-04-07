from utils.experiment import run_experiment
from utils.thompson import *
from utils.dgp import *

import numpy as np 
from copy import deepcopy 
import argparse 

parser = argparse.ArgumentParser() 
parser.add_argument('--setting', type = int, default = 1, help = 'simulation setting') 

args = parser.parse_args() 
setting = args.setting 
# ======================
# Setting 1: optimal arm is well explored
# saved to "./results_MAB_optimal.csv"
if setting == 1: 
    abs_mu = [0, 0.05, 0.01, 0, -0.01]
    ps = [0.07, 0.9, 0.01, 0.01, 0.01]
    SAVE_PATH = './results/results_MAB_optimal.csv'
    SAVE_PATH_FREQ = './results/freq_MAB_optimal.csv'
# ======================

# ======================
# Setting 2: optimal arm is under-explored, but near-optimal is well-explored
# saved to "./results/results_MAB_near-opt.csv"
if setting == 2: 
    abs_mu = [0, 0.05, 0.04, 0.01, -0.01]
    ps = [0.07, 0.1, 0.8, 0.02, 0.01]
    SAVE_PATH = './results/results_MAB_near-opt.csv'
    SAVE_PATH_FREQ = './results/freq_MAB_near-opt.csv'
# ====================== 

# ======================
# Setting 3: uniform exploration
# saved to "./results/results_MAB_uniform.csv"
if setting == 3: 
    abs_mu = [0, 0.05, 0.03, 0.01, -0.01]
    ps = [0.2] * 5
    SAVE_PATH = './results/results_MAB_uniform.csv'
    SAVE_PATH_FREQ = './results/freq_MAB_uniform.csv'
# ====================== 
  
K = 5
beta_list = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15]
Nrep = 1000
results = np.zeros((Nrep, len(beta_list)+1))
rewards = np.zeros((Nrep, len(beta_list)+1))

T_list = [100, 500, 1000, 2000, 5000, 10000, 20000]
summ_df = pd.DataFrame()
freq_df = pd.DataFrame()
for T in T_list: 
    mu = np.array(abs_mu) 
        
    dgp = MABandit(mu = mu)
    dgp.set_ps(ps)

    for seed in range(Nrep):
        np.random.seed(seed)
        
        # ======================
        # # batched sampling  

        data = dgp.sample_data(T) 
        ys = data['ys']  
        ws = dgp.sample_arms(T)

        # ======================
        # # greedy algorithm """

        hat_mu = np.zeros(K) 
        for w in range(K):
            hat_mu[w] = np.mean(ys[ws==w, w]) 
        greedy_w = np.argmax(hat_mu)

        # ======================
        # # pessimism algorithm """ 

        hat_V = np.zeros(K)
        for w in range(K):
            hat_V[w] = np.max([np.sqrt(T * np.mean(ws==w) / ps[w]**2 ) / T, 
                            np.power(T * np.mean(ws==w) / ps[w]**3, 1/4) / T])
                            
        pess_w_list = []
        for ii in range(len(beta_list)):
            pess_w_list.append(np.argmax(hat_mu - beta_list[ii] * hat_V))

        # ======================
        # # summarize results """
        all_w_list = [greedy_w] + pess_w_list

        results[seed,:] = np.array(all_w_list)
        rewards[seed,:] = np.array([mu[i] for i in all_w_list])
        
    # ======================
    # summarize results across all seeds
    # ======================
    
    # probability of choosing the best arm 
    prob_correct = np.mean(results==np.argmax(mu), axis=0)
    prob_correct_std = np.sqrt(prob_correct * (1-prob_correct) / Nrep)
    
    # average suboptimality
    avg_subopt = np.max(mu) - np.mean(rewards, axis=0)
    avg_subopt_std = np.std(rewards) / np.sqrt(Nrep)
    
     

    sum_single = pd.DataFrame({"prob_correct": prob_correct, 
                               "prob_correct_sd": prob_correct_std,
                               "subopt": avg_subopt,
                               "subopt_std": avg_subopt_std,
                               "method_param": ['greedy'] + ["pess_"+str(beta) for beta in beta_list],
                               "param": [0] + beta_list,
                               "method": ['greedy'] + ['pessimism'] * len(beta_list),
                               "T": T})
    
    summ_df = pd.concat([summ_df, sum_single], axis=0)
    
    # record frequency of choosing each arm
    
    for w in range(K):
        freq_list = [np.mean(results[:, s]==w) for s in range(results.shape[1])]
        freq_df = pd.concat([freq_df,
                             pd.DataFrame({"freq": freq_list,
                                           "arm": w, 
                                           "mu": mu[w],
                                           'mu_subopt': np.max(mu) - mu[w],
                                           "method_param": ['greedy'] + ["pess_"+str(beta) for beta in beta_list],
                                           "param": [0] + beta_list,
                                           "method": ['greedy'] + ['pessimism'] * len(beta_list),
                                           "T": T})], axis = 0)

    
summ_df.to_csv(SAVE_PATH)
freq_df.to_csv(SAVE_PATH_FREQ)