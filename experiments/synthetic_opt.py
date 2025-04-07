from utils.experiment import run_experiment, run_experiment_opt
from utils.thompson import *
from utils.dgp import *
from algs.ptree import *
from algs.pess import *

import numpy as np 
from copy import deepcopy 
import argparse 
import os

parser = argparse.ArgumentParser() 
parser.add_argument('--setting', type = int, default = 1, help = 'setting number')
parser.add_argument('--T', type = int, default = 1000, help = 'total sample size') 
parser.add_argument('--p', type = int, default = 2, help = 'dimension of feature')
parser.add_argument('--beta', type = int, default = 0, help = 'penalty parameter')
parser.add_argument('--batch_size', type = int, default = 10, help = 'number of sample in each batch')
parser.add_argument('--depth', type = int, default = 5, help = 'depth of trees to build')
parser.add_argument('--seed', type = int, default = 0, help = 'seed group for reproducibility')
parser.add_argument('--cv', type = int, default = 0, 
                    help = 'whether cross validate penalty parameter (1=yes, 0=no)')

args = parser.parse_args() 
setting = args.setting
p = args.p
T = args.T
beta_id = args.beta
depth = args.depth
seed = args.seed
batch_size = args.batch_size
if_cv = args.cv

if if_cv == 0: 
    SAVE_PATH = "./results/opt_synthetic/" 
if if_cv == 1:
    SAVE_PATH = "./results/cv_opt_synthetic/"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH) 
 
batch_sizes = [100]+[batch_size]*int((T-100)/batch_size) 
T_eval = 100000
BETA_LIST=  [0.1, 0.2, 0.5, 1, 5, 10]
beta = BETA_LIST[beta_id-1]
Nrep = 1
K = 10
p = 2

if setting == 1:
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0) 
    if_floor = True
    floor_start = 0.001
    floor_decay = 0
    
if setting == 2:
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = 0.5

if setting == 3:
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = 0.8
    
if setting == 4:
    dgp = MultiQuad(p, K)
    dgp_eval = MultiQuad(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = 0.2
    

eval_data = dgp_eval.sample_data(T_eval) 



res_df  = pd.DataFrame()
for ii_seed in range(Nrep):
    np.random.seed(Nrep * seed + ii_seed)

    """ sample all x and y """
    data = dgp.sample_data(T)
    xs = data['xs']
    ys = data['ys']

    """ run thompson sampling to collect adaptive data """
    results = run_experiment_opt(xs, ys, 'TS', dgp, 
                            batch_sizes = batch_sizes, num_mc=1000, add_floor_const = 0.1,
                            record_idx=[1,2, len(batch_sizes)-1],
                            if_floor=if_floor, floor_start = floor_start, floor_decay = floor_decay)
    TSagents = results['agents']

    agent = TSagents[-1]

    ws = results['ws']
    yobs = results['yobs']
    ps = results['ps']
    
    ws_opt = dgp.compute_optimal(xs)
    ws_opt = np.array(ws_opt)
    ps_opt = np.zeros((T))
    for k in range(K):
        if np.sum(ws_opt==k)>0:
            ps_opt[ws_opt==k] = ps[ws_opt==k,k]

    """ greedy policy tree """
    if ii_seed == 0:
        print("-- start running greedy ptree ...")
    greedy_ptree = PL_greedy(xs, yobs, ws, ps, depth=depth)

    # ====================================================
    """ # run pessimism with the given beta without CV """ 

    if if_cv == 0:

        """ pessimistic policy tree """
        if ii_seed == 0:
            print("-- start running pess tree ...") 
        pess_ptree, _ = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                                beta = beta, depth = depth, 
                                lower_bound = 0.0001, muxs = None, verbose=(ii_seed==0)) 
        
        """  evaluation  """
        greedy_w, greedy_eval_reward, rw_greedy = eval_ptree(greedy_ptree, eval_data)
        pess_w, pess_eval_reward, rw_pess = eval_ptree(pess_ptree, eval_data) 
        print((rw_greedy, rw_pess))
        
        res_df = pd.concat([res_df, 
                            pd.DataFrame({"eval_reward": [rw_greedy, rw_pess], 
                                        "method_param": ['greedy', 'pess_'+str(beta)], 
                                        "method": ['greedy', 'pessimism'], 
                                        "batch_size": batch_size,  
                                        "T": T,  
                                        "beta": [0, beta],  
                                        "seed": seed * 100 + ii_seed})], axis = 0)
        
        print("seed:" + str(ii_seed) + "| greedy:" + str(round(rw_greedy,4)) + 
              "| pess:" + str(round(rw_pess,4)))
     
         
    # ====================================================
    """ # run pessimistic policy tree with CV """
    """ # use I_t+1:Nfold to evaluate policy learned on I_0:t """

    if if_cv == 1:

        if ii_seed == 0:
            print("-- start running CV pess tree ...") 
        opt_beta, opt_beta_1se, opt_beta_lcb, loss_list = PPL_CV_v3(xxs = xs, yobs = yobs, 
                                                   wws = ws, exs = ps, 
                                    beta_list = BETA_LIST, Nfold = 5, depth = depth, 
                                    lower_bound = 0.0001, muxs = None, verbose = False)
        if ii_seed == 0:
            print(("beta list: ", BETA_LIST))
            print(("loss: ", loss_list)) 
        
        CV_pess_ptree, _ = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                                beta = opt_beta, depth = depth, 
                                lower_bound = 0.0001, muxs = None, verbose=False) 
        CV_1se_ptree, _ = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                                beta = opt_beta_1se, depth = depth, 
                                lower_bound = 0.0001, muxs = None, verbose=False) 
        CV_lcb_ptree, _ = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                                beta = opt_beta_lcb, depth = depth, 
                                lower_bound = 0.0001, muxs = None, verbose=False) 

        """  evaluation  """
        greedy_w, greedy_eval_reward, rw_greedy = eval_ptree(greedy_ptree, eval_data) 
        CV_pess_w, CV_pess_eval_reward, CV_rw_pess = eval_ptree(CV_pess_ptree, eval_data)
        CV_1se_w, CV_1se_eval_reward, CV_1se_rw = eval_ptree(CV_1se_ptree, eval_data)
        CV_lcb_w, CV_lcb_eval_reward, CV_lcb_rw = eval_ptree(CV_lcb_ptree, eval_data)
        
        res_df = pd.concat([res_df, 
                            pd.DataFrame({"eval_reward": [rw_greedy, CV_rw_pess, 
                                                          CV_1se_rw, CV_lcb_rw], 
                                        "method_param": ['greedy', "CV_pess", 
                                                         "CV_pess_1se", "CV_pess_lcb"], 
                                        "method": ['greedy', "CV_pessimism", "CV_pessismism_1se", "CV_pessimism_lcb"], 
                                        "batch_size": batch_size,  
                                        "T": T,  
                                        "beta": [0, opt_beta, opt_beta_1se, opt_beta_lcb],  
                                        "seed": seed * 100 + ii_seed})], axis = 0)

    
res_df.to_csv(SAVE_PATH + "setting_" + str(setting) + "_seed_" + str(seed) + "_batch_" + str(batch_size) + "_T_" + str(T) + "_beta_" + str(beta_id) + ".csv")
 
