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

args = parser.parse_args() 
setting = args.setting
p = args.p
T = args.T
beta_id = args.beta
depth = args.depth
seed = args.seed
batch_size = args.batch_size 
 
SAVE_PATH = "./results/lin_synthetic/" 

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
 
 
batch_sizes = [100]+[batch_size]*int((T-100)/batch_size) 
T_eval = 100000
BETA_LIST=  [0.1, 0.2, 0.5, 1, 5, 10]
beta = BETA_LIST[beta_id-1]
Nrep = 50
K = 10
p = 2

if setting == 1:
    dgp = MultiLinear(p, K)
    dgp_eval = MultiLinear(p, K, sigma=0)
    if_floor = True
    floor_start = 0.001
    floor_decay = 0 
    
if setting == 2:
    dgp = MultiLinear(p, K)
    dgp_eval = MultiLinear(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = 0.5

if setting == 3:
    dgp = MultiLinear(p, K)
    dgp_eval = MultiLinear(p, K, sigma=0)
    if_floor = True
    floor_start = 1/K
    floor_decay = 0.8
    
if setting == 4:
    dgp = MultiLinear(p, K)
    dgp_eval = MultiLinear(p, K, sigma=0)
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
    results = run_experiment(xs, ys, 'TS', dgp, 
                            batch_sizes = batch_sizes, num_mc=1000, 
                            record_idx=[1,2, len(batch_sizes)-1],
                            if_floor=if_floor, floor_start = floor_start, floor_decay = floor_decay)
    TSagents = results['agents']

    agent = TSagents[-1]

    ws = results['ws']
    yobs = results['yobs']
    ps = results['ps']

    """ greedy policy tree """
    if ii_seed == 0:
        print("-- start running greedy ptree ...")
    greedy_ptree = PL_greedy(xs, yobs, ws, ps, depth=depth)

    # ====================================================
    """ # run pessimism with the given beta without CV """   

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
      
res_df.to_csv(SAVE_PATH + "setting_" + str(setting) + "_seed_" + str(seed) + "_batch_" + str(batch_size) + "_T_" + str(T) + "_beta_" + str(beta_id) + ".csv")
 
