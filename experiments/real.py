from utils.experiment import run_experiment
from utils.thompson import *
from utils.dgp import *
from algs.ptree import *
from algs.pess import *  

import numpy as np 
from copy import deepcopy 
import argparse 
import os
import openml

parser = argparse.ArgumentParser() 
parser.add_argument('--data', type = int, default = 0, help = 'data id')
parser.add_argument('--setting', type = int, default = 0, help = 'exploration setting')
parser.add_argument('--beta_id', type = int, default = 0, help = 'penalty parameter id')
parser.add_argument('--batch_size', type = int, default = 10, help = 'number of sample in each batch')
parser.add_argument('--depth', type = int, default = 5, help = 'depth of trees to build')
parser.add_argument('--seed', type = int, default = 0, help = 'seed group for reproducibility') 

args = parser.parse_args() 
batch_size = args.batch_size
depth = args.depth
beta_id = args.beta_id
data_id = args.data
setting = args.setting 
seed = args.seed

DECAY_LIST = [None, 0.2, 0.5, 0.8]
floor_decay = DECAY_LIST[setting]
if floor_decay is None:
    if_floor = False 
else:
    if_floor = True
BETA_LIST = [0.1, 0.5, 1, 2, 5, 10]
beta = BETA_LIST[beta_id]

SAVE_PATH = "./results/real/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

DATASET_LIST = ['waveform-5000', 'Long',  'cmc', 'artificial-characters',
                'Click_prediction_small', 'allrep',
                'mfeat-morphological', 'satellite_image', 'jungle_chess_2pcs_endgame_elephant_elephant', 'wilt', 'Satellite', 'ringnorm', 'mammography', 'delta_ailerons', 
                'PhishingWebsites', 'splice', 'pendigits', 'texture', 
                'cardiotocography', 'volcanoes-d4', 'volcanoes-b3', 'dis', 
                'optdigits', 'electricity', 'kr-vs-kp', 'bank-marketing', 'satimage',
                'MagicTelescope', 'houses', 'eeg-eye-state', 'car', 'segment']

np.random.seed(seed)
""" Load data sets """ 
dname = DATASET_LIST[data_id]
openml_list = openml.datasets.list_datasets()
dataset = openml.datasets.get_dataset(dname)
X, y, _, _ = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

""" Run and analyze the experiment """
data_exp, mus = generate_bandit_data(X=X, y=y, noise_std = 0.1, signal_strength=0.5)
xs, ys = data_exp['xs'], data_exp['ys']
K, p, T = data_exp['K'], data_exp['p'], data_exp['T']

xs_test, muxs_test, T_test = data_exp['xs_test'], data_exp['muxs_test'], data_exp['T_test']
eval_data = dict(xs = xs_test, ys = muxs_test, muxs = muxs_test)

# divide into batches
explore_size = int(np.min((100, T//50)))
batch_sizes = [explore_size] + [batch_size] * int(np.floor((T-explore_size)/batch_size))
T = np.sum(batch_sizes)
xs = xs[:T,:]
ys = ys[:T,:]


data = run_experiment(xs, ys, 'TS', batch_sizes = batch_sizes, num_mc = 1000,
                      if_floor = if_floor, floor_start = 1/2, 
                      floor_decay = floor_decay)
yobs, ws, ps = data['yobs'], data['ws'], data['ps']
ws = np.array([int(x) for x in ws])

""" greedy policy tree """
greedy_ptree = PL_greedy(xs, yobs, ws, ps, depth=depth)
pess_ptree, _, pess_tree_list = PL_pessimism(xxs = xs, yobs = yobs, wws = ws, exs = ps, 
                                beta = beta, depth = depth, 
                                lower_bound = 0.0001, muxs = None, verbose=True) 

greedy_w, greedy_eval_reward, rw_greedy = eval_ptree(greedy_ptree, eval_data)
pess_w, pess_eval_reward, rw_pess = eval_ptree(pess_ptree, eval_data)   


""" CV pessimistic policy tree """
opt_beta, opt_beta_1se, opt_beta_lcb, loss_list = PPL_CV_v3(xxs = xs_train, yobs = yobs, 
                                                   wws = ws, exs = ps_train, 
                                    beta_list = BETA_LIST, Nfold = 5, depth = DEPTH, 
                                    lower_bound = 0.0001, muxs = None, verbose = False)
pess_cv_ptree, _, pess_cv_tree_list = PL_pessimism(xxs = xs_train, yobs = yobs, wws = ws, exs = ps_train, 
                                beta = opt_beta, depth = DEPTH, 
                                lower_bound = 0.0001, muxs = None, verbose=True) 
pess_cv_1se_ptree, _, pess_cv_1se_tree_list = PL_pessimism(xxs = xs_train, yobs = yobs, wws = ws, exs = ps_train, 
                                beta = opt_beta_1se, depth = DEPTH, 
                                lower_bound = 0.0001, muxs = None, verbose=True) 
pess_cv_lcb_ptree, _, pess_cv_lcb_tree_list = PL_pessimism(xxs = xs_train, yobs = yobs, wws = ws, exs = ps_train, 
                                beta = opt_beta_lcb, depth = DEPTH, 
                                lower_bound = 0.0001, muxs = None, verbose=True) 

CV_pess_w, CV_pess_eval_reward, CV_rw_pess = eval_ptree(pess_cv_ptree, eval_data)
CV_1se_w, CV_1se_eval_reward, CV_1se_rw = eval_ptree(pess_cv_1se_ptree, eval_data)
CV_lcb_w, CV_lcb_eval_reward, CV_lcb_rw = eval_ptree(pess_cv_lcb_ptree, eval_data)
print((CV_rw_pess, CV_1se_rw, CV_lcb_rw))


"""  evaluation  """ 

res_df = pd.DataFrame({"eval_reward": [rw_greedy, CV_rw_pess, CV_1se_rw, CV_lcb_rw], 
                       "method_param": ['greedy', 'pess_cv', 'pess_cv_1se', 'pess_cv_lcb'],  
                       "floor": FLOOR_LIST[floor_id-1],  
                       "T": T,  
                       "beta": [0, opt_beta, opt_beta_1se, opt_beta_lcb], 
                       "data": data_id}) 


res_df.to_csv(SAVE_PATH + "explore_" + str(setting) + "data_" + str(data_id) + "_batch_" + str(batch_size) + "_beta_" + str(beta_id) + ".csv")