from utils.thompson import *
from utils.dgp import *

import numpy as np 
from copy import deepcopy 

def run_experiment(xs, ys, bandit_model, dgp_model, 
                   batch_sizes, num_mc = 100, record_idx = None,
                   if_floor = False, floor_start = None, floor_decay = None):
    T, K = ys.shape 
    _, p = xs.shape 
    ws = np.empty(T, dtype = np.int_)
    yobs = np.empty(T)
    probs = np.zeros((T, T, K))
    probs_t = np.zeros((T, K))

    if record_idx is None:
        record_idx = [len(batch_sizes)-1]
 

    if bandit_model == 'TS':
        agent = LinTS(K, p, dgp_model, num_mc = num_mc)
        agent.initialize_ps(T)
        agent.initialize_w(T)
        agent.set_floor(if_floor, floor_start, floor_decay)

    # process batch sizes
    batch_size_cumsum = list(np.cumsum(batch_sizes)) 
    
    """ uniform sampling at first batch  """
    initial_batch = batch_size_cumsum[0]
    ws[:initial_batch] = [int(x) for x in (np.arange(initial_batch) % K)]
    yobs_initial = np.zeros(initial_batch)
    for k in range(K):
        yobs_initial[ws[:initial_batch]==k] = ys[:initial_batch][ws[:initial_batch]==k, k] 
    yobs[:initial_batch] = yobs_initial
    agent.ps[:initial_batch] = 1/K 

    agent.update_TS(xs[:initial_batch], ws[:initial_batch], yobs[:initial_batch])
    
    """ sampling and updating in each batch """
    print("===== start sampling =====")
    record_agents = []
    for idx, (st, ed) in enumerate(zip(batch_size_cumsum[:-1], batch_size_cumsum[1:]), 1):
        # print("sampling [" + str(st) + ", "+str(ed)+'] ...')
        if bandit_model == 'TS':
            # sample one batch using thompson sampling
            w, _ = agent.draw_TS_one_batch(xs, st, ed, current_t = st) 
            # update mean and covariance matrix estimation
            new_yobs = np.zeros(ed-st)
            sub_ys = deepcopy(ys[st:ed])
            w = np.array(w)
            for k in range(K):
                new_yobs[w==k] = sub_ys[w==k, k]
            yobs[st:ed] = new_yobs
            del sub_ys 
            agent.update_TS(xs[st:ed], agent.ws[st:ed], yobs[st:ed])
            # print("Arms taken:", (len(agent.X[0]), len(agent.X[1]), len(agent.X[2])))
        if idx in record_idx:
            record_agents.append(deepcopy(agent))
            
    data = dict(agents = record_agents, yobs = yobs, ws = agent.ws, xs = xs, ys = ys, ps = agent.ps)

    return data





def run_experiment_opt(xs, ys, bandit_model, dgp_model, 
                   batch_sizes, num_mc = 100, record_idx = None, add_floor_const = 0.1,
                   if_floor = False, floor_start = None, floor_decay = None):
    T, K = ys.shape 
    _, p = xs.shape 
    ws = np.empty(T, dtype = np.int_)
    yobs = np.empty(T)
    probs = np.zeros((T, T, K))
    probs_t = np.zeros((T, K))

    if record_idx is None:
        record_idx = [len(batch_sizes)-1]
 

    if bandit_model == 'TS':
        agent = LinTS(K, p, dgp_model, num_mc = num_mc)
        agent.initialize_ps(T)
        agent.initialize_w(T)
        agent.set_floor(if_floor, floor_start, floor_decay)
        opt_w = np.array(dgp_model.compute_optimal(xs))

    # process batch sizes
    batch_size_cumsum = list(np.cumsum(batch_sizes)) 
    
    add_floor = np.zeros((T, K))
    for tt in range(T):
        add_floor[tt, opt_w[tt]] = add_floor_const
    
    """ uniform sampling at first batch  """
    initial_batch = batch_size_cumsum[0]
    ws[:initial_batch] = [int(x) for x in (np.arange(initial_batch) % K)]
    yobs_initial = np.zeros(initial_batch)
    for k in range(K):
        yobs_initial[ws[:initial_batch]==k] = ys[:initial_batch][ws[:initial_batch]==k, k] 
    yobs[:initial_batch] = yobs_initial
    agent.ps[:initial_batch] = 1/K 

    agent.update_TS(xs[:initial_batch], ws[:initial_batch], yobs[:initial_batch])
    
    """ sampling and updating in each batch """
    print("===== start sampling =====")
    record_agents = []
    for idx, (st, ed) in enumerate(zip(batch_size_cumsum[:-1], batch_size_cumsum[1:]), 1):
        # print("sampling [" + str(st) + ", "+str(ed)+'] ...')
        if bandit_model == 'TS':
            # sample one batch using thompson sampling
            w, _ = agent.draw_TS_one_batch(xs, st, ed, current_t = st, add_floor = add_floor) 
            # update mean and covariance matrix estimation
            new_yobs = np.zeros(ed-st)
            sub_ys = deepcopy(ys[st:ed])
            w = np.array(w)
            for k in range(K):
                new_yobs[w==k] = sub_ys[w==k, k]
            yobs[st:ed] = new_yobs
            del sub_ys 
            agent.update_TS(xs[st:ed], agent.ws[st:ed], yobs[st:ed])
            # print("Arms taken:", (len(agent.X[0]), len(agent.X[1]), len(agent.X[2])))
        if idx in record_idx:
            record_agents.append(deepcopy(agent))
            
    data = dict(agents = record_agents, yobs = yobs, ws = agent.ws, xs = xs, ys = ys, ps = agent.ps)

    return data
