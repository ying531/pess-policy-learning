from algs.ptree import *




""" compute sample variance """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# what is the T*1 predicted actions from a policy
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# three variance terms

def compute_V(xxs, yobs, wws, what, exs, lower_bound = 0.001):
    T, K = exs.shape
      
    # compute three terms in uncertainty quanfier
    Vs_items = np.zeros((T,K))
    Vp_items = np.zeros((T,K))
    Vh_items = np.zeros((T,K))
    for k in range(K):
        Vs_items[(what == k) & (wws == k), k] = 1 / exs[(what == k) & (wws == k), k]**2
        Vp_items[what==k, k] = 1 / np.maximum(lower_bound, exs[what == k, k])
        Vh_items[what==k, k] = 1 / np.maximum(lower_bound, exs[what == k, k])**3
     
    Vs = np.sqrt(np.sum(Vs_items)) / T
    Vp = np.sqrt(np.sum(Vp_items)) / T
    Vh = np.power(np.sum(Vh_items), 1/4) / T
    
    res = dict(Vs_items = Vs_items,
               Vp_items = Vp_items,
               Vh_items = Vh_items,
               Vs = Vs, Vp = Vp, Vh = Vh,
               maxV = np.max((Vs, Vp, Vh)))
    
    return res

""" compute majorized upper bound for the variance term """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# what_0 is the T*1 predicted actions from a baseline policy pi_0
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# T*K matrix of augment score G(w,w0) that upper bounds the variance term 
def compute_Gamma_upp(xxs, yobs, wws, what_0, exs, 
                      lower_bound = 0.001, muxs = None):
    T, K = exs.shape
     
    if muxs is None:
        muxs = np.zeros((T, K))
    
    Gamma_mat = PL_aipw_score(xxs, yobs, wws, exs, muxs)
    Gamma_0 = np.zeros(T)
    for k in range(K):
        Gamma_0[(what_0==k) & (wws == k)] = Gamma_mat[(what_0==k) & (wws == k), k] 
    
    V0_res = compute_V(xxs, yobs, wws, what_0, exs, lower_bound=lower_bound)
    Vgamma = V0_res['Vs_items'] + V0_res['Vp_items']
    A0 = - np.sum(Gamma_0)/ (T * (T-1) * V0_res['maxV'])
    B0 = 1 / (2 * (T-1) * V0_res['maxV'])
    
    Gamma_mat_aug = A0 * Vgamma + B0 * Vgamma**2
    
    return Gamma_mat_aug
    
""" compute policy tree for one step when using majorized variance """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# what_0 is the T*1 predicted actions from a baseline policy pi_0
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# beta is the scalar for the penalty parameter 
# depth is the depth of the policy tree  
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# a policy tree optimizing the variance upper bound
def ptree_aug_one(xxs, yobs, wws, what_0, exs, beta = 0.1, depth = 2,
                  lower_bound = 0.001, muxs = None):
    T, K = exs.shape
    
    if muxs is None:
        muxs = np.zeros((T, K))
        
    Gamma_mat = muxs.copy()
    for w in range(K):
        Gamma_mat[wws==w,w] = muxs[wws==w,w] + (yobs[wws==w] - muxs[wws==w,w]) / exs[wws==w,w]
    
    Gamma_mat_aug = compute_Gamma_upp(xxs, yobs, wws, what_0, exs, lower_bound, muxs)
    Gamma_mat_all = Gamma_mat - beta * Gamma_mat_aug
    
    one_ptree = pt.hybrid_policy_tree(X = xxs, Gamma = Gamma_mat_all, depth=depth)
    
    return one_ptree


""" compute policy tree until converge when using majorized variance """
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t 
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# beta is the scalar for the penalty parameter 
# depth is the depth of the policy tree  
# muxs (optional) is the T*K matrix for hat{mu}(x_t,a) for a=1,..,K
### == outputs:
# a policy tree optimizing the variance upper bound
def PL_pessimism(xxs, yobs, wws, exs, beta = 0.1, depth = 2,
                  lower_bound = 0.0001, muxs = None, maxround = 50, verbose=False):
    T, K = exs.shape
    
    if muxs is None:
        muxs = np.zeros((T, K))
    
    # start with greedy policy tree
    current_ptree = PL_greedy(xxs, yobs, wws, exs, depth=depth, muxs = None)
    # predict the current ptree as criterion
    fitted_actions = predict_ptree(current_ptree, xxs)
    
    # iteratively fit policy trees
    flag = True
    fit_count = 0
    while flag:
        fit_count += 1
        if fit_count < 10 or fit_count % 10 == 0:
            if verbose:
                print("learning in the "+str(fit_count)+"-th round ...")
        new_ptree = ptree_aug_one(xxs, yobs, wws, 
                                  what_0=fitted_actions, exs = exs, 
                                  beta = beta, depth = depth,
                                  lower_bound = lower_bound, muxs = muxs)
        new_fitted_actions = predict_ptree(new_ptree, xxs)
        
        if np.sum(new_fitted_actions == fitted_actions) > 0.999 * T or fit_count > maxround:
            flag = False 
        
        current_ptree = deepcopy(new_ptree)
        fitted_actions = deepcopy(new_fitted_actions)
    
    return current_ptree, fit_count



""" cross-validated pessimistic policy learning """
# using the i-th fold to train and i+1-th fold to evaluate
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# beta_list is the list of penalty parameters to tune over
# Nfold is the number of folds for cross-validation
# depth, lower_bound, muxs, maxround are parameters for running PL_pessimism()
### == outputs:
# opt_beta: the optimal penalty parameter based on average loss
# opt_beta_1se: the optimal penalty parameter based on 1-SE rule
# opt_beta_lcb: the optimal penalty parameter based on lower confidence bound
# avg_loss_list: the average loss for each beta in beta_list across folds

def PPL_CV(xxs, yobs, wws, exs, 
           beta_list = [0.1, 1, 5, 10, 15], Nfold = 2,
           depth = 2, lower_bound = 0.0001, muxs = None, 
           maxround = 50, verbose=False):
    T, K = exs.shape
    if muxs is None:
        muxs = np.zeros((T,K))
    
    # sample splitting 
    idx_list = []
    for ii in range(Nfold):
        idx_list.append(np.arange(np.floor(T*ii/Nfold), np.floor(T*(ii+1)/Nfold)))
    
    # evlauation scheme: use I_t to train, and I_t+1 to evaluate 
    loss_list = [[] for _ in range(len(beta_list))]
    
    for ibeta in range(len(beta_list)):

        current_beta = beta_list[ibeta]
        for ifold in range(Nfold-1):
            current_idx = [int(x) for x in idx_list[ifold]] 
            # train pess(beta) on I_t
            current_ptree, _ = PL_pessimism(xxs[current_idx].copy(), 
                                         yobs[current_idx].copy(), 
                                         wws[current_idx].copy(), 
                                         exs[current_idx].copy(), 
                                         beta = current_beta, depth = depth, 
                                         lower_bound = lower_bound, 
                                         muxs = muxs[current_idx].copy(), 
                                         maxround = maxround, verbose=False)
            # evaluate learned tree on I_t+1
            eval_idx = [int(x) for x in idx_list[ifold+1]]  
            current_score = emp_eval_ptree(current_ptree, 
                                           eval_xs = xxs[eval_idx].copy(), 
                                           eval_yobs = yobs[eval_idx].copy(), 
                                           eval_ws = wws[eval_idx].copy(), 
                                           eval_exs = exs[eval_idx].copy(), 
                                           eval_muxs = muxs[eval_idx].copy())
            loss_list[ibeta].append(current_score)
    
    avg_loss_list = [np.mean(ll) for ll in loss_list]
    sse_ratio_list = [(np.mean(ll) - np.min(ll)) / np.std(ll) for ll in loss_list] 
    sse_loss_list = [np.mean(ll) - np.std(ll)/np.sqrt(len(ll)) for ll in loss_list]
     
    
    
    opt_beta = beta_list[np.argmax(avg_loss_list)] 
    opt_beta_lcb = beta_list[np.argmax(sse_loss_list)]
    
    smaller_beta = [beta_list[x] for x in range(len(beta_list)) if sse_ratio_list[x] <= 1]
    if len(smaller_beta) > 0:
        opt_beta_1se = np.max(smaller_beta) 
    else:
        opt_beta_1se = beta_list[np.argmax(sse_loss_list)]
    
    
    return opt_beta, opt_beta_1se, opt_beta_lcb, loss_list

""" cross-validated pessimistic policy learning (scheme 2) """
# using the i-th fold to train and all subsequent data to evaluate
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# beta_list is the list of penalty parameters to tune over
# Nfold is the number of folds for cross-validation
# depth, lower_bound, muxs, maxround are parameters for running PL_pessimism()
### == outputs:
# opt_beta: the optimal penalty parameter based on average loss
# opt_beta_1se: the optimal penalty parameter based on 1-SE rule
# opt_beta_lcb: the optimal penalty parameter based on lower confidence bound
# avg_loss_list: the average loss for each beta in beta_list across folds

def PPL_CV_v2(xxs, yobs, wws, exs, 
           beta_list = [0.1, 1, 5, 10, 15], Nfold = 2,
           depth = 2, lower_bound = 0.0001, muxs = None, 
           maxround = 50, verbose=False):
    T, K = exs.shape
    if muxs is None:
        muxs = np.zeros((T,K))
    
    # sample splitting 
    idx_list = []
    for ii in range(Nfold):
        idx_list.append(np.arange(np.floor(T*ii/Nfold), np.floor(T*(ii+1)/Nfold)))
    
    # evlauation scheme: use I_t to train, and I_t+1 to evaluate 
    loss_list = [[] for _ in range(len(beta_list))]
    
    for ibeta in range(len(beta_list)): 

        current_beta = beta_list[ibeta]
        for ifold in range(Nfold-1):
            current_idx = [int(x) for x in idx_list[ifold]] 
            # train pess(beta) on I_t
            current_ptree, _ = PL_pessimism(xxs[current_idx].copy(), 
                                         yobs[current_idx].copy(), 
                                         wws[current_idx].copy(), 
                                         exs[current_idx].copy(), 
                                         beta = current_beta, depth = depth, 
                                         lower_bound = lower_bound, 
                                         muxs = muxs[current_idx].copy(), 
                                         maxround = maxround, verbose=False)
            # evaluate learned tree on I_t+1
            eval_idx = range(np.max(current_idx), T)  
            current_score = emp_eval_ptree(current_ptree, 
                                           eval_xs = xxs[eval_idx].copy(), 
                                           eval_yobs = yobs[eval_idx].copy(), 
                                           eval_ws = wws[eval_idx].copy(), 
                                           eval_exs = exs[eval_idx].copy(), 
                                           eval_muxs = muxs[eval_idx].copy())
            loss_list[ibeta].append(current_score)
    
    avg_loss_list = [np.mean(ll) for ll in loss_list]
    sse_ratio_list = [(np.mean(ll) - np.min(ll)) / np.std(ll) for ll in loss_list] 
    sse_loss_list = [np.mean(ll) - np.std(ll)/np.sqrt(len(ll)) for ll in loss_list] 
    
    
    opt_beta = beta_list[np.argmax(avg_loss_list)] 
    opt_beta_lcb = beta_list[np.argmax(sse_loss_list)]
    
    smaller_beta = [beta_list[x] for x in range(len(beta_list)) if sse_ratio_list[x] <= 1]
    if len(smaller_beta) > 0:
        opt_beta_1se = np.max(smaller_beta) 
    else:
        opt_beta_1se = beta_list[np.argmax(sse_loss_list)]

    
    return opt_beta, opt_beta_1se, opt_beta_lcb, avg_loss_list



""" cross-validated pessimistic policy learning (scheme 3) """
# using samples before the i-th fold to train and all subsequent data to evaluate
### == inputs:
# xxs is the T*p covariate matrix
# yobs is the T*1 observed outcomes 
# wws is the T*1 observed actions A_t
# exs is the T*K matrix for e_t(x_t,a) for a=1,..,K
# beta_list is the list of penalty parameters to tune over
# Nfold is the number of folds for cross-validation
# depth, lower_bound, muxs, maxround are parameters for running PL_pessimism()
### == outputs:
# opt_beta: the optimal penalty parameter based on average loss
# opt_beta_1se: the optimal penalty parameter based on 1-SE rule
# opt_beta_lcb: the optimal penalty parameter based on lower confidence bound
# avg_loss_list: the average loss for each beta in beta_list across folds

def PPL_CV_v3(xxs, yobs, wws, exs, 
           beta_list = [0.1, 1, 5, 10, 15], Nfold = 2,
           depth = 2, lower_bound = 0.0001, muxs = None, 
           maxround = 50, verbose=False):
    T, K = exs.shape
    if muxs is None:
        muxs = np.zeros((T,K))
    
    # sample splitting 
    idx_list = []
    for ii in range(Nfold):
        idx_list.append(np.arange(np.floor(T*ii/Nfold), np.floor(T*(ii+1)/Nfold)))
    
    # evlauation scheme: use I_t to train, and I_t+1 to evaluate 
    loss_list = [[] for _ in range(len(beta_list))]
    
    for ibeta in range(len(beta_list)):

        current_beta = beta_list[ibeta]
        for ifold in range(Nfold *3 // 4 - 1):
            current_idx = range(0, int(np.max(idx_list[ifold])))
            
            # train pess(beta) on I_t
            current_ptree, _ = PL_pessimism(xxs[current_idx].copy(), 
                                         yobs[current_idx].copy(), 
                                         wws[current_idx].copy(), 
                                         exs[current_idx].copy(), 
                                         beta = current_beta, depth = depth, 
                                         lower_bound = lower_bound, 
                                         muxs = muxs[current_idx].copy(), 
                                         maxround = maxround, verbose=False)
            # evaluate learned tree on I_t+1
            eval_idx = range(np.max(current_idx), T)  
            current_score = emp_eval_ptree(current_ptree, 
                                           eval_xs = xxs[eval_idx].copy(), 
                                           eval_yobs = yobs[eval_idx].copy(), 
                                           eval_ws = wws[eval_idx].copy(), 
                                           eval_exs = exs[eval_idx].copy(), 
                                           eval_muxs = muxs[eval_idx].copy())
            loss_list[ibeta].append(current_score)
    
    avg_loss_list = [np.mean(ll) for ll in loss_list]
    sse_ratio_list = [(np.mean(ll) - np.min(ll)) / np.std(ll) for ll in loss_list] 
    sse_loss_list = [np.mean(ll) - np.std(ll)/np.sqrt(len(ll)) for ll in loss_list] 
    
    
    opt_beta = beta_list[np.argmax(avg_loss_list)] 
    opt_beta_lcb = beta_list[np.argmax(sse_loss_list)]
    
    smaller_beta = [beta_list[x] for x in range(len(beta_list)) if sse_ratio_list[x] <= 1]
    if len(smaller_beta) > 0:
        opt_beta_1se = np.max(smaller_beta) 
    else:
        opt_beta_1se = beta_list[np.argmax(sse_loss_list)]

    
    return opt_beta, opt_beta_1se, opt_beta_lcb, avg_loss_list