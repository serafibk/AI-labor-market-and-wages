import pickle
from tqdm import tqdm
import labor_market_q_learning_simulation
from labor_market_q_learning_simulation import Market, Firm, Worker
import numpy as np
import matplotlib.pyplot as plt


def load_firm_worker_info(folder, run_folder, use_mvpt, type_conditioning):

    firm_info = [{"q-vals":None,"ind-range":None, "profit":None,"benchmark":None,"AT":None}for i in range(N_f)]
    for i in range(N_f):
        with open(f"{folder}/{run_folder}/q_vals_firm_{i}.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/ind_bench_l_{i}.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/ind_bench_u_{i}.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/firm_{i}_profits.pkl","rb") as pkl_fl4, open(f"{folder}/{run_folder}/firm_{i}_benchmarks.pkl","rb") as pkl_fl7,open(f"{folder}/{run_folder}/firm_{i}_ATs.pkl","rb") as pkl_fl8:
            firm_info[i]["q-vals"] = dict(pickle.load(pkl_fl1))
            firm_info[i]["ind-range"] = (list(pickle.load(pkl_fl2)), list(pickle.load(pkl_fl3)))
            firm_info[i]["profit"] = list(pickle.load(pkl_fl4))
            firm_info[i]["benchmark"] = list(pickle.load(pkl_fl7))
            firm_info[i]["AT"] = list(pickle.load(pkl_fl8))
            if use_mvpt:
                with open(f"{folder}/{run_folder}/mvpt_bench_l_{i}.pkl","rb") as pkl_fl5, open(f"{folder}/{run_folder}/mvpt_bench_u_{i}.pkl","rb") as pkl_fl6:
                    firm_info[i]["mvpt-range"] = (list(pickle.load(pkl_fl5)), list(pickle.load(pkl_fl6)))
    with open(f"{folder}/{run_folder}/sal_bench_l.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/sal_bench_u.pkl","rb") as pkl_fl2:
        sal_bench_l = list(pickle.load(pkl_fl1))
        sal_bench_u = list(pickle.load(pkl_fl2))
    
    worker_info = [{"q-vals":None, "type":None, "wage":None, "type_idx":None,"firm-choice":None,"offer":None} for i in range(N_w)]
    idx_R = 0
    idx_S = 0
    for i in range(N_w):
        try: # risky worker
            with open(f"{folder}/{run_folder}/q_vals_worker_{i}_R.pkl","rb") as pkl_fl:
                worker_info[i]["q-vals"] = dict(pickle.load(pkl_fl))
                worker_info[i]["type"] = "risky"
                worker_info[i]["type_idx"] = idx_R
                with open(f"{folder}/{run_folder}/worker_R_{idx_R}_wage.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/worker_R_{idx_R}_firm_choice.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/worker_R_{idx_R}_offer.pkl","rb") as pkl_fl4:
                    worker_info[i]["wages"] = list(pickle.load(pkl_fl2))
                    worker_info[i]["firm-choices"] = list(pickle.load(pkl_fl3))
                    worker_info[i]["offers"] = list(pickle.load(pkl_fl4))
                idx_R = idx_R + 1
        except: # safe worker
            with open(f"{folder}/{run_folder}/q_vals_worker_{i}_S.pkl","rb") as pkl_fl:
                worker_info[i]["q-vals"] = dict(pickle.load(pkl_fl))
                worker_info[i]["type"] = "safe"
                worker_info[i]["type_idx"] = idx_S
                with open(f"{folder}/{run_folder}/worker_S_{idx_S}_wage.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/worker_S_{idx_S}_firm_choice.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/worker_S_{idx_S}_offer.pkl","rb") as pkl_fl4:
                    worker_info[i]["wages"] = list(pickle.load(pkl_fl2))
                    worker_info[i]["firm-choices"] = list(pickle.load(pkl_fl3))
                    worker_info[i]["offers"] = list(pickle.load(pkl_fl4))
                idx_S = idx_S + 1
    
    return firm_info, worker_info, sal_bench_l, sal_bench_u



def firm_deviation_test(folder, run_folder, worker_info, firm_info, N_w,N_f,W,p_s,alpha,delta,initial_belief_strength,p_risky,type_conditioning,p_reset,beta,use_mvpt,posting,mixed_posts):
    

    valid_case = 1
    for i in range(N_f):
        if  firm_info[i]["ind-range"][1] == 1 and valid_case: # can't be at the top of the wage spectrum 
            valid_case = 0
            print(f"Not a valid case: some workers capturing all the surplus")

    with open(f"{folder}/{run_folder}/sal_bench_l.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/sal_bench_u.pkl","rb") as pkl_fl2:
        sal_bench_l = float(list(pickle.load(pkl_fl1))[-1])
        sal_bench_u = float(list(pickle.load(pkl_fl2))[-1]) 
            
    if valid_case:
        print("Valid case!")
        # set states and actions
        S_w_sharing = [(w,l,u) for w in W for l in W for u in W if l<= u] 
        S_w_firm_choice = [(r1,r2,r3,r4,r5) for r1 in ranges for r2 in ranges for r3 in ranges for r4 in ranges for r5 in ranges] # r == res. price of worker: max(w,m_m) TODO -- how to make this tractable
        S_w_negotiation = [(o_l, o_h) for o_l in W for o_h in W if o_l <= o_h] # range of offers to consider
        A_w_sharing = ["share", "no share"] # action set for each worker and firm deciding whether to share wage info. Worker shares current salary, firm "plugs in" HR software
        A_w_firm_choice = range(N_f) # all firm indices
        A_w_negotiation = ["no offer"] + W # no offer + range 

        S_f_benchmark = [((l_b,u_b),(l_i,u_i)) for l_b in W for u_b in W for l_i in W  for u_i in W if l_b<= u_b and l_i <= u_i] # salary benchmark data + individual data
        S_f_negotiation = [(l,m,u) for l in W for u in W for m in W if l<= u] # will see lower, median, and upper bound values for one of the sources
        A_f_benchmark = ["salary benchmark", "independent"]
        if use_mvpt:
            S_f_benchmark = S_f_benchmark + [((l_b,u_b),(l_i,u_i),(l_m,u_m)) for l_b in W for u_b in W for l_i in W  for u_i in W  for l_m in W for u_m in W if l_b<= u_b and l_i <= u_i and l_m <= u_m] # salary benchmark data + individual data + MVPT data (probably too many states)
            A_f_benchmark = A_f_benchmark + ["mvpt"] 

        eval_market = Market(N_w,N_f,S_w_sharing,A_w_sharing,S_w_negotiation,A_w_negotiation,S_w_firm_choice, A_w_firm_choice,S_f_benchmark,A_f_benchmark,S_f_negotiation,W,p_s,alpha,delta,initial_belief_strength,p_risky=p_risky,type_conditioning=type_conditioning,p_reset=p_reset,beta=beta,use_mvpt=use_mvpt,posting=posting,mixed_posts=mixed_posts,eval=True)

        # update firm starting info
        deviating_idx = 0 # keep this consistent for now 
        for idx, f in enumerate(eval_market.firms):
            f.Q_vals = firm_info[idx]["q-vals"] # update q-vals
            f.state = (firm_info[idx]["ind-range"], (sal_bench_l, sal_bench_u)) # update state (ind range, sal range)
            # update individual range (bot10,bot90) => infer bot50=bot90=bot10 for this setting
            f.bot_10 = firm_info[idx]["ind-range"][0]
            f.bot_50 = firm_info[idx]["ind-range"][0]
            f.bot_90 = firm_info[idx]["ind-range"][1] 
            f.profit = firm_info[idx]["profit"]  # update profit

            if idx == deviating_idx:
                f.bot_10 = 1.0#min(W, key = lambda x:abs(x-(firm_info[idx]["ind-range"][1] + 3/k)))
                f.bot_50 = 1.0#min(W, key = lambda x:abs(x-(firm_info[idx]["ind-range"][1] + 3/k)))
                f.bot_90 = 1.0#min(W, key = lambda x:abs(x-(firm_info[idx]["ind-range"][1] + 3/k)))
                f.deviating = True
                f.benchmark = "independent"
                f.acceptance_threshold = (f.bot_10, f.bot_10)


        # update worker starting info
        idx_R = 0
        idx_S = 0
        for idx, w in enumerate(eval_market.workers):
            w.Q_vals = worker_info[idx]["q-vals"]# update Q-vals
            w.wage = worker_info[idx]["wage"]# update wage
            w.type = worker_info[idx]["type"]# update type 
            w.firm_negotiation_choice = worker_info[idx]["firm-choice"]# update negotiation choice

        T = 20

        firm_profits = [[] for i in range(N_f)]
        firm_applicants = [[] for i in range(N_f)]
        worker_wages = []

        for t in tqdm(range(T)):
            
            all_wages = []
            applicants = [0 for i in range(N_f)]
            for i in range(N_w):
                if t==0 or t == T-1:
                    all_wages.append(eval_market.workers[i].wage)
                idx_f = eval_market.workers[i].firm_negotiation_choice
                applicants[idx_f] = applicants[idx_f] + 1 
            if len(all_wages)>0:
                worker_wages.append(all_wages)
            for i in range(N_f):
                firm_profits[i].append(eval_market.firms[i].profit)
                firm_applicants[i].append(applicants[i])

                # if eval_market.workers[i].firm_negotiation_choice == deviating_idx:
                #     print(eval_market.workers[i].offer)
            
            eval_market.market_time_step(t)

        print(f"Average worker surplus change = {sum(worker_wages[1])/len(worker_wages[1]) - sum(worker_wages[0])/len(worker_wages[0])}")
        dev_test_worker_surplus_change.append(sum(worker_wages[1])/len(worker_wages[1]) - sum(worker_wages[0])/len(worker_wages[0]))
        for i in range(N_f):
            dev_test_firm_profits[i].append(firm_profits[i])
            dev_test_firm_applicants[i].append(firm_applicants[i])


def worker_belief_test(folder, run_folder,worker_info, firm_info):
    print("--Worker Belief Test--")
    with open(f"{folder}/{run_folder}/mvpt_values.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/median_values.pkl","rb") as pkl_fl2:
        mvpt_median = float(list(pickle.load(pkl_fl1))[-1])
        median = float(list(pickle.load(pkl_fl2))[-1])
    
    if mvpt_median == median:
        print("invalid case: mvpt aligned with true median")
        return
    else:
        print("valid case: mvpt NOT aligned with true median")

    min_val = 1
    max_val = 0
    for idx_f in range(N_f):
        if firm_info[idx_f]["benchmark"] == 1:
            min_val = min(min_val, sal_bench_l)
            max_val = max(max_val, sal_bench_u)
        elif firm_info[idx_f]["benchmark"] == 0:
            min_val = min(min_val, firm_info[idx_f]["ind-range"][0])
            max_val = max(max_val, firm_info[idx_f]["ind-range"][1])
        elif firm_info[idx_f]["benchmark"] == 2:
            min_val = min(min_val, firm_info[idx_f]["mvpt-range"][0])
            max_val = max(max_val, firm_info[idx_f]["mvpt-range"][1])
        else:
            print(f"Error, invalid benchmark: {firm_info[idx_f]["benchmark"]}.")
    
    sharing_incorrect_R = 0
    sharing_incorrect_S = 0
    no_sharing_incorrect_R = 0
    no_sharing_incorrect_S = 0
    total_sharing_R = 0
    total_no_sharing_R = 0
    total_sharing_S = 0
    total_no_sharing_S = 0
    A_w_sharing = ["share", "no share"]
    for idx_w in range(N_w):
        state = (worker_info[idx_w]["wage"], float(min_val), float(max_val))
        action_index = np.argmax([worker_info[idx_w]["q-vals"][f"({state},{a})"] for a in A_w_sharing])
        action = A_w_sharing[action_index]

        idx_f = int(worker_info[idx_w]["firm-choice"])
        if type_conditioning:
            if worker_info[idx_w]["type"] == "risky":
                at = firm_info[idx_f]["AT"][0]
            else:
                at = firm_info[idx_f]["AT"][1]
        else:
            at = firm_info[idx_f]["AT"]
        
        if action == "share":
            if worker_info[idx_w]["type"] == "risky":
                total_sharing_R = total_sharing_R + 1
            else:
                total_sharing_S = total_sharing_S +1
            if worker_info[idx_w]["wage"] < at and (worker_info[idx_w]["offer"] < at or worker_info[idx_w]["offer"]=="no offer") : # asking too much doesn't happen in a stable market)
                if worker_info[idx_w]["type"] == "risky":
                    sharing_incorrect_R = sharing_incorrect_R +1
                else:
                    sharing_incorrect_S = sharing_incorrect_S +1
        else:
            if worker_info[idx_w]["type"] == "risky":
                total_no_sharing_R = total_no_sharing_R + 1
            else:
                total_no_sharing_S = total_no_sharing_S +1
            if worker_info[idx_w]["wage"] < at and (worker_info[idx_w]["offer"] < at or worker_info[idx_w]["offer"]=="no offer"): # asking too much doesn't happen in a stable market)
                if worker_info[idx_w]["type"] == "risky":
                    no_sharing_incorrect_R = no_sharing_incorrect_R +1
                else:
                    no_sharing_incorrect_S = no_sharing_incorrect_S +1
    print(f"Number of sharing R workers with incorrect beliefs: {sharing_incorrect_R} / {total_sharing_R}")
    print(f"Number of sharing S workers with incorrect beliefs: {sharing_incorrect_S} / {total_sharing_S}")
    print(f"Number of no sharing R workers with incorrect beliefs: {no_sharing_incorrect_R} / {total_no_sharing_R}")
    print(f"Number of no sharing S workers with incorrect beliefs: {no_sharing_incorrect_S} / {total_no_sharing_S}")


def track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u,mvpt_values,N_w,N_f,use_mvpt):


    ## Questions to answer
    # do (types of) workers get high offers rejected more frequently in the first (X) time steps in setting 3 than setting 2?
    tau = 19999
    rejected_offers = [0 for i in range(N_w)]
    no_offers = [0 for i in range(N_w)]
    offer_differential = [[] for i in range(N_w)]
    rejected_R = []
    rejected_S = []
    no_off_R = []
    no_off_S = []
    off_diff_R = []
    off_diff_S = []
    if use_mvpt:
        mvpt_over_estimates = [0 for i in range(N_w)]  # mvpt value > AT of chosen firm
        m_o_e_R = []
        m_o_e_S = []
    for i in range(N_w):
        for t in range(tau):
            f_idx = worker_info[i]["firm-choices"][t]
            if use_mvpt:
                if worker_info[i]["type"] == "risky" and mvpt_values[t] > firm_info[f_idx]["AT"][t][0]:
                     mvpt_over_estimates[i] =  mvpt_over_estimates[i] +1
                if worker_info[i]["type"] == "safe" and mvpt_values[t] > firm_info[f_idx]["AT"][t][1]:
                     mvpt_over_estimates[i] =  mvpt_over_estimates[i] +1
            if worker_info[i]["offers"][t] == -1:
                no_offers[i] = no_offers[i]+1
            else:
                offer_differential[i].append(worker_info[i]["offers"][t] - worker_info[i]["wages"][t])
                if worker_info[i]["offers"][t] != worker_info[i]["wages"][t+1]: # indicates offer was rejected
                    rejected_offers[i] = rejected_offers[i] + 1
        # assert int(len(offer_differential[i])) == int(tau-no_offers[i]),print(f"{len(offer_differential[i])}, {tau-no_offers[i]}")
        if worker_info[i]["type"] == "risky":
            rejected_R.append(rejected_offers[i]/(tau-no_offers[i])) # when you do make an offer
            no_off_R.append(no_offers[i]/tau)
            off_diff_R.append(sum(offer_differential[i])/len(offer_differential[i]))
            if use_mvpt:
                m_o_e_R.append(mvpt_over_estimates[i]/tau)
        else:
            rejected_S.append(rejected_offers[i]/(tau-no_offers[i])) # when you do make an offer
            no_off_S.append(no_offers[i]/tau)
            off_diff_S.append(sum(offer_differential[i])/len(offer_differential[i]))
            if use_mvpt:
                m_o_e_S.append(mvpt_over_estimates[i]/tau)


    # are salary ranges wider over time in setting 3 than setting 2?
    range_length = [[] for i in range(N_f)]
    upper_bound = [[] for i in range(N_f)]
    lower_bound = [[] for i in range(N_f)]
    used_largest_u = [0 for i in range(N_f)]
    used_largest_r = [0 for i in range(N_f)]
    r_l = []
    u_b = []
    l_b = []
    u_l_u = []
    u_l_r = []
    prop_identical = []
    if use_mvpt:
        mvpt_widest_range = [0 for i in range(N_f)] # range widest
        mvpt_lowest_u = [0 for i in range(N_f)] # upper bound strictly smaller than other two
        m_w_r = []
        m_l_u = []
    for i in range(N_f):
        num_identical = 0
        for t in range(tau):
            ind_range = firm_info[i]["ind-range"][1][t] - firm_info[i]["ind-range"][0][t]
            sal_range = sal_bench_u[t] - sal_bench_l[t]
            largest_r = max(ind_range, sal_range)
            largest_u = max(firm_info[i]["ind-range"][1][t],sal_bench_u[t])
            if ind_range == sal_range and firm_info[i]["ind-range"][1][t]  == sal_bench_u[t]:
                identical = 1
                num_identical = num_identical + 1
            else:
                identical = 0
            if use_mvpt:
                mvpt_range = firm_info[i]["mvpt-range"][1][t] - firm_info[i]["mvpt-range"][0][t]
                largest_u = max(largest_u, firm_info[i]["mvpt-range"][1][t])
                largest_r = max(largest_r, mvpt_range)
                if (mvpt_range > sal_range and mvpt_range > ind_range):
                    mvpt_widest_range[i] = mvpt_widest_range[i] +1 
                if (firm_info[i]["mvpt-range"][1][t] < firm_info[i]["ind-range"][1][t] and firm_info[i]["mvpt-range"][1][t] < sal_bench_u[t]):
                    mvpt_lowest_u[i] = mvpt_lowest_u[i] +1
                if identical:
                    identical = (mvpt_range == ind_range) and (firm_info[i]["mvpt-range"][1][t] == sal_bench_u[t])
                    if not identical:
                        num_identical = num_identical - 1 # take away count above
            if firm_info[i]["benchmark"][t] == 0:
                range_length[i].append(ind_range)
                upper_bound[i].append(firm_info[i]["ind-range"][1][t])
                lower_bound[i].append(firm_info[i]["ind-range"][0][t])
                if firm_info[i]["ind-range"][1][t] == largest_u and not identical:
                    used_largest_u[i] = used_largest_u[i] + 1
                if ind_range == largest_r and not identical:
                    used_largest_r[i] = used_largest_r[i] + 1
            elif firm_info[i]["benchmark"][t] == 1:
                range_length[i].append(sal_range)
                upper_bound[i].append(sal_bench_u[t])
                lower_bound[i].append(sal_bench_l[t])
                if sal_bench_u[t] == largest_u and not identical:
                    used_largest_u[i] = used_largest_u[i] + 1
                if sal_range == largest_r and not identical:
                    used_largest_r[i] = used_largest_r[i] + 1
            else:
                range_length[i].append(mvpt_range)
                upper_bound[i].append(firm_info[i]["mvpt-range"][1][t])
                lower_bound[i].append(firm_info[i]["mvpt-range"][0][t])
                if firm_info[i]["mvpt-range"][1][t] == largest_u and not identical:
                    used_largest_u[i] = used_largest_u[i] + 1
                if mvpt_range == largest_r and not identical:
                    used_largest_r[i] = used_largest_r[i] + 1
        
        r_l.append(sum(range_length[i])/tau)
        u_b.append(sum(upper_bound[i])/tau)
        l_b.append(sum(lower_bound[i])/tau)
        if tau-num_identical >0:
            u_l_u.append(used_largest_u[i]/(tau-num_identical)) # track among those where there was a choice
            u_l_r.append(used_largest_r[i]/(tau-num_identical)) # track among those where there was a choice
        if use_mvpt:
            m_w_r.append(mvpt_widest_range[i]/tau)
            m_l_u.append(mvpt_lowest_u[i]/tau)
        prop_identical.append(num_identical/tau)
    
    if use_mvpt:
        return r_l, u_b,l_b, u_l_u, u_l_r,prop_identical, m_w_r,m_l_u,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S, m_o_e_R,m_o_e_S

    return r_l,u_b,l_b, u_l_u, u_l_r, prop_identical,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S




def get_worker_wages(worker_info,N_w):
    
    wages_R = []
    wages_S = []
    for i in range(N_w):
        if worker_info[i]["type"] == "risky":
            wages_R.append(worker_info[i]["wage"])
        else:
            wages_S.append(worker_info[i]["wage"])
    return wages_R,wages_S


seed=42

betas = [(6.91*10**(-4),1.15*10**(-3))]#9.21*10**(-4)
beta_labels = ["slow-fast"]#, "medium","fast"]

settings = [(False, True, False),(True, True, False),(True, True, True)] # (False,False, False),(False, True, False),
setting_label = ["setting 2","setting 3","setting 4"] # "setting 1","setting 2"

riskiness = [(0.75,0.25)]#(0.25,0.5), (0.5,0.25),,(0.25,0.25)](0.75,0.5),,(0.5,0.5)
riskiness_label = ["r-s"]#"r-r","s-r","e-s", "s-s",,"e-r"

p_labels = ["k-r","s-k-r","s-k-l","k-l","u", "b-e", "b-l","b-r"] 

# fixed across settings 
N_w = 100 # small number of workers to start
N_f = 5 # small number of firms
k = 5 # number of intervals to break [0,1] up into
W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
ranges = W + [-1] # -1 indicates no range given 
alpha = 0.3 # more weight on present rewards
delta = 0.9 # more patient
initial_belief_strength = 0.05
p_s = [1/(k+1) for i in range(k+1)] # this parameter doesn't matter, but just for initializing market
type_conditioning = True


# change these indices to change settings
p_l = "u"
N = 15
b_label = beta_labels[0]

for r,r_label in enumerate(riskiness_label):
    for s,s_label in enumerate(setting_label):
        print(f"evaluating {s_label}...")
        # across_all_distributions_surplus = []
        # across_all_distributions_gap = []
        # set values
        use_mvpt,posting,mixed_posts  = settings[s]
        beta = betas[0]
        p_risky, p_reset = riskiness[r]

        # dev_test_firm_profits = [[]for i in range(N_f)] # for each firm, collect t
        # dev_test_firm_applicants = [[] for i in range(N_f)]
        # dev_test_worker_surplus_change = []
        all_p_R = []
        all_p_S = []
        all_p_no_R = []
        all_p_no_S = []
        all_p_od_R = []
        all_p_od_S = []
        all_p_r_l = []
        all_p_u_b = []
        all_p_l_b = []
        all_p_u_l_u = []
        all_p_u_l_r = []
        all_p_p_ident = []
        if use_mvpt:
            all_p_m_w_r = []
            all_p_m_l_u = []
            all_p_m_o_e_R = []
            all_p_m_o_e_S = []
        for p_l in p_labels:
            print(f"evaluating {p_l}...")
            # avg_worker_surplus = []
            # avg_wage_gap = []
            for n in range(N):
                # print("--Firm Deviation Test--")
                folder = "simulation_output_data"
                run_folder= f"N_w={N_w}_N_f={N_f}_k={k}_{s_label}_initial_dist={p_l}_beta={b_label}_risk={r_label}_seed={seed}_N={n}"

                if use_mvpt:
                    with open(f"{folder}/{run_folder}/mvpt_values.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/median_values.pkl","rb") as pkl_fl2:
                        mvpt_values = list(pickle.load(pkl_fl1))
                        medians = list(pickle.load(pkl_fl2))

                firm_info, worker_info, sal_bench_l, sal_bench_u = load_firm_worker_info(folder, run_folder, use_mvpt,type_conditioning)
                
                if use_mvpt:
                    r_l, u_b,l_b, u_l_u, u_l_r,p_ident, m_w_r,m_l_u,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S,m_o_e_R,m_o_e_S = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, mvpt_values,N_w, N_f,use_mvpt)
                    all_p_m_w_r.append(sum(m_w_r)/len(m_w_r))
                    all_p_m_l_u.append(sum(m_l_u)/len(m_l_u))
                    all_p_m_o_e_R.append(sum(m_o_e_R)/len(m_o_e_R))
                    all_p_m_o_e_S.append(sum(m_o_e_S)/len(m_o_e_S))
                else:
                    r_l,u_b,l_b, u_l_u, u_l_r,p_ident,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, None,N_w, N_f,use_mvpt)

                all_p_R.append(sum(rejected_R)/len(rejected_R))
                all_p_S.append(sum(rejected_S)/len(rejected_S))
                all_p_no_R.append(sum(no_off_R)/len(no_off_R))
                all_p_no_S.append(sum(no_off_S)/len(no_off_S))
                all_p_od_R.append(sum(off_diff_R)/len(off_diff_R))
                all_p_od_S.append(sum(off_diff_S)/len(off_diff_S))
                all_p_r_l.append(sum(r_l)/len(r_l))
                all_p_u_b.append(sum(u_b)/len(u_b))
                all_p_l_b.append(sum(l_b)/len(l_b))
                all_p_u_l_u.append(sum(u_l_u)/len(u_l_u))
                all_p_u_l_r.append(sum(u_l_r)/len(u_l_r))
                all_p_p_ident.append(sum(p_ident)/len(p_ident))


               
                # avg_worker_surplus.append((1/N_w)*sum(wages_R + wages_S))
                # avg_wage_gap.append((1/idx_R)*sum(wages_R) - (1/idx_S)*sum(wages_S))
                # across_all_distributions_surplus.append((1/N_w)*sum(wages_R + wages_S))
                # across_all_distributions_gap.append((1/idx_R)*sum(wages_R) - (1/idx_S)*sum(wages_S))
                

                # print("loading data...")
                ## Firm Deviation Test
            #    firm_deviation_test(params)
                


                ## Worker belief test
                # Q: any correlation between MVPT settings and incorrect worker beliefs? -- those that share vs. those that do not
                # if use_mvpt:
                    

        
        #     print("--INTERIM STATS--")
        #     print(f"Setting: {s_label}, riskiness: {r_label}, distribution: {p_l}")
        #     print(f"Average avg worker surplus: {sum(avg_worker_surplus)/len(avg_worker_surplus)}")
        #     print(f"Average avg wage gap: {sum(avg_wage_gap)/len(avg_wage_gap)}")
        print("--FINAL STATS--")
        print(f"Setting: {s_label}, riskiness: {r_label}")
        print(f"average proportion of rejected offers risky: {sum(all_p_R)/len(all_p_R)}")
        print(f"average proportion of rejected offers safe: {sum(all_p_S)/len(all_p_S)}")
        print(f"average proportion of no offers risky: {sum(all_p_no_R)/len(all_p_no_R)}")
        print(f"average proportion of no offers safe: {sum(all_p_no_S)/len(all_p_no_S)}")
        print(f"average size of offer difference risky: {sum(all_p_od_R)/len(all_p_od_R)}")
        print(f"average size of offer difference safe: {sum(all_p_od_S)/len(all_p_od_S)}")
        print(f"average range length: {sum(all_p_r_l)/len(all_p_r_l)}")
        print(f"average upper bound: {sum(all_p_u_b)/len(all_p_u_b)}")
        print(f"average lower bound: {sum(all_p_l_b)/len(all_p_l_b)}")
        print(f"average proportion of using largest upper bound range: {sum(all_p_u_l_u)/len(all_p_u_l_u)}")
        print(f"average proportion of using widest range: {sum(all_p_u_l_r)/len(all_p_u_l_r)}")
        print(f"average proportion of identical sal bench and ind ranges: {sum(all_p_p_ident)/len(all_p_p_ident)}")
        if use_mvpt:
            print(f"average proportion of mvpt with widest range: {sum(all_p_m_w_r)/len(all_p_m_w_r)}")
            print(f"average proportion of mvpt with lowest upper bound: {sum(all_p_m_l_u)/len(all_p_m_l_u)}")
            print(f"average proportion of mvpt over estimates risky: {sum(all_p_m_o_e_R)/len(all_p_m_o_e_R)}")
            print(f"average proportion of mvpt over estimates safe: {sum(all_p_m_o_e_S)/len(all_p_m_o_e_S)}")
        print()
        # # print(f"Average avg worker surplus: {sum(across_all_distributions_surplus)/len(across_all_distributions_surplus)}")
        # # print(f"Average avg wage gap: {sum(across_all_distributions_gap)/len(across_all_distributions_gap)}")
        # print(f"Average worker surplus change after deviation: {sum(dev_test_worker_surplus_change)/len(dev_test_worker_surplus_change)}")

        # avg_firm_profts_after_deviation = [[]for i in range(N_f)]
        # avg_firm_applicants_after_deviation = [[]for i in range(N_f)]
        # for i in range(N_f):
        #     for t in range(T):
        #         prof_at_t = [p[t] for p in dev_test_firm_profits[i]]
        #         ap_at_t = [a[t] for a in dev_test_firm_applicants[i]]
        #         avg_p_at_t = sum(prof_at_t)/len(prof_at_t)
        #         avg_a_at_t = sum(ap_at_t)/len(ap_at_t)
        #         avg_firm_profts_after_deviation[i].append(avg_p_at_t)
        #         avg_firm_applicants_after_deviation[i].append(avg_a_at_t)
        #     if i == 0:
        #         plt.plot(range(T), avg_firm_profts_after_deviation[i], label="deviator",color="red")
        #         # plt.plot(range(T),avg_firm_applicants_after_deviation[i], linestyle ="dashed",color="red")
        #     else:
        #         plt.plot(range(T), avg_firm_profts_after_deviation[i], label="other",color="blue")
        #         # plt.plot(range(T),avg_firm_applicants_after_deviation[i], linestyle ="dashed",color="blue")
        # plt.legend()
        # plt.xlabel("time after deviation")
        # plt.ylabel("average profits")
        # plt.ylim((-1,1))
        # plt.show()

        # for i in range(N_f):
        #     if i == 0:
        #         # plt.plot(range(T), avg_firm_profts_after_deviation[i], label="deviator",color="red")
        #         plt.plot(range(T),avg_firm_applicants_after_deviation[i], label="deviator",linestyle ="dashed",color="red")
        #     else:
        #         # plt.plot(range(T), avg_firm_profts_after_deviation[i], label="other",color="blue")
        #         plt.plot(range(T),avg_firm_applicants_after_deviation[i], linestyle ="dashed",color="blue")
        # plt.legend()
        # plt.xlabel("time after deviation")
        # plt.ylabel("average applicants")
        # plt.ylim((0,N_w))
        # plt.show()











 


