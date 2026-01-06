import pickle
from tqdm import tqdm
import labor_market_q_learning_simulation
from labor_market_q_learning_simulation import Market, Firm, Worker
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_firm_worker_info(folder, run_folder, use_mvpt, type_conditioning):

    firm_info = [{"q-vals":None,"ind-range":None, "profit":None,"benchmark":None,"AT":None}for i in range(N_f)]
    for i in range(N_f):
        with  open(f"{folder}/{run_folder}/ind_bench_l_{i}.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/ind_bench_u_{i}.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/firm_{i}_profits.pkl","rb") as pkl_fl4, open(f"{folder}/{run_folder}/firm_{i}_benchmarks.pkl","rb") as pkl_fl7,open(f"{folder}/{run_folder}/firm_{i}_ATs.pkl","rb") as pkl_fl8:
            # firm_info[i]["q-vals"] = dict(pickle.load(pkl_fl1)) open(f"{folder}/{run_folder}/q_vals_firm_{i}.pkl","rb") as pkl_fl1,
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
    

    if type_conditioning:
        worker_info = []
        for i in range(N_w):
            worker = dict()
            try: # risky worker
                with open(f"{folder}/{run_folder}/worker_R_{i}_wage.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/worker_R_{i}_firm_choice.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/worker_R_{i}_offer.pkl","rb") as pkl_fl4:
                    worker["wages"] = list(pickle.load(pkl_fl2))
                    worker["firm-choices"] = list(pickle.load(pkl_fl3))
                    worker["offers"] = list(pickle.load(pkl_fl4))
                    worker["type"] = "risky"
                worker_info.append(worker)
            except: # safe worker
                # with open(f"{folder}/{run_folder}/q_vals_worker_{i}_S.pkl","rb") as pkl_fl:
                #     worker_info[i]["q-vals"] = dict(pickle.load(pkl_fl))
                #     worker_info[i]["type"] = "safe"
                #     worker_info[i]["type_idx"] = idx_S
                pass
            worker = dict()
            try:
                with open(f"{folder}/{run_folder}/worker_S_{i}_wage.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/worker_S_{i}_firm_choice.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/worker_S_{i}_offer.pkl","rb") as pkl_fl4:
                    worker["wages"] = list(pickle.load(pkl_fl2))
                    worker["firm-choices"] = list(pickle.load(pkl_fl3))
                    worker["offers"] = list(pickle.load(pkl_fl4))
                    worker["type"] = "safe"
                worker_info.append(worker)
            except:
                continue
                
    else:
        worker_info = [{"wages":None,"firm-choices":None,"offers":None} for i in range(N_w)]
        for i in range(N_w):
            with open(f"{folder}/{run_folder}/worker_{i}_wage.pkl","rb") as pkl_fl2, open(f"{folder}/{run_folder}/worker_{i}_firm_choice.pkl","rb") as pkl_fl3, open(f"{folder}/{run_folder}/worker_{i}_offer.pkl","rb") as pkl_fl4:
                    worker_info[i]["wages"] = list(pickle.load(pkl_fl2))
                    worker_info[i]["firm-choices"] = list(pickle.load(pkl_fl3))
                    worker_info[i]["offers"] = list(pickle.load(pkl_fl4))
    
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


def track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u,mvpt_values,N_w,N_f,use_mvpt,type_conditioning):


    ## Questions to answer
    # do (types of) workers get high offers rejected more frequently in the first (X) time steps in setting 3 than setting 2?
    tau = 19999
    rejected_offers = [0 for i in range(N_w)]
    no_offers = [0 for i in range(N_w)]
    offer_differential = [[] for i in range(N_w)]

    if type_conditioning:
        rejected_R = []
        rejected_S = []
        no_off_R = []
        no_off_S = []
        off_diff_R = []
        off_diff_S = []
        m_o_e_R = []
        m_o_e_S = []
    else:
        props_rejected = []
        props_no_offer = []
        avg_off_diff = []
        props_mvpt_oe = []
    if use_mvpt:
        mvpt_over_estimates = [0 for i in range(N_w)]  # mvpt value > AT of chosen firm
    for i in range(N_w):
        for t in range(tau):
            f_idx = worker_info[i]["firm-choices"][t]
            # mvpt over-estimate check 
            if use_mvpt:
                if type_conditioning:
                    if worker_info[i]["type"] == "risky" and mvpt_values[t] > firm_info[f_idx]["AT"][t][0]:
                        mvpt_over_estimates[i] =  mvpt_over_estimates[i] +1
                    if worker_info[i]["type"] == "safe" and mvpt_values[t] > firm_info[f_idx]["AT"][t][1]:
                        mvpt_over_estimates[i] =  mvpt_over_estimates[i] +1
                else:
                    if mvpt_values[t] > firm_info[f_idx]["AT"][t]:
                        mvpt_over_estimates[i] =  mvpt_over_estimates[i] +1
            # no-offer and offer differential check
            if worker_info[i]["offers"][t] == -1:
                no_offers[i] = no_offers[i]+1
            else:
                offer_differential[i].append(worker_info[i]["offers"][t] - worker_info[i]["wages"][t])
                if worker_info[i]["offers"][t] != worker_info[i]["wages"][t+1]: # indicates offer was rejected
                    rejected_offers[i] = rejected_offers[i] + 1
        # assert int(len(offer_differential[i])) == int(tau-no_offers[i]),print(f"{len(offer_differential[i])}, {tau-no_offers[i]}")
        if type_conditioning:
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
        else:
            props_rejected.append(rejected_offers[i]/(tau-no_offers[i]))
            props_no_offer.append(no_offers[i]/tau)
            avg_off_diff.append(sum(offer_differential[i])/len(offer_differential[i]))
            if use_mvpt:
                   props_mvpt_oe.append(mvpt_over_estimates[i]/tau)


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
        if type_conditioning:
            return r_l, u_b,l_b, u_l_u, u_l_r,prop_identical, m_w_r,m_l_u,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S, m_o_e_R,m_o_e_S
        return r_l, u_b,l_b, u_l_u, u_l_r,prop_identical, m_w_r,m_l_u, props_rejected, props_no_offer,avg_off_diff,props_mvpt_oe
    if type_conditioning:
        return r_l,u_b,l_b, u_l_u, u_l_r, prop_identical,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S
    return r_l, u_b,l_b, u_l_u, u_l_r,prop_identical,props_rejected, props_no_offer,avg_off_diff


def plot_wage_differences_per_distribution(setting_info):
    '''assume setting info is list of length 4 and each list gives info for each distribution seperately'''
    pos = [j for i in range(0,48,6) for j in range(i,i+4)]
    # print(pos)
    data_reorged = [[] for i in range(8)]
    for s in setting_info:
        for i in range(8):
            data_reorged[i].append(s[0][i])
    data_final = [d_i for d in data_reorged for d_i in d]
    # print(np.shape(data_final))
    plt.boxplot(data_final,positions=pos)
    plt.xlabel("Distributions and settings")
    plt.xticks([y for y in pos],
                  labels=[ f"{s}-{i}"for s in ["k-r","s-k-r","s-k-l","k-l","u", "b-e", "b-l","b-r"]  for i in range(1,5)], rotation=45)
    plt.ylabel("Average wage gap")
    plt.title("Comparison of distribution of average wage gap per setting.")
    plt.show()

def plot_wage_differences_per_setting(setting_info,y="Average Wage",title="Comparison of distribution of average wage per setting."):
    '''assume setting info is list of length 4 and each list gives info for each distribution seperately'''
    data_reorged = [[] for i in range(4)]
    for j,s in enumerate(setting_info):
        for i in range(8):
            for v in s[0][i]:# save all runs and all distributions to each setting
                data_reorged[j].append(v)
    # data_final = [d_i for d in data_reorged for d_i in d]
    plt.violinplot(data_reorged)
    plt.xlabel("Settings")
    plt.xticks([y+1 for y in range(len(data_reorged))],
                  labels=["Setting 1", "Setting 2", "Setting 3", "Setting 4"], rotation=45)
    plt.ylabel(y)
    # plt.ylim((-0.2,0.5))
    plt.title(title)
    plt.show()

    # test empirical cdf hypothesis 
    for j in range(4):
        plt.ecdf(data_reorged[j],label=f"Setting {j+1} ECDF")
    plt.xlabel("Average Wage")
    plt.ylabel("P(w <= W)")
    plt.title("Empirical CDF Comparison")
    plt.legend()
    plt.show()

    return data_reorged


def plot_offer_differences(setting_info, type_conditioning):
    '''assume setting info is list of length 4 and each list gives info for each distribution seperately

    each bar shows, per distribution, per setting, avg num of no offers  + avg number of rejected offers + avg number of accepted
    '''
    if type_conditioning:
        pos = [j for i in range(0,12,3) for j in range(i,i+2)]
        labels = ["Setting 1 - R", "Setting 1 - S","Setting 2 - R", "Setting 2 - S","Setting 3 - R", "Setting 3 - S","Setting 4 - R", "Setting 4 - S"]
        bottom = np.zeros(len(pos))
    else:
        pos = range(4)
        labels = ["Setting 1","Setting 2","Setting 3", "Setting 4"]
        bottom = np.zeros(len(labels))

    offer_info = {"no-offers":[], "rejected-offers":[],"accepted-offers":[]} # each list is s1:R,S, ..., s4:R,S

    for s in setting_info:
        if type_conditioning:
            no_R = []
            no_S = []
            r_R = []
            r_S = []
        else:
            no = []
            r = []
        for dist in s[0]:
            for run in dist:
                if type_conditioning:
                    no_R.append(run["no-offer-R"])
                    no_S.append(run["no-offer-S"])
                    r_R.append(run["rejected-R"])
                    r_S.append(run["rejected-S"])
                else:
                    no.append(run["no-offer"])
                    r.append(run["rejected"])
        if type_conditioning:
            offer_info["no-offers"].append(sum(no_R)/len(no_R))
            offer_info["no-offers"].append(sum(no_S)/len(no_S))
            offer_info["rejected-offers"].append((1-sum(no_R)/len(no_R)) * sum(r_R)/len(r_R))
            offer_info["rejected-offers"].append((1-sum(no_S)/len(no_S)) * sum(r_S)/len(r_S))
            offer_info["accepted-offers"].append((1-sum(no_R)/len(no_R)) * (1-sum(r_R)/len(r_R)))
            offer_info["accepted-offers"].append((1-sum(no_S)/len(no_S)) * (1-sum(r_S)/len(r_S)))
        else:
            offer_info["no-offers"].append(sum(no)/len(no))
            offer_info["rejected-offers"].append((1-sum(no)/len(no)) * sum(r)/len(r))
            offer_info["accepted-offers"].append((1-sum(no)/len(no)) * (1-sum(r)/len(r)))
    

    for o, o_count in offer_info.items():
        plt.bar(labels, o_count,label=o,bottom=bottom)
        bottom += o_count
    plt.legend()
    plt.show()

def plot_benchmark_choice_comparison(setting_info, stat="Largest"):
    
    pos = range(4)
    labels = [f"Setting 1 - {stat}",f"Setting 2 - {stat}",f"Setting 3 - {stat}",f"Setting 4 - {stat}"]
    bottom = np.zeros(len(labels))

    bench_info = {"identical ranges":[], f"{stat}":[],f"non-{stat}":[]} # each list is s1:R,S, ..., s4:R,S

    for s in setting_info:
        iden = []
        if stat == "Largest":
            large = []
        else:
            wide = []
        for dist in s[0]:
            for run in dist:
                iden.append(run["identical ranges"])
                if stat == "Largest":
                    large.append(run["largest"])
                else:
                    wide.append(run["widest"])
        bench_info["identical ranges"].append(sum(iden)/len(iden))
        if stat == "Largest":
            bench_info["Largest"].append((1-sum(iden)/len(iden)) * sum(large)/len(large))
            bench_info["non-Largest"].append((1-sum(iden)/len(iden)) * (1-sum(large)/len(large)))
        else:
            bench_info["Widest"].append((1-sum(iden)/len(iden)) * sum(wide)/len(wide))
            bench_info["non-Widest"].append((1-sum(iden)/len(iden)) * (1-sum(wide)/len(wide)))
    
    print(bench_info)
    for b, b_count in bench_info.items():
        plt.bar(labels, b_count,label=b,bottom=bottom)
        bottom += b_count
    plt.title("Comparison of Firm Benchmark Choices")
    plt.legend()
    plt.show()


def setting_comparison_statistics(data_reorged):
    '''data_reorged has same 4xnum samples (120 in our case) so that empirical distributions of observations can be compared.'''
    
    for j in range(4):
        print(f"Describing wage gap Setting {j+1}")
        print(stats.describe(data_reorged[j]))

    # # K-S two sample tests, compare all settings with each other
    # print("K-S two sample test")
    # for i in range(4):
    #     if i == 1:
    #         continue
    #     print(f"Comparison of Setting {2} with Setting {i+1}")
    #     print(stats.ks_2samp(data_reorged[i],data_reorged[1],alternative="greater"))
    #     print()
    
    # for i in range(4):
    #     if i == 3:
    #         continue
    #     print(f"Comparison of Setting {4} with Setting {i+1}")
    #     print(stats.ks_2samp(data_reorged[i],data_reorged[3],alternative="greater"))
    #     print()

    
    # print("Mann-Whitney U two sample test")
    # for i in range(4):
    #     if i == 1:
    #         continue
    #     print(f"Comparison of Setting {2} with Setting {i+1}")
    #     print(stats.mannwhitneyu(data_reorged[i],data_reorged[1],alternative="less"))
    #     print()
    
    # for i in range(4):
    #     if i == 3:
    #         continue
    #     print(f"Comparison of Setting {4} with Setting {i+1}")
    #     print(stats.mannwhitneyu(data_reorged[i],data_reorged[3],alternative="less"))
    #     print()





    


    


def get_worker_wages_typed(worker_info,N_w):
    
    wages_R = []
    wages_S = []
    for i in range(N_w):
        if worker_info[i]["type"] == "risky":
            wages_R.append(worker_info[i]["wages"][-1])
        else:
            wages_S.append(worker_info[i]["wages"][-1])
    return wages_R,wages_S

def get_worker_wages(worker_info,N_w):
    
    wages = []
    for i in range(N_w):
        wages.append(worker_info[i]["wages"][-1])
    return wages


seed=42

betas = [(6.91*10**(-4))]#,1.15*10**(-3))]#med - 9.21*10**(-4)(6.91*10**(-4),1.15*10**(-3))
beta_labels = ["slow-fast"]#, "medium","fast"]

settings = [(False,False, False),(False, True, False),(True, True, False),(True, True, True)] # ,(False, True, False),
setting_label = ["setting 1","setting 2","setting 3","setting 4"] # "setting 2"

riskiness = [(0.75,0.25)]#(0.75,0.5),(0.25,0.5), (0.5,0.25),,(0.25,0.25),(0.75,0.5),(0.75,0.25),(0.5,0.5)
riskiness_label = ["r-s"]#"r-r","s-r","e-s", "s-s","r-s","e-r"

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
setting_data_1 = [[] for i in range(len(setting_label))]
setting_data_2 = [[] for i in range(len(setting_label))]
for r,r_label in enumerate(riskiness_label):
    for s,s_label in enumerate(setting_label):
        print(f"evaluating {s_label}...")
        distribution_data_1 = [[] for i in range(len(p_labels))]
        distribution_data_2 = [[] for i in range(len(p_labels))]
        # across_all_distributions_surplus = []
        # across_all_distributions_gap = []
        # set values
        use_mvpt,posting,mixed_posts  = settings[s]
        beta = betas[0]
        p_risky, p_reset = riskiness[r]

        # dev_test_firm_profits = [[]for i in range(N_f)] # for each firm, collect t
        # dev_test_firm_applicants = [[] for i in range(N_f)]
        # dev_test_worker_surplus_change = []
        if type_conditioning:
            all_p_R = []
            all_p_S = []
            all_p_no_R = []
            all_p_no_S = []
            all_p_od_R = []
            all_p_od_S = []
            all_p_m_o_e_R = []
            all_p_m_o_e_S = []
        else:
            all_p_rej = []
            all_p_no = []
            all_p_od = []
            all_p_m_o_e = []
        all_p_r_l = []
        all_p_u_b = []
        all_p_l_b = []
        all_p_u_l_u = []
        all_p_u_l_r = []
        all_p_p_ident = []
        if use_mvpt:
            all_p_m_w_r = []
            all_p_m_l_u = []
        for p,p_l in enumerate(p_labels):
            print(f"evaluating {p_l}...")
            # avg_wage_gap = []
            for n in range(N):
                # print("--Firm Deviation Test--")
                folder = "simulation_output_data_experiment_2_asym_betas"
                run_folder= f"N_w={N_w}_N_f={N_f}_k={k}_{s_label}_initial_dist={p_l}_beta={b_label}_risk={r_label}_seed={seed}_N={n}"

                if use_mvpt:
                    with open(f"{folder}/{run_folder}/mvpt_values.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/median_values.pkl","rb") as pkl_fl2:
                        mvpt_values = list(pickle.load(pkl_fl1))
                        medians = list(pickle.load(pkl_fl2))
                
                firm_info, worker_info, sal_bench_l, sal_bench_u = load_firm_worker_info(folder, run_folder, use_mvpt,type_conditioning)
                all_wages  = get_worker_wages(worker_info,N_w)
                # wages_R,wages_S = get_worker_wages_typed(worker_info,N_w)

                if type_conditioning:
                    AT_gaps = []
                    for i in range(N_f):
                        AT_gaps.append(firm_info[i]["AT"][-1][0] - firm_info[i]["AT"][-1][1]) # risky - safe AT at last time step



                if use_mvpt:
                    if type_conditioning:
                        r_l, u_b,l_b, u_l_u, u_l_r,p_ident, m_w_r,m_l_u,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S,m_o_e_R,m_o_e_S = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, mvpt_values,N_w, N_f,use_mvpt,type_conditioning)
                        all_p_m_w_r.append(sum(m_w_r)/len(m_w_r))
                        all_p_m_l_u.append(sum(m_l_u)/len(m_l_u))
                        # all_p_m_o_e_R.append(sum(m_o_e_R)/len(m_o_e_R))
                        # all_p_m_o_e_S.append(sum(m_o_e_S)/len(m_o_e_S))
                    else:
                        r_l, u_b,l_b, u_l_u, u_l_r,p_ident, m_w_r,m_l_u, props_rejected, props_no_offer,avg_off_diff,props_mvpt_oe = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, mvpt_values,N_w, N_f,use_mvpt,type_conditioning)
                        all_p_m_w_r.append(sum(m_w_r)/len(m_w_r))
                        all_p_m_l_u.append(sum(m_l_u)/len(m_l_u))
                        # all_p_m_o_e.append(sum(props_mvpt_oe)/len(props_mvpt_oe))
                else:
                    if type_conditioning:
                        r_l,u_b,l_b, u_l_u, u_l_r,p_ident,rejected_R, rejected_S, no_off_R, no_off_S, off_diff_R,off_diff_S = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, None,N_w, N_f,use_mvpt,type_conditioning)
                    else:
                        r_l, u_b,l_b, u_l_u, u_l_r,p_ident, props_rejected, props_no_offer,avg_off_diff = track_worker_behavior(worker_info, firm_info, sal_bench_l,sal_bench_u, None,N_w, N_f,use_mvpt,type_conditioning)

                # if type_conditioning:
                #     all_p_R.append(sum(rejected_R)/len(rejected_R))
                #     all_p_S.append(sum(rejected_S)/len(rejected_S))
                #     all_p_no_R.append(sum(no_off_R)/len(no_off_R))
                #     all_p_no_S.append(sum(no_off_S)/len(no_off_S))
                #     all_p_od_R.append(sum(off_diff_R)/len(off_diff_R))
                #     all_p_od_S.append(sum(off_diff_S)/len(off_diff_S))
                # else:
                #     all_p_rej.append(sum(props_rejected)/len(props_rejected))
                #     all_p_no.append(sum(props_no_offer)/len(props_no_offer))
                #     all_p_od.append(sum(avg_off_diff)/len(avg_off_diff))
                all_p_r_l.append(sum(r_l)/len(r_l))
                all_p_u_b.append(sum(u_b)/len(u_b))
                all_p_l_b.append(sum(l_b)/len(l_b))
                all_p_u_l_u.append(sum(u_l_u)/len(u_l_u))
                all_p_u_l_r.append(sum(u_l_r)/len(u_l_r))
                all_p_p_ident.append(sum(p_ident)/len(p_ident))


               
                distribution_data_1[p].append((1/N_w)*sum(all_wages))
                distribution_data_2[p].append(sum(AT_gaps)/len(AT_gaps))
                # distribution_data_2[p].append({"identical ranges":sum(p_ident)/len(p_ident),"largest":sum(u_l_u)/len(u_l_u),"widest":sum(u_l_r)/len(u_l_r)})
                # distribution_data_2[p].append(sum(wages_R)/len(wages_R) - sum(wages_S)/len(wages_S))
                # if type_conditioning:
                #     p_no_R = sum(no_off_R)/len(no_off_R)
                #     p_no_S = sum(no_off_S)/len(no_off_S)
                #     p_r_R = sum(rejected_R)/len(rejected_R)
                #     p_r_S = sum(rejected_S)/len(rejected_S)
                #     distribution_data[p].append({"no-offer-R":p_no_R,"no-offer-S":p_no_S,"rejected-R":p_r_R,"rejected-S":p_r_S})
                # else:
                #     p_no = sum(props_no_offer)/len(props_no_offer)
                #     p_r = sum(props_rejected)/len(props_rejected)
                #     distribution_data[p].append({"no-offer":p_no,"rejected":p_r,})

        setting_data_1[s].append(distribution_data_1)
        setting_data_2[s].append(distribution_data_2)
        all_at_gaps = [at_gap for all_gaps in distribution_data_2 for at_gap in all_gaps]
                # across_all_distributions_surplus.append((1/N_w)*sum(all_wages))
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
        # #     print(f"Average avg wage gap: {sum(avg_wage_gap)/len(avg_wage_gap)}")
        print("--FINAL STATS--")
        print(f"Setting: {s_label}, riskiness: {r_label}")
        # print(f"Average avg worker surplus: {sum(across_all_distributions_surplus)/len(across_all_distributions_surplus)}")
        # if type_conditioning:
        #     print(f"average proportion of rejected offers risky: {sum(all_p_R)/len(all_p_R)}")
        #     print(f"average proportion of rejected offers safe: {sum(all_p_S)/len(all_p_S)}")
        #     print(f"average proportion of no offers risky: {sum(all_p_no_R)/len(all_p_no_R)}")
        #     print(f"average proportion of no offers safe: {sum(all_p_no_S)/len(all_p_no_S)}")
        #     print(f"average size of offer difference risky: {sum(all_p_od_R)/len(all_p_od_R)}")
        #     print(f"average size of offer difference safe: {sum(all_p_od_S)/len(all_p_od_S)}")
        # else:
        #     print(f"average proportion of rejected offers: {sum(all_p_rej)/len(all_p_rej)}")
        #     print(f"average proportion of no offers: {sum(all_p_no)/len(all_p_no)}")
        #     print(f"average size of offer difference with wage: {sum(all_p_od)/len(all_p_od)}")
        print(f"average range length: {sum(all_p_r_l)/len(all_p_r_l)}")
        print(f"average upper bound: {sum(all_p_u_b)/len(all_p_u_b)}")
        print(f"average lower bound: {sum(all_p_l_b)/len(all_p_l_b)}")
        print(f"average proportion of using largest upper bound range: {sum(all_p_u_l_u)/len(all_p_u_l_u)}")
        print(f"average proportion of using widest range: {sum(all_p_u_l_r)/len(all_p_u_l_r)}")
        print(f"average proportion of identical sal bench and ind ranges: {sum(all_p_p_ident)/len(all_p_p_ident)}")
        if type_conditioning:
            print(f"average AT gap across all firms and all runs/distributions: {sum(all_at_gaps)/len(all_at_gaps)}")
        if use_mvpt:
            print(f"average proportion of mvpt with widest range: {sum(all_p_m_w_r)/len(all_p_m_w_r)}")
            print(f"average proportion of mvpt with lowest upper bound: {sum(all_p_m_l_u)/len(all_p_m_l_u)}")
            # if type_conditioning:
            #     print(f"average proportion of mvpt over estimates risky: {sum(all_p_m_o_e_R)/len(all_p_m_o_e_R)}")
            #     print(f"average proportion of mvpt over estimates safe: {sum(all_p_m_o_e_S)/len(all_p_m_o_e_S)}")
            # else:
            #     print(f"average proportion of mvpt over estimates: {sum(all_p_m_o_e)/len(all_p_m_o_e)}")
        print()
        
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

# data_reorged = plot_wage_differences_per_setting(setting_data_1)
# plot_benchmark_choice_comparison(setting_data_2,stat="Largest")
# plot_benchmark_choice_comparison(setting_data_2,stat="Widest")
# data_reorged = plot_wage_differences_per_setting(setting_data_2,y="Average Wage Gap",title="Comparison of distribution of average wage gap per setting.")
# setting_comparison_statistics(data_reorged)
# plot_offer_differences(setting_data,type_conditioning)




## Extra analysis old code
### Analyzing

## Overall wage analysis 
# Wage distribution final vs. initial
# if type_conditioning:
#     plot_attribute_distribution_market(market,"wage",N_w=N_w,N_f=N_f,initial_counts = (initial_c_R,initial_c_S),initial_bins =(initial_b_R,initial_b_S),split_workers=type_conditioning,k=k,p_l=p_l,s_l = s_label,b_l=b_label,p_reset=p_reset,seed=seed,save=save,folder=folder)
# else:
#     plot_attribute_distribution_market(market,"wage",N_w=N_w,N_f=N_f,initial_counts = initial_c,initial_bins =initial_b,split_workers=type_conditioning,k=k,p_l=p_l,s_l = s_label,b_l=b_label,p_reset=p_reset,seed=seed,save=save,folder=folder)

# Median value over time
# if use_mvpt:
#     plt.plot(m_hat_values, label="m_hat median estimate",linestyle="dashed")
# plt.plot(true_median, color="orange",label="True median")
# plt.ylim((0,1))
# plt.title(f"Median wage values over time")
# plt.xlabel("Time")
# plt.ylabel("Median values")
# plt.legend()
# if save:
#     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_median_values_seed={seed}.png")
# plt.clf()


 # all_wages_initial = [wages[0] for wages in worker_wages]
# bot_10 = np.percentile(all_wages_initial,10)
# bot_50 = np.percentile(all_wages_initial,50)
# bot_90 = np.percentile(all_wages_initial,90)

# bot_10_final = np.percentile(all_wages_final,10)
# bot_90_final = np.percentile(all_wages_final,90)
# bot_50_final = np.percentile(all_wages_final,50)

# print(f"initial percentiles: {bot_10}, {bot_50}, {bot_90}")
# print(f"final percentiles: {bot_10_final}, {bot_50_final}, {bot_90_final}")
# if bot_50 >0:
#     print(f"initial dispersion ratio: {(bot_90-bot_10)/bot_50}")
# if bot_50_final >0:
#     print(f"final dispersion ratio: {(bot_90_final-bot_10_final)/bot_50_final}")



## Firm analysis
# firm profit in last 1000 time steps  
# tau = T
# for i in range(N_f):
#     plt.plot(range(T-tau,T), firm_profits[i][T-tau:])
#     plt.title(f"Firm {i} average profits")
#     # plt.ylim((-1,1)) # normalize?
#     plt.xlabel(f"Last {tau} timesteps")
#     plt.ylabel("Firm profit")
#     if save:
#         plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_firm_{i}_profits_seed={seed}.png")
#     plt.clf()


# information use numbers
# sal_bench_numbers = [info_use[0] for info_use in market.firm_information_use]
# independent_numbers = [info_use[1] for info_use in market.firm_information_use]

# plt.plot(range(len(market.firm_information_use)), sal_bench_numbers,label="Num. firms using Salary Benchmark")
# plt.plot(range(len(market.firm_information_use)), independent_numbers,label="Num. firms using Independent data")
# if use_mvpt:
#     mvpt_numbers = [N_f - iu_s - iu_i for iu_s,iu_i in zip(sal_bench_numbers,independent_numbers)]
#     plt.plot(range(len(market.firm_information_use)), mvpt_numbers,label="Num. firms using MVPT data")
# plt.title("Firm information use over time")
# plt.ylim((0,N_f))
# plt.xlabel("Time")
# plt.ylabel("Firm counts per source")
# plt.legend()
# if save:
#     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_information_source_numbers_seed={seed}.png")
# plt.clf()

# find "stability" time to limit size of graph
# stable_t = len(firms_ATs[0])

# if type_conditioning:
#     risky_ATs = [[at[0] for at in firms_ATs[i]]for i in range(N_f)]
#     safe_ATs = [[at[1] for at in firms_ATs[i]]for i in range(N_f)]
#     for t in range(T):
#         stable_r = 1
#         stable_s = 1
#         for i in range(N_f):
#             if min(risky_ATs[i][t:]) != max(risky_ATs[i][t:]):
#                 stable_r = 0
#             if min(safe_ATs[i][t:]) != max(safe_ATs[i][t:]):
#                 stable_s = 0
#         if stable_r and stable_s:
#             stable_t = t
#             break
# else:
#     for t in range(T):
#         stable = 1
#         for i in range(N_f):
#             if min(firms_ATs[i][t:]) != max(firms_ATs[i][t:]):
#                 stable = 0
#         if stable:
#             stable_t = t
#             break
    

# for i in range(N_f):
#     plt.plot(range(stable_t), sal_bench_l[:stable_t], label="Sal. bench low",color="blue")
#     plt.plot(range(stable_t), sal_bench_u[:stable_t], label="Sal. bench high",color="cornflowerblue")
#     plt.plot(range(stable_t), ind_range_l[i][:stable_t], label="Ind. low", color="red")
#     plt.plot(range(stable_t), ind_range_u[i][:stable_t], label="Ind. high", color="lightcoral")
#     if use_mvpt:
#         plt.plot(range(stable_t), mvpt_range_l[i][:stable_t], label="MVPT low", color="purple")
#         plt.plot(range(stable_t), mvpt_range_u[i][:stable_t], label="MVPT high", color="mediumorchid")
#     plt.legend()
#     plt.title(f"Benchmark ranges for firm {i} until stable firm acceptance thresholds")
#     plt.ylim((0,1))
#     plt.ylabel("Values")
#     plt.xlabel("Time")
#     if save:
#         plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_information_source_values_firm{i}_seed={seed}.png")
#     plt.clf()

# firm decisions
# for i in range(len(market.firms)):
#     if type_conditioning:
#         plt.plot(range(stable_t), risky_ATs[i][:stable_t],color ="red",label="acceptance threshold for risky")
#         plt.plot(range(stable_t), safe_ATs[i][:stable_t],color ="blue",label="acceptance threshold for safe")
#     else:
#         plt.plot(range(stable_t), firms_ATs[i][:stable_t],color ="purple",label="acceptance threshold")
#         lower_bound = [r[0] for r in firms_ranges[i][:stable_t]]
#         upper_bound = [r[1] for r in firms_ranges[i][:stable_t]]
#         plt.plot(range(stable_t), lower_bound,color ="red",label="lower bound of range")
#         plt.plot(range(stable_t), upper_bound,color ="blue",label="upper bound of range")
#     plt.ylim((0,1))
#     plt.title(f"Firm index {i} acceptance thresholds over time")
#     plt.legend()
#     if save:
#         plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_firm_{i}_seed={seed}.png")
#     plt.clf()


## Worker analysis 
# average worker surplus over last tau time steps
# avg_worker_surplus = []
# for t in range(T-tau,T):
#     avg_worker_surplus.append((1/N_w) * sum(worker_wages[t]))
# plt.plot(range(T-tau,T), avg_worker_surplus)
# plt.title("Average Worker Surplus")
# plt.ylim((0,1))
# plt.xlabel(f"Last {tau} timesteps")
# plt.ylabel("Worker Surplus")
# if save:
#     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_average_worker_surplus_seed={seed}.png")
# plt.clf()

# if type_conditioning:
    # avg_worker_wage_risky = []
    # avg_worker_wage_safe = []

    # firm_set_risky_x = []
    # firm_set_risky_y = []
    # firm_set_safe_x = []
    # firm_set_safe_y = []

    # for t in range(T-tau,T):
    #     worker_wages_risky = []
    #     worker_wages_safe = []

    #     chosen_firms_risky = set()
    #     chosen_firms_safe = set()

    #     for idx, w in enumerate(market.workers):
    #         if w.type=="risky":
    #             worker_wages_risky.append(worker_wages[t][idx])
    #             if worker_firm_choice[idx][t] != -1:
    #                 chosen_firms_risky.add(worker_firm_choice[idx][t])
    #         else:
    #             worker_wages_safe.append(worker_wages[t][idx])
    #             if worker_firm_choice[idx][t] != -1:
    #                 chosen_firms_safe.add(worker_firm_choice[idx][t])
    #     avg_worker_wage_risky.append(np.mean(worker_wages_risky))
    #     avg_worker_wage_safe.append(np.mean(worker_wages_safe))
    #     for c_f in chosen_firms_risky:
    #         firm_set_risky_x.append(t)
    #         firm_set_risky_y.append(c_f)
    #     for c_f in chosen_firms_safe:
    #         firm_set_safe_x.append(t)
    #         firm_set_safe_y.append(c_f)

    # plot wages
    # plt.plot(range(T-tau,T), avg_worker_wage_risky,label="Avg. Risky Worker Wages")
    # plt.plot(range(T-tau,T), avg_worker_wage_safe,label="Avg. Safe Worker Wages")
    # plt.title("Worker Wage Comparison")
    # plt.ylim((0,1))
    # plt.legend()
    # plt.xlabel(f"Last {tau} timesteps")
    # plt.ylabel("Average Worker Wage")
    # if save:
    #     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_worker_wage_comparison_seed={seed}.png")
    # plt.clf()

    # plot firm choices
    # plt.scatter(firm_set_risky_x,firm_set_risky_y, label="Risky Workers",marker="o",color="red")
    # plt.scatter(firm_set_safe_x,firm_set_safe_y, label="Safe Workers",marker="x",color="blue")
    # plt.title("Worker Firm Negotiation Choices")
    # plt.ylim((0,4))
    # plt.legend()
    # plt.xlabel(f"Last {tau} timesteps")
    # plt.ylabel("Chosen Firms")
    # if save:
    #     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_worker_firm_choices_seed={seed}.png")
    # plt.clf()
    # mvpt use numbers over time
    # if use_mvpt:
    #     plt.plot(range(len(market.worker_mvpt_use)), market.worker_mvpt_use)
    #     plt.ylim((0,N_w))
    #     plt.title("Number of workers sharing with MVPT over time")
    #     if save:
    #         plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_mvpt_sharing_numbers_seed={seed}.png")
    #     plt.clf()

    # stable_t = len(worker_offers[0])
    # for t in range(T):
    #     stable = 1
    #     for i in range(N_f):
    #         if min(worker_offers[i][t:]) != max(worker_offers[i][t:]):
    #             stable = 0
    #     if stable:
    #         stable_t = t
    #         break

    # # sample of offers over time compared to wage over time
    # differentials = [[]for i in range(N_w)]
    # for i in range(N_w):
    #     differentials[i] = [v-w[i] if v >= 0 else -1 for v,w in zip(worker_offers[i],worker_wages)]

    # # proportion of no offers in last 100 time steps by type
    # p_no_offer_R = []
    # p_no_offer_S = []
    # for idx,w in enumerate(market.workers):
    #     if w.type=="risky":
    #         p_no_offer_R.append(sum([1 for d in differentials[idx][T-1000:] if d == -1])/1000)
    #     else:
    #         p_no_offer_S.append(sum([1 for d in differentials[idx][T-1000:] if d == -1])/1000)

    # print("Statistics of proportion of no offer actions in last 1000 time steps by worker type.")
    # print("---Risky Workers---")
    # print(f"min proportion: {min(p_no_offer_R)}")
    # print(f"max proportion: {max(p_no_offer_R)}")
    # print(f"avg. proportion: {np.mean(p_no_offer_R)}")

    # print("---Safe Workers---")
    # print(f"min proportion: {min(p_no_offer_S)}")
    # print(f"max proportion: {max(p_no_offer_S)}")
    # print(f"avg. proportion: {np.mean(p_no_offer_S)}")
    
    # plot distribution of last time step of differential > 0 for both risky and safe workers
    # last_higher_offer_time_step = [T for i in range(N_w)]
    # for t in range(T):
    #     for idx,w in enumerate(market.workers):
    #         if last_higher_offer_time_step[idx] < T:
    #             continue
    #         if max(differentials[idx][t:])==0: # no more high offers
    #             last_higher_offer_time_step[idx] = t
    # l_h_o_t_R = []
    # l_h_o_t_S = []
    # for idx,w in enumerate(market.workers):
    #     if w.type=="risky":
    #         l_h_o_t_R.append(last_higher_offer_time_step[idx])
    #     else:
    #         l_h_o_t_S.append(last_higher_offer_time_step[idx])

    # c_R, b_R = np.histogram(l_h_o_t_R,bins=500,range=(0,T))
    # c_S, b_S = np.histogram(l_h_o_t_S,bins=500,range=(0,T))
    # plt.stairs(c_R,b_R,label=f"Risky",color="red")
    # plt.stairs(c_S,b_S,label=f"Safe",color="blue")
    # plt.title(f"Distribution of last time step of non-zero offer of Workers throughout market")
    # plt.legend()
    # plt.ylim((0,N_w))
    # plt.xlabel(f"last time step, between 0 and {T}")
    # plt.ylabel(f"Density of last time step value throughout market")
    # if save:
    #     plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_last_high_offer_time_step_seed={seed}.png")
    # plt.clf()
        

    # for i in range(0,100,10):
    #     # differential = [v-w[i] if v >= 0 else -1 for v,w in zip(worker_offers[i],worker_wages)]
    #     plt.scatter(range(stable_t),differential[:stable_t],color="blue")
    #     # plt.plot(range(len(wages)), wages, color="red",label="Current wage")
    #     plt.ylim((-1,1))
    #     # plt.legend()
    #     plt.title(f"Worker index {i} offer-wage over time (or no offer). Final employer: {market.workers[i].employer}")
    #     if save:
    #         plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_worker_{i}_seed={seed}.png")
    #     plt.clf()
    
    # plt.close() # close all figures between runs
                    

 # if min(all_wages_final) < 1:
#     print("All workers DID NOT end up at MRPL.")
#     all_wages_less_than_1 = [w for w in all_wages_final if w <1]
#     all_wages_eq_1 = [w for w in all_wages_final if w ==1]
    
#     all_wages_less_than_b10 = [w for w in all_wages_initial if w <=bot_10]
#     all_wages_less_than_b25 = [w for w in all_wages_initial if w <=bot_25]
#     all_wages_less_than_b50 = [w for w in all_wages_initial if w <=bot_50]
#     all_wages_less_than_b75 = [w for w in all_wages_initial if w <=bot_75]
#     all_wages_less_than_b90 = [w for w in all_wages_initial if w <=bot_90 and w > bot_75]

#     wages_at_i = [[i for i in range(len(all_wages_initial)) if all_wages_initial[i] == w] for w in W]
#     wages_at_i_final_eq_1 = [[i for i in range(len(all_wages_final)) if all_wages_final[i] == 1 and all_wages_initial[i] == w] for w in W]
#     wages_at_i_final_l_1 = [[i for i in range(len(all_wages_final)) if all_wages_final[i] < 1 and all_wages_initial[i] == w] for w in W]

#     for i in range(len(W)):
#         if len(wages_at_i[i])>0:
#             print(f"P(w_f <1 | w_i = {W[i]}) = {len(wages_at_i_final_l_1[i])/len(wages_at_i[i])}")
#             print(f"P(w_f ==1 | w_i = {W[i]}) = {len(wages_at_i_final_eq_1[i])/len(wages_at_i[i])}")
#         else:
#             print(f"No initial wages at {W[i]}.")


#     # all_wages_less_than_1_and_initial_bot_10_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_10]
#     # all_wages_less_than_1_and_initial_bot_25_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_25]
#     # all_wages_less_than_1_and_initial_bot_50_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_50] 
#     # all_wages_less_than_1_and_initial_bot_75_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_75] 
#     # all_wages_less_than_1_and_initial_bot_90_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_90 and all_wages_initial[i] > bot_75] 
#     # all_wages_less_than_1_and_initial_less_than_1 = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]< 1] 

#     # all_wages_eq_1_and_initial_bot_75_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] ==1 and all_wages_initial[i]<= bot_75]
#     # all_wages_eq_1_and_initial_bot_90_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] ==1 and all_wages_initial[i]<= bot_90 and all_wages_initial[i] > bot_75] 

#     # print(f"10th percentile={bot_10}, P(w_f <1 | w_i<=bot10) = {len(all_wages_less_than_1_and_initial_bot_10_p)/len(all_wages_less_than_b10)}")
#     # print(f"25th percentile={bot_25}, P(w_f <1 | w_i<=bot25) = {len(all_wages_less_than_1_and_initial_bot_25_p)/len(all_wages_less_than_b25)}")
#     # print(f"50th percentile={bot_50}, P(w_f <1 | w_i<=bot50) = {len(all_wages_less_than_1_and_initial_bot_50_p)/len(all_wages_less_than_b50)}")

#     # if len(all_wages_less_than_b75) >0:
#     #     print(f"75th percentile={bot_75}, P(w_f <1 | w_i<=bot75) = {len(all_wages_less_than_1_and_initial_bot_75_p)/len(all_wages_less_than_b75)}")
#     #     print(f"P(w_f==1| w_i<=bot75) = {len(all_wages_eq_1_and_initial_bot_75_p)/len(all_wages_less_than_b75)}")

#     # if len(all_wages_less_than_b90) >0:
#     #     print(f"90th percentile={bot_90}, P(w_f <1 | w_i<=bot90) = {len(all_wages_less_than_1_and_initial_bot_90_p)/len(all_wages_less_than_b90)}")
#     #     print(f"P(w_f==1| w_i<=bot90) = {len(all_wages_eq_1_and_initial_bot_90_p)/len(all_wages_less_than_b90)}")
# else:
#     print("All workers DID end up at MRPL.")

# m_hat_l_true_med = 0
# m_hat_g_true_med = 0
# m_hat_eq_true_med = 0
# t_last_m_l_t = 0
# t_last_m_g_t = 0
# t_last_m_eq_t = 0

# for m_hat,tm,t in zip(m_hat_values,true_median,range(T)):
#     if m_hat < tm:
#         m_hat_l_true_med  = m_hat_l_true_med  +1
#         t_last_m_l_t = t
#     elif m_hat == tm:
#         m_hat_eq_true_med = m_hat_eq_true_med +1
#         t_last_m_eq_t = t
#     else:
#         m_hat_g_true_med = m_hat_g_true_med +1
#         t_last_m_g_t = t

# print("M_hat vs. true median statistics")
# print(f"P(m_hat < tm) = {m_hat_l_true_med / T} and time of last instance {t_last_m_l_t}")
# print(f"P(m_hat == tm) = {m_hat_eq_true_med / T} and time of last instance {t_last_m_eq_t}")
# print(f"P(m_hat > tm) = {m_hat_g_true_med / T} and time of last instance {t_last_m_g_t}")


# # graphs


# # worker attributes 
# # t_last_share  = []
# # for i in range(N):
# #     workers_actions[i].reverse()
# #     tls = len(workers_actions[i])/2 + (len(workers_actions[i])/2 - workers_actions[i].index(1)-1)

# #     t_last_share.append((tls,worker_wages[0][i],worker_wages[-1][i])) # (tls, initial wage, last wage)

# # last_share = max([t[0] for t in t_last_share])

# # ts_w_i_le_bot_75_eq_1 = [t[0] for t in t_last_share if t[1]<=bot_75 and t[2]==1]
# # ts_w_i_le_bot_75_l_1 = [t[0] for t in t_last_share if t[1]<=bot_75 and t[2]<1]
# # ts_w_i_g_bot_75_le_bot_90_eq_1 = [t[0] for t in t_last_share if t[1]>bot_75 and t[1]<=bot_90 and t[2]==1]
# # ts_w_i_g_bot_75_le_bot_90_l_1 = [t[0] for t in t_last_share if t[1]>bot_75 and t[1]<=bot_90 and t[2] <1]
# # ts_w_i_g_bot_90_eq_1 = [t[0] for t in t_last_share if t[1]>bot_90 and t[2] ==1]
# # ts_w_i_g_bot_90_l_1 = [t[0] for t in t_last_share if t[1]>bot_90 and t[2]<1]

# # ts_w_init_eq_i_f_eq_1 = [[t[0] for t in t_last_share if t[1] == w and t[2] == 1] for w in W]
# # ts_w_init_eq_i_f_l_1 = [[t[0] for t in t_last_share if t[1] == w and t[2] <1] for w in W]

# # colors = ["blue", "red", "black", "orange", "pink", "purple","green", "cyan", "grey", "brown","gold"]
# # for i in range(len(W)):
# #     counts_a, bins_a = np.histogram(ts_w_init_eq_i_f_eq_1[i],bins=50,range=(0,last_share))
# #     counts_b, bins_b = np.histogram(ts_w_init_eq_i_f_l_1[i],bins=50,range=(0,last_share))
# #     plt.stairs(counts_a,bins_a,label=f"initial wage = {W[i]} and final wage=1",color=colors[i])
# #     plt.stairs(counts_b,bins_b,label=f"initial wage = {W[i]} and final wage<1",color=colors[i],linestyle="dashed")
# # # counts_1a, bins_1a = np.histogram(ts_w_i_le_bot_75_eq_1,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # counts_1b, bins_1b = np.histogram(ts_w_i_le_bot_75_l_1,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # counts_2a, bins_2a = np.histogram(ts_w_i_g_bot_75_le_bot_90_eq_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # counts_2b, bins_2b = np.histogram(ts_w_i_g_bot_75_le_bot_90_l_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # counts_3a, bins_3a = np.histogram(ts_w_i_g_bot_90_eq_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # counts_3b, bins_3b = np.histogram(ts_w_i_g_bot_90_l_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
# # # plt.stairs(counts_2a,bins_2a,label=f"initial wage>bot75,<=bot90 and final wage=1",color="purple")
# # # plt.stairs(counts_2b,bins_2b,label=f"initial wage>bot75,<=bot90 and final wage<1",color="purple",linestyle="dashed")
# # # plt.stairs(counts_3a,bins_3a,label=f"initial wage>bot90 and final wage=1",color="blue")
# # # plt.stairs(counts_3b,bins_3b,label=f"initial wage>bot90 and final wage<1",color="blue",linestyle="dashed")
# # plt.title(f"Distribution of time of last share of Workers throughout market")
# # plt.legend()
# # plt.ylim((0,N))
# # plt.xlabel(f"time of last share, up to t={T}")
# # plt.ylabel("Density of time of last share throughout market")
# # if save:
# #     plt.savefig(f"q_learning_simulation_results/time_of_last_share/p=0.1_N={N}_k={k}_initial_distribution={p_l}_tls_distribution_seed={seed}.png")
# # plt.show()




  ## this should really be a function for generating initial wage distributions given percentiles and number of workers
# translate 10th, 25th, 50th, 75th, 90th percentile wages to a distribution
# w10 = 20380		
# w25 = 29120
# w50 = 44810
# w75 = 80350
# w90 = 137820
# N_workers = 26510

# t10 = w10/w90
# t25 = w25/w90
# t50 = w50/w90
# t75 = w75/w90
# t90 = 1











 


