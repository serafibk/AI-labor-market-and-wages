import pickle
from tqdm import tqdm
import labor_market_q_learning_simulation
from labor_market_q_learning_simulation import Market, Firm, Worker
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_firm_worker_info(folder, run_folder, use_mvpt, type_conditioning):
    """Loading saved simulation data for firms and workers"""

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
    """Testing outcomes when one firm deviates from their converged strategy."""

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
    """Testing whether workers converged to a strategy that is a best response to the firms' converged strategies"""

    print("--Worker Belief Test--")
    with open(f"{folder}/{run_folder}/mvpt_values.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/median_values.pkl","rb") as pkl_fl2:
        mvpt_median = float(list(pickle.load(pkl_fl1))[-1])
        median = float(list(pickle.load(pkl_fl2))[-1])
    
    if mvpt_median == median:
        print("skipping case: mvpt aligned with true median")
        return
    else:
        print("investigating case: mvpt NOT aligned with true median")

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

def plot_wage_differences_per_setting(setting_info,y="Average Wage",title="Comparison of distribution of average wage per setting.", gap = False,at=False,exp="EXP2",conditions="e-r"):
    '''assume setting info is list of length 4 and each list gives info for each distribution seperately'''
    data_reorged = [[] for i in range(4)]
    for j,s in enumerate(setting_info):
        for i in range(8):
            for v in s[0][i]:# save all runs and all distributions to each setting
                data_reorged[j].append(v)
    # data_final = [d_i for d in data_reorged for d_i in d]
    plt.violinplot(data_reorged)
    # plt.xlabel("Settings")
    plt.xticks([y+1 for y in range(len(data_reorged))],
                  labels=["Neither", "Firm-verified", "Predicted", "Both"],rotation=45,fontsize=16)
    plt.ylabel(y,fontsize = 16)
    if gap:
        plt.ylim((-0.2,0.5))
    elif at:
        plt.ylim((-0.5,0.5))
    # plt.show()
    if gap:
        plt.savefig(f"Formatted Final Graphs/avg_wage_gap_per_setting_violin-{exp}-{conditions}.png")
    elif at:
        plt.savefig(f"Formatted Final Graphs/avg_at_gap_per_setting_violin-{exp}-{conditions}.png")
    else:
        plt.savefig(f"Formatted Final Graphs/avg_wage_per_setting_violin-{exp}-{conditions}.png")

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
        # pos = [j for i in range(0,12,3) for j in range(i,i+2)]
        pos = range(8)
        labels = ["Neither,\nType $H$", "Neither,\nType $L$","Firm-verified,\nType $H$", "Firm-verified,\nType $L$","Predicted,\nType $H$", "Predicted,\nType $L$","Both,\nType $H$", "Both,\nType $L$"]
        bottom = np.zeros(len(pos))
    else:
        pos = range(4)
        labels = ["Neither","Firm-verified","Predicted", "Both"]
        bottom = np.zeros(len(labels))

    offer_info = {"No Offer":[], "Rejected Offer":[],"Accepted Offer":[]} # each list is s1:R,S, ..., s4:R,S

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
            offer_info["No Offer"].append(sum(no_R)/len(no_R))
            offer_info["No Offer"].append(sum(no_S)/len(no_S))
            offer_info["Rejected Offer"].append((1-sum(no_R)/len(no_R)) * sum(r_R)/len(r_R))
            offer_info["Rejected Offer"].append((1-sum(no_S)/len(no_S)) * sum(r_S)/len(r_S))
            offer_info["Accepted Offer"].append((1-sum(no_R)/len(no_R)) * (1-sum(r_R)/len(r_R)))
            offer_info["Accepted Offer"].append((1-sum(no_S)/len(no_S)) * (1-sum(r_S)/len(r_S)))
        else:
            offer_info["No Offer"].append(sum(no)/len(no))
            offer_info["Rejected Offer"].append((1-sum(no)/len(no)) * sum(r)/len(r))
            offer_info["Accepted Offer"].append((1-sum(no)/len(no)) * (1-sum(r)/len(r)))
    
    colors = ["#56B4E9","#E69F00","#009E73"]
    hatch = ["","//","--"]
    for i,(o, o_count) in enumerate(offer_info.items()):
        if i >0:
            plt.bar(labels, o_count,label=o,bottom=bottom,color=colors[i],hatch=hatch[i])
        else:
            plt.bar(labels, o_count,label=o,bottom=bottom,color=colors[i])
        bottom += o_count
        plt.xticks(pos,labels,fontsize=16,rotation=45)
        # plt.bar_label(p,fontsize=12,rotation=45)
    if type_conditioning:
        plt.ylabel("Proportion of $T$\nAveraged across all worker types",fontsize=16)
    else:
        plt.ylabel("Proportion of $T$\nAveraged across all workers",fontsize=16)
    plt.legend(loc="lower left")
    plt.show()

def plot_benchmark_choice_comparison(setting_info, stat="Largest"):
    
    pos = range(4)
    labels = [f"Neither",f"Firm-verified",f"Predicted",f"Both"]
    bottom = np.zeros(len(labels))

    bench_info = {"Identical Ranges":[], f"{stat}":[],f"Non-{stat}":[]} # each list is s1:R,S, ..., s4:R,S

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
        bench_info["Identical Ranges"].append(sum(iden)/len(iden))
        if stat == "Largest":
            bench_info["Largest"].append((1-sum(iden)/len(iden)) * sum(large)/len(large))
            bench_info["Non-Largest"].append((1-sum(iden)/len(iden)) * (1-sum(large)/len(large)))
        else:
            bench_info["Widest"].append((1-sum(iden)/len(iden)) * sum(wide)/len(wide))
            bench_info["Non-Widest"].append((1-sum(iden)/len(iden)) * (1-sum(wide)/len(wide)))
    
    # print(bench_info)
    colors = ["#CC79A7","#0072B2","#F0E442"]
    hatch = ["","**","||"]
    for i,(b, b_count) in enumerate(bench_info.items()):
        if i > 0:
            plt.bar(labels, b_count,label=b,bottom=bottom,color=colors[i],hatch=hatch[i])
        else:
            plt.bar(labels, b_count,label=b,bottom=bottom,color=colors[i])
        plt.xticks(pos,labels,fontsize=16,rotation=45)
        # plt.bar_label(p,fontsize=12,rotation=45)
        bottom += b_count
    # plt.title("Comparison of Firm Benchmark Choices")
    plt.ylabel("Proportion of $T$\nAveraged across all firms",fontsize=16)
    plt.legend(loc="lower left")
    plt.show()


def setting_comparison_statistics(data_reorged):
    '''data_reorged has same 4xnum samples (120 in our case) so that empirical distributions of observations can be compared.'''
    
    for j in range(4):
        print(f"Describing wage gap Setting {j+1}")
        print(stats.describe(data_reorged[j]))

    
    print("Mann-Whitney U two sample test")
    for i in range(4):
        if i == 1:
            continue
        print(f"Comparison of Setting {2} with Setting {i+1}")
        print(stats.mannwhitneyu(data_reorged[i],data_reorged[1],alternative="less"))
        print()
    
    for i in range(4):
        if i == 3:
            continue
        print(f"Comparison of Setting {4} with Setting {i+1}")
        print(stats.mannwhitneyu(data_reorged[i],data_reorged[3],alternative="less"))
        print()  


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

# riskiness = [(0.5,0.5)]#(0.75,0.5),(0.25,0.5), (0.5,0.25),,(0.25,0.25),(0.75,0.5),(0.75,0.25),(0.5,0.5)
riskiness_label = ["s-r"]#"r-r","s-r","e-s", "s-s","r-s","e-r"
exp = "EXP2"
folder = "simulation_output_data_experiment_2_asym_betas_risky_market"

p_labels = ["k-r","s-k-r","s-k-l","k-l","u", "b-e", "b-l","b-r"] 

# fixed across settings 
N_w = 100 # small number of workers to start
N_f = 5 # small number of firms
k = 5 # number of intervals to break [0,1] up into
W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
ranges = W + [-1] # -1 indicates no range given 
alpha = 0.3 # more weight on present rewards
delta = 0.9 # more patient
p_s = [1/(k+1) for i in range(k+1)] # this parameter doesn't matter, but just for initializing market
type_conditioning = True
T = 20000


# change these indices to change settings
p_l = "u"
N = 15
b_label = beta_labels[0]
setting_data_1 = [[] for i in range(len(setting_label))] # W_F
setting_data_2 = [[] for i in range(len(setting_label))] # AT gaps
setting_data_3 = [[] for i in range(len(setting_label))] # benchmark choice
setting_data_4 = [[] for i in range(len(setting_label))] # W_G
setting_data_5 = [[] for i in range(len(setting_label))] # Offer choices
for r,r_label in enumerate(riskiness_label):
    for s,s_label in enumerate(setting_label):
        print(f"evaluating {s_label}...")
        distribution_data_1 = [[] for i in range(len(p_labels))]
        distribution_data_2 = [[] for i in range(len(p_labels))]
        distribution_data_3 = [[] for i in range(len(p_labels))]
        distribution_data_4 = [[] for i in range(len(p_labels))]
        distribution_data_5 = [[] for i in range(len(p_labels))]

        # set values
        use_mvpt,posting,mixed_posts  = settings[s]
        beta = betas[0]
        # p_risky, p_reset = riskiness[r]

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
            for n in range(N):

                run_folder= f"N_w={N_w}_N_f={N_f}_k={k}_{s_label}_initial_dist={p_l}_beta={b_label}_risk={r_label}_seed={seed}_N={n}"

                if use_mvpt:
                    with open(f"{folder}/{run_folder}/mvpt_values.pkl","rb") as pkl_fl1, open(f"{folder}/{run_folder}/median_values.pkl","rb") as pkl_fl2:
                        mvpt_values = list(pickle.load(pkl_fl1))
                        medians = list(pickle.load(pkl_fl2))
                
                firm_info, worker_info, sal_bench_l, sal_bench_u = load_firm_worker_info(folder, run_folder, use_mvpt,type_conditioning)
                all_wages  = get_worker_wages(worker_info,N_w)
                if type_conditioning:
                    wages_R,wages_S = get_worker_wages_typed(worker_info,N_w)

                
                firms_ATs = [firm_info[i]["AT"] for i in range(N_f)]

                stable_t = len(firms_ATs[0])

                if type_conditioning:
                    risky_ATs = [[at[0] for at in firms_ATs[i]]for i in range(N_f)]
                    safe_ATs = [[at[1] for at in firms_ATs[i]]for i in range(N_f)]
                    for t in range(T):
                        stable_r = 1
                        stable_s = 1
                        for i in range(N_f):
                            if min(risky_ATs[i][t:]) != max(risky_ATs[i][t:]):
                                stable_r = 0
                            if min(safe_ATs[i][t:]) != max(safe_ATs[i][t:]):
                                stable_s = 0
                        if stable_r and stable_s:
                            stable_t = t
                            break
                else:
                    for t in range(T):
                        stable = 1
                        for i in range(N_f):
                            if min(firms_ATs[i][t:]) != max(firms_ATs[i][t:]):
                                stable = 0
                        if stable:
                            stable_t = t
                            break
                print(f"AT stability at {stable_t}/{T}")


                continue



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

                if type_conditioning:
                    all_p_R.append(sum(rejected_R)/len(rejected_R))
                    all_p_S.append(sum(rejected_S)/len(rejected_S))
                    all_p_no_R.append(sum(no_off_R)/len(no_off_R))
                    all_p_no_S.append(sum(no_off_S)/len(no_off_S))
                    all_p_od_R.append(sum(off_diff_R)/len(off_diff_R))
                    all_p_od_S.append(sum(off_diff_S)/len(off_diff_S))
                else:
                    all_p_rej.append(sum(props_rejected)/len(props_rejected))
                    all_p_no.append(sum(props_no_offer)/len(props_no_offer))
                    all_p_od.append(sum(avg_off_diff)/len(avg_off_diff))
                all_p_r_l.append(sum(r_l)/len(r_l))
                all_p_u_b.append(sum(u_b)/len(u_b))
                all_p_l_b.append(sum(l_b)/len(l_b))
                all_p_u_l_u.append(sum(u_l_u)/len(u_l_u))
                all_p_u_l_r.append(sum(u_l_r)/len(u_l_r))
                all_p_p_ident.append(sum(p_ident)/len(p_ident))


               
                distribution_data_1[p].append((1/N_w)*sum(all_wages)) # W_F
                if type_conditioning:
                    distribution_data_2[p].append(sum(AT_gaps)/len(AT_gaps)) # AT gaps
                    distribution_data_4[p].append(sum(wages_R)/len(wages_R) - sum(wages_S)/len(wages_S))# W_G
                distribution_data_3[p].append({"identical ranges":sum(p_ident)/len(p_ident),"largest":sum(u_l_u)/len(u_l_u),"widest":sum(u_l_r)/len(u_l_r)}) # benchmark choices
                if type_conditioning:
                    p_no_R = sum(no_off_R)/len(no_off_R)
                    p_no_S = sum(no_off_S)/len(no_off_S)
                    p_r_R = sum(rejected_R)/len(rejected_R)
                    p_r_S = sum(rejected_S)/len(rejected_S)
                    distribution_data_5[p].append({"no-offer-R":p_no_R,"no-offer-S":p_no_S,"rejected-R":p_r_R,"rejected-S":p_r_S}) # offer choices, typed
                else:
                    p_no = sum(props_no_offer)/len(props_no_offer)
                    p_r = sum(props_rejected)/len(props_rejected)
                    distribution_data_5[p].append({"no-offer":p_no,"rejected":p_r,})# offer choices

        setting_data_1[s].append(distribution_data_1)
        if type_conditioning:
            setting_data_2[s].append(distribution_data_2)
            setting_data_4[s].append(distribution_data_4)
        setting_data_3[s].append(distribution_data_3)
        setting_data_5[s].append(distribution_data_5)
        # all_at_gaps = [at_gap for all_gaps in distribution_data_2 for at_gap in all_gaps]
        # across_all_distributions_surplus.append((1/N_w)*sum(all_wages))
        # across_all_distributions_gap.append((1/idx_R)*sum(wages_R) - (1/idx_S)*sum(wages_S))
                


data_reorged = plot_wage_differences_per_setting(setting_data_1,y=r"$W_F$",exp=exp,conditions=riskiness_label[0]) # W_F
if type_conditioning:
    data_reorged = plot_wage_differences_per_setting(setting_data_2,y=r"$AT_G$",at=True,exp=exp,conditions=riskiness_label[0]) # AT gaps
    data_reorged = plot_wage_differences_per_setting(setting_data_4,y=r"$W_G$",gap=True,exp=exp,conditions=riskiness_label[0]) # W_G
plot_benchmark_choice_comparison(setting_data_3,stat="Widest") # benchmark choice 
plot_offer_differences(setting_data_5,type_conditioning) # offer choice 

# setting_comparison_statistics(data_reorged)













 


