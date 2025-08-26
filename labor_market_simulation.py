import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import math

from labor_market_classes import Market
from market_outcome_analysis import get_wage_distribution_within_firm, get_wage_distribution_market, plot_attribute_distribution_market, plot_attribute_distribution_within_firm, print_wage_distribution_within_firm, get_wage_distribution_market_split_workers

seed = 1 # use for reproducibility of initial simulations to debug 


def check_convergence(market, failure_threshold, acceptance_threshold, reject_threshold=0.51, accept_threshold=0.5, negotiations_threshold=50):

    num_stop_sharing = 0
    num_acceptance = 0

    for w in market.workers:
        num_stop_sharing = num_stop_sharing + w.stop_sharing
        num_acceptance = num_acceptance + (w.always_share)

    num_negotiations = [s+f for s,f in zip(market.num_successful_mvpt_negotiations, market.num_failed_mvpt_negotiations)]
    
    if num_stop_sharing >= int(reject_threshold*len(market.workers)):
        print(f"At least {reject_threshold} proportion of workers have less than {failure_threshold*100}% confidence in mvpt, tool too unreliable.")
        return 2
    # if num_acceptance >= int(accept_threshold*len(market.workers)):
    #     print(f"At least {accept_threshold} proportion of workers have at least {acceptance_threshold*100}% confidence in mvpt, tool is credible.")
    #     return 1

    if len(num_negotiations) >= negotiations_threshold and all([s == 0 for s in num_negotiations[-negotiations_threshold:]]):
        print(f"No negotiations for {negotiations_threshold} time steps.")
        return 3
    
    return -1



if __name__ == "__main__":

    # labor market attributes
    T = 2000 # total number of time steps in one simulation run
    benchmark_proportion =  1 # proportion of firms that get added to random sample, 1=> firms perfectly know wage distributions in market, shared beliefs
    J = 1000 # max number of job switches the workers can do 
    lam_high = 200 / T # rate of job offers per simulation run
    lam_low = 150 / T # rate of job offers per simulation run 
    # print(f"p(k>= 2| lam_H) * 2000 = {(1-np.exp(-1 *lam_high) * sum([lam_high**i  / math.factorial(i) for i in range(2)]))*2000}")
    # print(f"p(k>= 2| lam_L) * 2000 = {(1-np.exp(-1 *lam_low) * sum([lam_low**i  / math.factorial(i) for i in range(2)]))*2000}")
    outside_offer_cutoff = [0.9] # cutoff where workers switch from L type to H type 
    lambdas = [(300 / T, 1/ T)]#, (250/T, 5/T), (200/T,2/T)]


    # firm attributes
    N_firms = 10
    firm_capacity = 20 # total number of employees a firm can have
    counter_t = [0]*5 + [1]*5 # 0 = conservative and 1 = competitive for default counter type of firms if lam_high = lam_low = 0

    # worker attribtues 
    N_workers = 100 # evenly split among firms initially
    search_threshold = 0 # threshold for workers to renegotiate with their current firm or seek out new firm, i.e., if they could make more than (1+s) times their curent pay
    failure_threshold = [0.05] # tolerance of workers to stop using mvpt
    acceptance_threshold = 0.9 # tolerance of workers to always use mvpt

    # mvpt attributes
    sd_cap = 0.1 # standard deviation cap 
    initial_pool_proportion = [0.75]#,0.25]
    initial_pool_max_wage_cap = [0.4,0.6,0.7,0.8]
    
    for i_p_p in initial_pool_proportion:
        for i_p_w_c in initial_pool_max_wage_cap:
            for o_o_c in outside_offer_cutoff:
                for f_t in failure_threshold:
                    for l_H, l_L in lambdas:
                        print(f"p(k>= 2| lam_H) * 2000 = {(1-np.exp(-1 *l_H) * sum([l_H**i  / math.factorial(i) for i in range(2)]))*2000}")
                        print(f"p(k>= 2| lam_L) * 2000 = {(1-np.exp(-1 *l_L) * sum([l_L**i  / math.factorial(i) for i in range(2)]))*2000}")
                        N = 1 # number of simualations to run with the above parameters
                        save = True # save plots?

                        # run simulation
                        final_accept_m_hats = []
                        for n in range(N):

                            # if n == 5:
                            #     save = True # just save 1 run 
                            # else:
                            #     save = False

                            market = Market(N_f = N_firms, N_w=N_workers,counter_t=counter_t,C=firm_capacity, f=f_t, a=acceptance_threshold, s=search_threshold,b_k = benchmark_proportion,sd_cap=sd_cap,i_p=i_p_p,o_o_c=o_o_c, lam_H=l_H, lam_L=l_L, J=J,i_p_w_c=i_p_w_c)

                            # data points to track during market evolution
                            initial_counts_bins = [get_wage_distribution_within_firm(firm) for firm in market.firms]
                            initial_counts_bins_market = get_wage_distribution_market(market)
                            initial_wages = [w.wage for w in market.workers]

                            c_b_H = []
                            c_b_L = []

                            counts_bins_H, counts_bins_L = get_wage_distribution_market_split_workers(market,l_H)
                            c_b_H.append(counts_bins_H)
                            c_b_L.append(counts_bins_L)
                            

                            m_hat_over_time = [market.mvpt.m_hat]
                            l_hat_over_time = [market.mvpt.l_hat]
                            u_hat_over_time = [market.mvpt.u_hat]
                            mvpt_pool_size = [len(market.mvpt.data_pool)]
                            worker_mvpt_confidence = [[] for w in range(N_workers)]
                            # prop_H = [sum([1 for w in market.workers if w.outside_opp_rate == lam_high])/N_workers]

                            for t in tqdm(range(T)):
                                
                                # run a market time step 
                                market.market_time_step(T, t)

                                # track data points
                                counts_bins_H, counts_bins_L = get_wage_distribution_market_split_workers(market,l_H)
                                c_b_H.append(counts_bins_H)
                                c_b_L.append(counts_bins_L)

                                # m_hat_over_time.append(market.mvpt.m_hat)
                                # l_hat_over_time.append(market.mvpt.l_hat)
                                # u_hat_over_time.append(market.mvpt.u_hat)
                                # mvpt_pool_size.append(len(market.mvpt.data_pool)) 
                                # prop_H.append(sum([1 for w in market.workers if w.outside_opp_rate == lam_high])/N_workers)
                                # for i in range(N_workers):
                                #     worker_mvpt_confidence[i].append(market.workers[i].mvpt_confidence)
                                
                                # # check for convergence 
                                # conv = check_convergence(market,failure_threshold, acceptance_threshold, 0.51,1, 5000)
                                # if conv > 0:
                                #     if conv == 1 or conv == 3:
                                #         final_accept_m_hats.append(market.mvpt.m_hat)
                                #     break
                            
                            # Analyze Results 

                            # plt.bar(range(len(market.num_successful_mvpt_negotiations)), market.num_successful_mvpt_negotiations, color= "blue", label="Num. successful negotiations / time step")
                            # plt.bar(range(len(market.num_failed_mvpt_negotiations)),[-1*f for f in market.num_failed_mvpt_negotiations], color ="red", label = "Num. failed negotiations / time step")
                            # plt.xlabel("Time")
                            # plt.ylabel("Count of negotiation type (+:success, -:failed)")
                            # plt.title("Successful vs. Failed negotiations with MVPT over time")
                            # if save:
                            #     plt.savefig(f"simulation_results/setting_2/seed={seed}_market_wages_animations_job_switches_{J}/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}_mvpt_negotiations.png")
                            # plt.clf()

                            # plt.bar(range(len(market.num_successful_outside_negotiations)), market.num_successful_outside_negotiations, color= "blue", label="Num. successful negotiations / time step")
                            # plt.bar(range(len(market.num_failed_outside_negotiations)),[-1*f for f in market.num_failed_outside_negotiations], color ="red", label = "Num. failed negotiations / time step")
                            # plt.xlabel("Time")
                            # plt.ylabel("Count of negotiation type (+:success, -:failed)")
                            # plt.title("Successful vs. Failed negotiations with outside options over time")
                            # if save:
                            #     plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/successful_vs_failed_outside_negotiations")
                            # plt.clf()

                            # plt.bar(range(len(market.num_better_to_wait_H)), market.num_better_to_wait_H, color= "blue", label="Num. type H workers waiting / time step")
                            # plt.bar(range(len(market.num_better_to_wait_L)),[-1*f for f in market.num_better_to_wait_L], color ="red", label = "Num. type L workers waiting / time step")
                            # plt.xlabel("Time")
                            # plt.ylabel("Count of negotiation type (+:success, -:failed)")
                            # plt.title("Number of type H workers vs. type L workers that choose to wait over time")
                            # # if save:
                            # #     plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/type_H_vs_type_L_waiting")
                            # plt.show()

                        
                            # for k in range(0,100,5):
                            #     for i in range(k,k+5):
                            #         plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].job_switches},{market.workers[i].wage}")
                            #     plt.legend(title="Worker wage delta")
                            #     plt.xlabel("Time")
                            #     plt.ylabel("Confidence in MVPT")
                            #     plt.title(f"Worker confidence in MVPT over time, worker indices {k} to {k+4}")
                            #     plt.ylim((0,1))
                            #     if save:
                            #         plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/mvpt_confidence_worker_group_k={k}")
                            #     plt.clf()

                            # plt.plot(prop_H)
                            # plt.title("Proportion of high value outside option workers over time")
                            # plt.xlabel("Time")
                            # plt.ylabel(f"Proportion of workers with lambda={lam_high}")
                            # plt.ylim((0,1))
                            # plt.clf()

                            # plt.plot(m_hat_over_time, label="m_hat value (median + error)")
                            # plt.plot(l_hat_over_time, color="red",label="Min value in data pool")
                            # plt.plot(u_hat_over_time, color="purple",label="Max value in data pool")
                            # plt.ylim((0,1))
                            # plt.title("MVPT summary statistics values over time")
                            # plt.xlabel("Time")
                            # plt.ylabel("MVPT summary statistics values")
                            # plt.legend()
                            # if save:
                            #     plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/mvpt_values_over_time")
                            # plt.clf()
                            # plt.plot(mvpt_pool_size)
                            # plt.ylim((0,N_workers))
                            # plt.title("data pool size of mvpt")
                            # if save:
                            #     plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/mvpt_pool_size")
                            # plt.clf()
                            # print(f"Final m_hat value: {market.mvpt.m_hat}")
                            # print(f"Lower bound: {market.mvpt.l_hat}, Upper bound: {market.mvpt.u_hat}")

                            # plot total wage distribution
                            plot_attribute_distribution_market(market,"wage",N_workers,initial_counts_bins_market[0],initial_counts_bins_market[1],i_p_p,seed,True,n,i_p_w_c,o_o_c,f_t,l_H,l_L,J,c_b_H,c_b_L)

                            # for i,f in enumerate(market.firms):
                            #     # print(f"Firm id: {f}")
                            #     # print(f"number of workers: {len(f.workers)}")
                            #     # print(f"market pay beliefs: {f.market_pay_belief}")
                            #     plot_attribute_distribution_within_firm(i,f,"wage",firm_capacity,initial_counts_bins[i][0],initial_counts_bins[i][1],i_p_p,seed,save,n,i_p_w_c)
                            
                        # print out statistics of converged wage
                        # if len(final_accept_m_hats)>0:
                        #     print(f"average value of accepted m_hat: {np.mean(final_accept_m_hats)}")
                        #     print(f"max value of accepted m_hat: {np.max(final_accept_m_hats)}")
                        #     print(f"min value of accepted m_hat: {np.min(final_accept_m_hats)}")
                            
                                
                            