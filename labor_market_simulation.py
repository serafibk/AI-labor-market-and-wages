import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

from labor_market_classes import Market
from market_outcome_analysis import get_wage_distribution_within_firm, get_wage_distribution_market, plot_attribute_distribution_market, plot_attribute_distribution_within_firm, print_wage_distribution_within_firm

seed = 63 # use for reproducibility of initial simulations to debug 


def check_convergence(market, failure_threshold, acceptance_threshold, reject_threshold=0.51, accept_threshold=0.5, negotiations_threshold=50):

    num_stop_sharing = 0
    num_acceptance = 0

    for w in market.workers:
        num_stop_sharing = num_stop_sharing + w.stop_sharing
        num_acceptance = num_acceptance + (w.always_share)

    num_negotiations = [s+f for s,f in zip(market.num_successful_negotiations, market.num_failed_negotiations)]
    
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
    T = 5000 # total number of time steps in one simulation run
    benchmark_proportion =  1 # proportion of firms that get added to random sample, 1=> firms perfectly know wage distributions in market, shared beliefs
    J = 10 # max number of job switches the workers can do 
    lam_high = 200 / T # rate of job offers per simulation run
    lam_low = 5 / T # rate of job offers per simulation run 
    outside_offer_cutoff = 0.7 # cutoff where workers switch from L type to H type 
    p_mrpl = 0.01 
    
    # firm attributes
    N_firms = 100
    firm_capacity = 20 # total number of employees a firm can have
    counter_t = [0]*N_firms # 0 = conservative and 1 = competitive for default counter type of firms if lam_high = lam_low = 0

    # worker attribtues 
    N_workers = 1000 # evenly split among firms initially
    search_threshold = 0 # threshold for workers to renegotiate with their current firm or seek out new firm, i.e., if they could make more than (1+s) times their curent pay
    failure_threshold = 0.05 # tolerance of workers to stop using mvpt
    acceptance_threshold = 0.90 # tolerance of workers to always use mvpt

    # mvpt attributes
    sd_cap = 0.1 # standard deviation cap 
    initial_pool_proportion = 0.5
    

    N = 1 # number of simualations to run with the above parameters
    save = False # save plots?


    # run simulation
    final_accept_m_hats = []
    for n in range(N):

        if n == -1:
            save = True # just save 1 run 
        else:
            save = False

        market = Market(N_f = N_firms, N_w=N_workers,counter_t=counter_t,C=firm_capacity, f=failure_threshold, a=acceptance_threshold, s=search_threshold,b_k = benchmark_proportion,sd_cap=sd_cap,i_p=initial_pool_proportion,o_o_c=outside_offer_cutoff, lam_H=lam_high, lam_L=lam_low, J=J, p_mrpl=p_mrpl)

        # data points to track during market evolution
        initial_counts_bins = [get_wage_distribution_within_firm(firm) for firm in market.firms]
        initial_counts_bins_market = get_wage_distribution_market(market)
        initial_wages = [w.wage for w in market.workers]
        m_hat_over_time = [market.mvpt.m_hat]
        l_hat_over_time = [market.mvpt.l_hat]
        u_hat_over_time = [market.mvpt.u_hat]
        mvpt_pool_size = [len(market.mvpt.data_pool)]
        worker_mvpt_confidence = [[] for w in range(N_workers)]
        prop_H = [sum([1 for w in market.workers if w.outside_opp_rate == lam_high])/N_workers]

        for t in tqdm(range(T)):
            
            # run a market time step 
            market.market_time_step(T, t)

            # track data points
            m_hat_over_time.append(market.mvpt.m_hat)
            l_hat_over_time.append(market.mvpt.l_hat)
            u_hat_over_time.append(market.mvpt.u_hat)
            mvpt_pool_size.append(len(market.mvpt.data_pool)) 
            prop_H.append(sum([1 for w in market.workers if w.outside_opp_rate == lam_high])/N_workers)
            for i in range(N_workers):
                worker_mvpt_confidence[i].append(market.workers[i].mvpt_confidence)
            
            # check for convergence 
            conv = check_convergence(market,failure_threshold, acceptance_threshold, 0.51,1, 5000)
            if conv > 0:
                if conv == 1 or conv == 3:
                    final_accept_m_hats.append(market.mvpt.m_hat)
                break
        
        # Analyze Results 

        # plt.bar(range(len(market.num_successful_negotiations)), market.num_successful_negotiations, color= "blue", label="Num. successful negotiations / time step")
        # plt.bar(range(len(market.num_failed_negotiations)),[-1*f for f in market.num_failed_negotiations], color ="red", label = "Num. failed negotiations / time step")
        # plt.xlabel("Time")
        # plt.ylabel("Count of negotiation type (+:success, -:failed)")
        # if save:
        #     plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/successful_vs_failed_negotiations")
        # plt.show()
    
        # for k in range(0,100,5):
        #     for i in range(k,k+5):
        #         plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].wage - initial_wages[i]}")
        #     plt.legend(title="Worker wage delta")
        #     plt.xlabel("Time")
        #     plt.ylabel("Confidence in MVPT")
        #     plt.title(f"Worker confidence in MVPT over time, worker indices {k} to {k+4}")
        #     plt.ylim((0,1))
        #     if save:
        #         plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/mvpt_confidence_worker_group_k={k}")
        #     plt.show()

        plt.plot(prop_H)
        plt.title("Proportion of high value outside option workers over time")
        plt.xlabel("Time")
        plt.ylabel(f"Proportion of workers with lambda={lam_high}")
        plt.ylim((0,1))
        plt.show()

        plt.plot(m_hat_over_time, label="m_hat value (median + error)")
        plt.plot(l_hat_over_time, color="red",label="Min value in data pool")
        plt.plot(u_hat_over_time, color="purple",label="Max value in data pool")
        plt.ylim((0,1))
        plt.title("MVPT summary statistics values over time")
        plt.xlabel("Time")
        plt.ylabel("MVPT summary statistics values")
        plt.legend()
        if save:
            plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/mvpt_values_over_time")
        plt.show()
        plt.plot(mvpt_pool_size)
        plt.ylim((0,N_workers))
        plt.title("data pool size of mvpt")
        if save:
            plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/mvpt_pool_size")
        plt.show()
        print(f"Final m_hat value: {market.mvpt.m_hat}")
        print(f"Lower bound: {market.mvpt.l_hat}, Upper bound: {market.mvpt.u_hat}")

    #     # plot total wage distribution
        plot_attribute_distribution_market(market,"wage",N_workers,initial_counts_bins_market[0],initial_counts_bins_market[1],initial_pool_proportion,seed,save,n)

        # for i,f in enumerate(market.firms):
        #     # print(f"Firm id: {f}")
        #     # print(f"number of workers: {len(f.workers)}")
        #     # print(f"market pay beliefs: {f.market_pay_belief}")
        #     plot_attribute_distribution_within_firm(i,f,"wage",firm_capacity,initial_counts_bins[i][0],initial_counts_bins[i][1],initial_pool_proportion,seed,save,n)
        
    # # print out statistics of converged wage
    # if len(final_accept_m_hats)>0:
    #     print(f"average value of accepted m_hat: {np.mean(final_accept_m_hats)}")
    #     print(f"max value of accepted m_hat: {np.max(final_accept_m_hats)}")
    #     print(f"min value of accepted m_hat: {np.min(final_accept_m_hats)}")
        
            
        