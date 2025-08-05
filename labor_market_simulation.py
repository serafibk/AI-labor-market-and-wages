import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

seed = 22#707202508312025 # use for reproducibility of initial simulations to debug 
gen = np.random.default_rng(seed=seed)

# agents and market running functions (which can be moved to a separate file)

class Worker:

    def __init__(self,s,f,a):
        '''initialization of Worker agent 

        params 
            s (float): search threshold for worker such that they will renegotiate their pay with own firm or another firm if their median pay belief is >(1+s)*current wage. s>=0.
            f (float): confidence level, in range [0,1], in mvpt at which worker stops sharing data for all future time steps
            a (float): confidence level, in range [0,1], in mvpt at and above which worker shares data and uses mvpt tool w.p. 1
            firm (Firm): instance of Firm that is Worker's employer.
        '''
        # worker characteristics, search threshold and failure threshold common to all workers. 
        self.fairness_uncertainty = gen.beta(a=4,b=5) # fairness uncertainty salience, slightly left skewed distribution
        self.search_threshold = s # just do one threshold, nothing special about negotiating with your own firm vs. another
        self.failure_threshold = f # discard the tool
        self.acceptance_threshold = a # always use the tool
        
        # worker employment attributes
        self.employer = None # initially unemployed
        self.wage = None # initially unemployed

        # counters for tool confidence
        self.sought_info = 0 # track whether median belief set via mvpt or not
        self.num_info_seeking = 0 # for debugging, tracking that workers do in fact track whether interacting with the tool was a success or not
        self.num_failures = 0 # counter to track failed negotiations
        self.num_successes = 0 # counter to track successful negotiations
        self.alpha = 1
        self.beta = 1
        self.mvpt_confidence = gen.beta(a=self.alpha,b=self.beta) # how much mvpt tool is weighted compared to current wage, start with a beta distribution and then posterior updating based on number of successes/failurs. 1,1 to start since failure and success are both possible.
       
        # flags used in updates
        self.in_initial_pool = 0
        self.always_share = 0 # track if they should non-probabilistically share data (unless they lose confidence in it)
        self.stop_sharing = 0 # track if they ever stop sharing
        self.negotiating_with = None # firm that worker is trying to negotiate with

        # worker market pay belief
        self.market_pay_belief = [0,0,0]

    def information_seeking(self,mvpt):
        '''
        '''
        self.sought_info = 0 # no info sought to start
        # do not share if completely pessimistic about mvpt
        if self.stop_sharing == 1:
            mvpt.remove_worker(self) # make sure worker is not in next data pool once they are completely pessimistic about the tool
            return # do not seek info

        # if in the initial pool or confident about the tool, share w.p. 1
        if self.always_share == 1 or self.in_initial_pool == 1: 
            self.sought_info = 1
            self.num_info_seeking = self.num_info_seeking + 1
            mvpt.add_worker(self)
            self.belief_update(m_hat = mvpt.m_hat)
            return # done

        # everyone else gets to check upper bounds for perceived usefulness (lower than lower bound means you still want to check)
        if self.wage > mvpt.u_hat: # if you don't see any possible improvement, don't use the tool
            self.num_failures = self.num_failures + 1
            mvpt.remove_worker(self)
            return # do not seek more info
         
        # otherwise seek info w.p. < 1
        p_seek_info = self.fairness_uncertainty #* self.employer.restrictiveness_multiplier (bring back in once transparency is meaningful)
        seek_info = gen.choice([0,1],p=[1-p_seek_info, p_seek_info])

        if seek_info: # share data, belief update
            self.sought_info = 1
            self.num_info_seeking = self.num_info_seeking + 1
            mvpt.add_worker(self) # share data 
            self.belief_update(m_hat = mvpt.m_hat) # update belief based on mvpt information
        else: # don't share data, no belief update
            mvpt.remove_worker(self) 
        

    def belief_update(self,initial_anchor=0.01,m_hat = None, l_w = None, u_w = None):
        '''
        '''
        if self.wage is None:
            print("Error: No wage set.")
            return

        # confidence in mvpt updated first based on past successes / failures, using expected value of posterior distribution now 
        self.mvpt_confidence = (self.alpha+self.num_successes) / (self.alpha+self.beta + self.num_successes + self.num_failures)
       
        self.stop_sharing = (self.mvpt_confidence <= self.failure_threshold) # if and only if 
        self.always_share = (self.mvpt_confidence >= self.acceptance_threshold) # if and only if

        if m_hat is not None:
            # just update median pay
            self.market_pay_belief =  [min(m_hat,self.wage), self.mvpt_confidence*m_hat + (1-self.mvpt_confidence)*self.wage, max(m_hat,self.wage)]
        else:
            # center belief back at wage with anchoring
            self.market_pay_belief = [self.wage*(1-initial_anchor), self.wage, self.wage*(1+initial_anchor)] # if no info seeking, anchors wage upper and lower bound around current pay



class Firm:

    def __init__(self,d,C):
        '''initialization of Firm agent

        params
            d (float): initial wage dispersion parameter, i.e., standard deviation of wages around 0.5.
            C (int): maximum number of employees firm can support.
        '''
        self.dispersion = d
        self.capacity = C
        self.workers = set([]) # list of employed workers
        self.market_pay_belief = [0, 0, 0]
       
  
    def information_seeking(self, benchmark):
        '''

        notes
        - this could better align with current information seeking practices (e.g., x% look at independent manual benchmarks, y% look at glass door estimates of competitors, z% have access to a real time tool with higher accuracy than worker tool, etc.)
        - okay, going to assume cost-free benchmark not based on number of applicants 

        possible information seeking mechanisms
        -- real-time benchmarking
        -- competing offers from candidates (salary history, true outside offers, etc.)
        -- old benchmarking + individual process from tailoring 
        -- mvpt & user reported salaries
        -- posted salary ranges of other companies
        -- some other unknown, individualized process
        -- (speculative) AI to suggest which benchmarks to use or to predict gaps in salary data collection or to set the strategy for benchmarking
        '''
        self.belief_update(benchmark=benchmark) # update belief with benchmark, always


    def belief_update(self, benchmark=None):
        '''
        '''
        if benchmark is not None:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]+[benchmark[0]]), benchmark[1], np.max([w.wage for w in self.workers]+[benchmark[2]])] # update w.r.t. provided benchmark
        elif len(self.workers) > 0:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]), np.median([w.wage for w in self.workers]), np.max([w.wage for w in self.workers])] # update w.r.t. current wage distribution
        # otherwise no market pay beliefs and no workers! (special case where firm will accept any wage offer from a worker)

class MVPT:

    def __init__(self, worker_data_pool, N_w, sd_cap=0.1):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
            sigma_e (float): standard deviation parameter of error term of median calculation. positively correlated with len(worker_data_pool).
        '''
        # data / noise parameter for predictions
        self.data_pool = worker_data_pool # to generate median 
        self.SD_cap = sd_cap # cap on error for stability (?), i.e., third-party will not release the tool if it has a SD of > SD_cap around true median.
        self.sigma_e = (1-len(worker_data_pool)/N_w) * self.SD_cap # initial variance tied to how much of the data pool we have
        self.N_w = N_w # to track how much of the user base is currently in data pool

        # values of predictions
        self.error = 0
        self.m_hat = 0
        self.u_hat = 0 # upper bound of data pool (no error)
        self.l_hat = 0 # lower bound of data pool (no error)

    
    def add_worker(self, worker):
        '''adds specified Worker object to data pool. decreases sigma_e by a small amount. 

        params
            worker (Worker): Worker object to add. 
        '''
        if worker not in self.data_pool:
            self.data_pool.add(worker)
            self.sigma_e = (1-len(self.data_pool)/self.N_w) * self.SD_cap # variance tied to how much of the data pool we have

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e by a small amount.

        params
            worker (Worker): Worker object to remove. 
        '''
        if worker in self.data_pool:
            self.data_pool.remove(worker)
            self.sigma_e = (1-len(self.data_pool)/self.N_w) * self.SD_cap # variance tied to how much of the data pool we have

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        if len(self.data_pool) == 0:
            print("Error, data pool empty")
            return 

        error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated, simulates inherent unpredictability of ML tools (over traditional statisical methods with valid confidence intervals)
        self.error = abs(error) # record magnitude of error in either direction to report

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within reasonable bounds of data pool
        self.m_hat = np.median([w.wage for w in self.data_pool]) + error
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)

       

class Market:
    '''Market class is used to create a single market instance to keep track of the connections between firms and workers and MVPT. Additionally, carries out each phase of the market evolution.'''

    def __init__(self, N_f, N_w, D, C, f, a, s, b_k, s_e, i_p):
        '''initializaiton of the market
        '''
        self.benchmark_proportion = b_k

        self.firms = [Firm(d, C) for d in D] # firms identical up to initial wage dispersion
        self.workers = [Worker(s,f,a) for i in range(N_w)] # workers identical up to initial wage and fairness uncertainty

        # initial matching of firms and workers (which sets their initial beliefs)
        self._hire_initial_workers(np.split(np.asarray(self.workers), N_f))

        initial_worker_pool = set(gen.permutation(self.workers)[:int(N_w * i_p)]) # grab i_p proportion of initial workers randomly for MVPT,could correlate with pay distribution of current market
        for w in initial_worker_pool:
            w.in_initial_pool = 1 # mark that they always share data (unless they lose confidence from enough failed negotiations when sharing)
        self.mvpt = MVPT(initial_worker_pool,N_w,s_e)
        self.mvpt.update_mvpt()

        # self.num_negotiations = [] # track how many negotiations occur at each time step
        self.num_successful_negotiations = []
        self.num_failed_negotiations = []

        # track successful/failed info seeking outside of negotiations
        self.mvpt_reasonable = []
        self.mvpt_unreasonable = []
        
    
    def _hire_initial_workers(self, worker_assignment):
        '''
        '''
        for i,f in enumerate(self.firms):
            for w in worker_assignment[i]:
                # generate initial wage
                if f.dispersion == 1: # indicates draw from uniform distribution instead
                    wage = gen.uniform(0,1)
                else:
                    wage = gen.normal(0.5,f.dispersion)
                
                # enforce constraints:
                wage = max(wage,0)
                wage = min(wage,1)

                w.wage = wage
                # if w.wage >=0.5:
                #     w.alpha = 2
                #     w.beta = 1
                #     w.mvpt_confidence = gen.beta(w.alpha,w.beta) # optimistic in tool to increase wages

                # link worker and employee together
                w.employer = f
                f.workers.add(w)

                # update worker belief
                w.belief_update()
            
            # update firm belief based on initial worker pool
            f.belief_update()
    
    def market_time_step(self):
        '''
        '''
        # phase 1 - information seeking
        self.information_seeking()

        # phase 2 - negotiation and belief updates    
        self.negotiation_and_belief_update()

        # phase 3 - update mvpt
        self.mvpt.update_mvpt()
    
    
    def _perform_benchmark(self):
        ''''''
        # sample firms
        number_firms_to_check = int(self.benchmark_proportion * len(self.firms))
        sampled_firms = gen.choice(self.firms,number_firms_to_check,replace=False)


        # for f in self.firms:
        #     assert(f in sampled_firms)

        # compute summary statistics
        all_wages = []
        for f in sampled_firms:
            for w in f.workers:
                all_wages.append(w.wage)
        
        return [np.min(all_wages), np.median(all_wages), np.max(all_wages)]

    def information_seeking(self):
        '''
        '''
        # new_applicants = [[] for i in self.firms] # tracking which workers are trying to negotiate with new firms
        mvpt_success = 0
        mvpt_fail = 0

        # worker seeking
        for w in self.workers:
            w.information_seeking(self.mvpt)

            # no search cost to start
            if w.market_pay_belief[1] > (1+w.search_threshold)*w.wage: # requires m_hat to be  >  wage
                new_firm = gen.choice(self.firms,1)[0] # randomly choose a firm (possibly including current firm)
                # new_applicants[self.firms.index(new_firm)].append(w)
                w.negotiating_with = new_firm
            elif w.market_pay_belief[1] >= w.wage and w.sought_info: #< (1+w.search_threshold)*w.wage and w.market_pay_belief[1] >= w.wage and w.sought_info:
                assert(w.market_pay_belief[1] == w.wage)
                w.num_successes = w.num_successes + 1 # success in the sense of your confidence in the tool?
                w.negotiating_with = None
                mvpt_success = mvpt_success + 1
            elif w.market_pay_belief[1] < w.wage: #* (1-0.01): # testing tolerance 
                w.num_failures = w.num_failures + 1    
                w.negotiating_with = None
                mvpt_fail = mvpt_fail +1 
            else: # median belief not based on mvpt, w.sought_info == 0
                w.negotiating_with = None


        market.mvpt_reasonable.append(mvpt_success)
        market.mvpt_unreasonable.append(mvpt_fail)
        
        # firms always get benchmark
        for i,f in enumerate(self.firms):
            benchmark_for_f = self._perform_benchmark()
            f.information_seeking(benchmark_for_f)

    def negotiation_and_belief_update(self):
        '''
        '''
        def _bargaining_outcome(o1, at1, at2, o2, at1_c):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(a_t_1), counter offer cap (a_t_2), '''
            if o1 <= at1: # opening offer lower than firm's median belief, accept
                return o1
            elif (o1 <= at2) and (o2 >= at1_c): # opening offer lower than firm's upper belief, worker accepts reduced counteroffer if it is above their lower bound belief
                return o2 
            else: # opening offer too high or counter offer too low, reject
                return -1

        # num_negotiations = 0
        num_successful_negotiations = 0
        num_failed_negotiations = 0

        # for i,f in enumerate(market.firms):
        #     print(f"Firm id: {f}")
        #     print(f"number of workers: {len(f.workers)}")
        #     print(f"market pay beliefs: {f.market_pay_belief}")
        
        # print(f"mhat: {self.mvpt.m_hat}")

        for w in gen.permutation(self.workers): # randomize sequence of negotiations so that the same workers are not getting consistenly filtered out due to capacity
            
            firm = w.negotiating_with # firm that worker is potentially negotiating with

            if firm == None:
                continue # skip workers not renegotiating, success or failure if they used the tool already tracked

            # special cases
            if len(firm.workers)>= firm.capacity:
                num_failed_negotiations = num_failed_negotiations+ 1 # failure to negotiate, but still attempted
                continue # firm already at capacity, not really a success or failure of the tool 
            
            if len(firm.workers) == 0:
                w.wage = w.market_pay_belief[1] # accept any wage offer 
                w.employer.workers.remove(w) # remove from previous firm 
                w.employer = firm # worker has firm as employer
                firm.workers.add(w) # firm has worker as employee
                w.num_successes = w.num_successes + 1
                num_successful_negotiations = num_successful_negotiations+ 1
                continue # successful switch

            outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1],firm.market_pay_belief[2], w.market_pay_belief[1] - self.mvpt.error,w.market_pay_belief[0])
            
            if outcome == -1:
                w.num_failures = w.num_failures + 1
                num_failed_negotiations = num_failed_negotiations+ 1
            else:
                w.wage = outcome
                w.employer.workers.remove(w) # remove from previous firm 
                w.employer = firm # worker has firm as employer
                firm.workers.add(w) # firm has worker as employee
                w.num_successes = w.num_successes + 1
                num_successful_negotiations = num_successful_negotiations + 1
        
        # workers update beliefs after all negotiations done
        for w in self.workers:
            w.belief_update()

        # firms update beliefs after all negotiations done
        for f in self.firms:
            f.belief_update()
        
        # self.num_negotiations.append(num_negotiations) # tracking how many attempted negotiations there were, 0=> wage stability 
        self.num_successful_negotiations.append(num_successful_negotiations)
        self.num_failed_negotiations.append(num_failed_negotiations)

def get_wage_distribution_within_firm(firm):
    wages = []
    for w in firm.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=100,range=(0,1))

    return counts, bins

def get_wage_distribution_market(firm):
    wages = []
    for w in market.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=100,range=(0,1))

    return counts, bins


def plot_attribute_distribution_market(market,attr, c, extra_counts = None, extra_bins = None,initial_pool_proportion=1,seed=0,save=False,n=0):
    
    attribute_values = []
    
    for w in market.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,bins=100,range=(0,1))
    plt.stairs(counts,bins,label=f"Final {attr} distribution")
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed",label=f"Initial {attr} distribution")
    plt.title(f"Distribution of {attr} of Workers throughout market")
    plt.legend()
    plt.ylim((0,c))
    plt.xlabel(f"{attr} value, between 0 and 1")
    plt.ylabel(f"Density of {attr} value throughout market")
    if save:
        plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/distribution_of_{attr}_market")
    plt.show()


def plot_attribute_distribution_within_firm(f_idx,firm,attr, c, extra_counts = None, extra_bins = None,initial_pool_proportion=1,seed=0,save=False,n=0):
    
    attribute_values = []
    
    for w in firm.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,bins=100,range=(0,1))
    plt.stairs(counts,bins,label=f"Final {attr} distribution")
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed",label=f"Initial {attr} distribution")
    plt.title(f"Distribution of {attr} of Workers at Firm {f_idx}")
    plt.legend()
    plt.ylim((0,c))
    plt.xlabel(f"{attr} value, between 0 and 1")
    plt.ylabel(f"Density of {attr} value at Firm {f_idx}")
    if save:
        plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/distribution_of_{attr}_at_firm_{f_idx}")
    plt.show()

def print_wage_distribution_within_firm(firm):
    
    wages = []
    for w in firm.workers:
        wages.append(w.wage)
    
    print(f"Firm id: {firm}")
    print(f"min wage: {np.min(wages)}")
    print(f"med wage: {np.median(wages)}")
    print(f"max wage: {np.max(wages)}")
    print("--------------------------")


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
    if num_acceptance >= int(accept_threshold*len(market.workers)):
        print(f"At least {accept_threshold} proportion of workers have at least {acceptance_threshold*100}% confidence in mvpt, tool is credible.")
        return 1

    if len(num_negotiations) >= negotiations_threshold and all([s == 0 for s in num_negotiations[-negotiations_threshold:]]):
        print(num_acceptance)
        print(num_stop_sharing)
        print(len(market.workers))
        print(f"No negotiations for {negotiations_threshold} time steps.")
        return 3
    
    return -1




if __name__ == "__main__":

    # parameters not currently used
    # worker_productivity =  1 # "quality" of worker -- ASSUME wages in [0,1], assume wages always below productivity
    # restrictive_firm_multiplier = 0.7 # relatively increases the chance of workers in restrictive firms of sharing data


    # labor market parameters
    N_firms = 9
    N_workers = 90 # evenly split among firms initially
    firm_capacity = 20 # total number of employees a firm can have
    search_threshold = 0 # threshold for workers to renegotiate with their current firm or seek out new firm, i.e., if they could make more than (1+s) times their curent pay
    benchmark_proportion =  0.6 # proportion of firms that get added to random sample, 1=> firms perfectly know wage distributions in market, shared beliefs
    sigma_e = 0.1 # variance cap 
    initial_pool_proportion = 0.2
    dispersions = [0.01]*int(N_firms/3) + [0.4]*int(N_firms/3) + [1]*int(N_firms/3)
    

    # tolerance for accepting or rejecting credibility of tool
    failure_threshold = 0.05 # tolerance of workers to stop using mvpt
    acceptance_threshold = 0.75 # tolerance of workers to always use mvpt

    N = 10 # number of simualations to run with the above parameters
    save = False # save plots?


    # run simulation
    final_accept_m_hats = []
    for n in range(N):
        if n == 0 or n ==1:
            save = True
        else:
            save = False

        market = Market(N_f = N_firms, N_w=N_workers,D=dispersions,C=firm_capacity, f=failure_threshold, a=acceptance_threshold, s=search_threshold,b_k = benchmark_proportion,s_e=sigma_e,i_p=initial_pool_proportion)
    
        initial_counts_bins = [get_wage_distribution_within_firm(firm) for firm in market.firms]
        initial_counts_bins_market = get_wage_distribution_market(market)
        initial_wages = [w.wage for w in market.workers]
        m_hat_over_time = [market.mvpt.m_hat]
        l_hat_over_time = [market.mvpt.l_hat]
        u_hat_over_time = [market.mvpt.u_hat]
        mvpt_pool_size = [len(market.mvpt.data_pool)]

        worker_mvpt_confidence = [[] for w in range(N_workers)]
        for t in tqdm(range(30000)):
            market.market_time_step()
            m_hat_over_time.append(market.mvpt.m_hat)
            l_hat_over_time.append(market.mvpt.l_hat)
            u_hat_over_time.append(market.mvpt.u_hat)
            mvpt_pool_size.append(len(market.mvpt.data_pool)) 
            for i in range(N_workers):
                worker_mvpt_confidence[i].append(market.workers[i].mvpt_confidence)
            conv = check_convergence(market,failure_threshold, acceptance_threshold, 0.51,0.50, 20000)
            if conv > 0:
                if conv == 1:
                    final_accept_m_hats.append(market.mvpt.m_hat)
                break
        
        # analyze results 
        plt.bar(range(len(market.mvpt_reasonable)), market.mvpt_reasonable, color= "blue", label="Num. m_hat >= wage / time step")
        plt.bar(range(len(market.mvpt_unreasonable)),[-1*u for u in market.mvpt_unreasonable], color ="red", label = "Num. m_hat < wage / time step")
        plt.xlabel("Time")
        plt.ylabel("Count of m_hat vs. wage type (+:m_hat >= wage, -:m_hat <wage)")
        if save:
            plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/successful_vs_failed_data_sharing_no_negotiation")
        plt.show()

        plt.bar(range(len(market.num_successful_negotiations)), market.num_successful_negotiations, color= "blue", label="Num. successful negotiations / time step")
        plt.bar(range(len(market.num_failed_negotiations)),[-1*f for f in market.num_failed_negotiations], color ="red", label = "Num. failed negotiations / time step")
        plt.xlabel("Time")
        plt.ylabel("Count of negotiation type (+:success, -:failed)")
        if save:
            plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/successful_vs_failed_negotiations")
        plt.show()
    
        for k in range(0,90,5):
            for i in range(k,k+5):
                plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].wage - initial_wages[i]}")
            plt.legend(title="Worker wage delta")
            plt.xlabel("Time")
            plt.ylabel("Confidence in MVPT")
            plt.title(f"Worker confidence in MVPT over time, worker indices {k} to {k+4}")
            plt.ylim((0,1))
            if save:
                plt.savefig(f"simulation_results/seed={seed}_i_p_{initial_pool_proportion}_{n}/mvpt_confidence_worker_group_k={k}")
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
        print(f"Lower bound: {market.mvpt.l_hat}")

        # plot total wage distribution
        plot_attribute_distribution_market(market,"wage",N_workers,initial_counts_bins_market[0],initial_counts_bins_market[1],initial_pool_proportion,seed,save,n)

        for i,f in enumerate(market.firms):
            print(f"Firm id: {f}")
            print(f"number of workers: {len(f.workers)}")
            print(f"market pay beliefs: {f.market_pay_belief}")
            plot_attribute_distribution_within_firm(i,f,"wage",firm_capacity,initial_counts_bins[i][0],initial_counts_bins[i][1],initial_pool_proportion,seed,save,n)
        
    # print out statistics of converged wage
    if len(final_accept_m_hats)>0:
        print(f"average value of accepted m_hat: {np.mean(final_accept_m_hats)}")
        print(f"max value of accepted m_hat: {np.max(final_accept_m_hats)}")
        print(f"min value of accepted m_hat: {np.min(final_accept_m_hats)}")
        
            
        