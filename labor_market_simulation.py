import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

seed = 14#707202508312025 # use for reproducibility of initial simulations to debug 
gen = np.random.default_rng(seed=seed)

# agents and market running functions (which can be moved to a separate file)

class Worker:

    def __init__(self,q,s,T):
        '''initialization of Worker agent 

        params
            q (float):   
            s1 (float): 
            s2 (float): 
            T (int): max number of failed negotiations a worker is willing to tolerate from MVPT.
            firm (Firm): instance of Firm that is Worker's initial employer.
        '''
        # worker characteristics, search threshold and failure threshold common to all workers. 
        self.fairness_uncertainty = gen.beta(a=4,b=5) # fairness uncertainty salience, slightly left skewed distribution
        # self.productivity = q # not used right now since firm always hires if worker offer is "reasonable"
        self.search_threshold = s # just do one threshold, nothing special about negotiating with your own firm vs. another
        self.failure_threshold = T # could also be failure threshold for how often you see the tool is wrong (or just increase this value)
        
        # worker employment attributes
        self.employer = None # initially unemployed
        self.wage = None # initially unemployed

        # counters for tool confidence
        self.num_failures = 0 # counter to track failed negotiations
        self.num_successes = 0 # counter to track successful negotiations
        self.mvpt_confidence = gen.beta(a=1,b=1) # how much mvpt tool is weighted compared to current wage, start with a beta distribution and then posterior updating based on number of successes/failurs. 1,1 to start since failure and success are both possible.
       
        # flags used in updates
        self.in_initial_pool = 0 # track if they are always sharing data or not
        self.stop_sharing = 0 # track if they ever stop sharing
        self.negotiating_with = None # firm that worker is trying to negotiate with

        # worker market pay belief
        self.market_pay_belief = [0,0,0]

    def information_seeking(self,mvpt):
        '''
        '''

        # workers get to view some proportion of their colleagues salaries to update belief always
        # colleagues_salaries =  gen.permutation([w.wage for w in self.employer.workers])[:int(np.ceil(len(self.employer.workers) * self.employer.dispersion_multiplier))]
        # l_w = min(colleagues_salaries)
        # u_w = max(colleagues_salaries)
        # self.belief_update(l_w = l_w,u_w=u_w) # everyone updates upper and lower bound

        if self.stop_sharing == 1:
            mvpt.remove_worker(self) # make sure worker is not in next data pool once they are completely pessimistic about the tool (and not in initial pool?)
            return # do not seek info

        if self.in_initial_pool: # always share, UNLESS at stop_sharing threshold
            mvpt.add_worker(self)
            self.belief_update(m_hat = mvpt.m_hat)
            return # done

        # everyone else gets to check upper bounds for perceived usefulness (lower than lower bound means you still want to check)
        if self.wage >= mvpt.u_hat: # if you don't see any possible improvement, don't use the tool
            self.num_failures = self.num_failures + 1
            self.belief_update() # to track stop sharing?
            mvpt.remove_worker(self)
            return # do not seek more info
         
        # otherwise seek info stochastically
        p_seek_info = self.fairness_uncertainty * self.employer.dispersion_multiplier 
        seek_info = gen.choice([0,1],p=[1-p_seek_info, p_seek_info])

        if seek_info: # share data, belief update
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
        self.mvpt_confidence = (1+self.num_successes) / (2 + self.num_successes + self.num_failures)# gen.beta(1+self.num_successes, 1+self.num_failures) 
        self.stop_sharing = (self.mvpt_confidence <= 0.2) # stop sharing in the next round based on confidence, not total number of failures

        if m_hat is not None:
            # just update median pay
            self.market_pay_belief =  [min(m_hat,self.wage), self.mvpt_confidence*m_hat + (1-self.mvpt_confidence)*self.wage, max(m_hat,self.wage)]
        elif l_w is not None:
            # update range of pay based on colleagues
            self.market_pay_belief[0] = min(l_w,self.wage)
            self.market_pay_belief[2] = max(u_w,self.wage)
        else:
            # center belief back at wage with anchoring
            self.market_pay_belief= [self.wage*(1-initial_anchor), self.wage, self.wage*(1+initial_anchor)] # (only initially) anchors wage upper and lower bound around current pay



class Firm:

    def __init__(self,t,C):
        '''initialization of Firm agent

        params
            t (float): multiplier between 0 and 1 that increases or decreases the chance of employed workers to seek information from MVPT.
            C (int): maximum number of employees firm can support.
        '''
        self.dispersion_multiplier = t
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
        # m_hat = mvpt.m_hat
        # error = mvpt.error
        # if self.market_pay_belief[1] + self.benchmark_cost / N_a <= (m_hat-error): # benchmark cost is worth potential savings

        self.belief_update(benchmark=benchmark) # update belief with benchmark, always


    def belief_update(self, benchmark=None):
        '''
        '''
        if benchmark is not None:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]+[benchmark[0]]), benchmark[1], np.max([w.wage for w in self.workers]+[benchmark[2]])] # update w.r.t. provided benchmark
        elif len(self.workers) > 0:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]), np.median([w.wage for w in self.workers]), np.max([w.wage for w in self.workers])] # update w.r.t. current wage distribution


class MVPT:

    def __init__(self, worker_data_pool, sigma_e, sd_cap=0.1):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
            sigma_e (float): standard deviation parameter of error term of median calculation. positively correlated with len(worker_data_pool).
        '''
        # data / noise parameter for predictions
        self.data_pool = worker_data_pool # to generate median 
        self.sigma_e = sigma_e # to generate error
        self.SD_cap = sd_cap

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
            self.sigma_e = max(self.sigma_e - 1e-5,0) # arbitrary decrease / floor for now

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e by a small amount.

        params
            worker (Worker): Worker object to remove. 
        '''
        if worker in self.data_pool:
            self.data_pool.remove(worker)
            self.sigma_e = min(self.sigma_e + 1e-5,self.SD_cap) # arbitrary increase / cap for now (assume tool is only released if SD of error is <=0.1)

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated
        self.error = abs(error) # record magnitude of error in either direction to report

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within wage bounds of 0,1
        self.m_hat = np.median([w.wage for w in self.data_pool]) + error

        # keep it within reasonable bounds of data pool
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)

       

class Market:
    '''Market class is used to create a single market instance to keep track of the connections between firms and workers and MVPT. Additionally, carries out each phase of the market evolution.'''

    def __init__(self, N_f, N_w, C,q,T,s,t_f_m,b_k,s_e,i_p):
        '''initializaiton of the market

        params:

        '''
        self.benchmark_proportion = b_k

        self.firms = [Firm(t_f_m, C) for i in range(int(N_f / 2))] + [Firm(1-t_f_m,C) for i in range(int(N_f / 2))] # half firms transparent and half not
        self.workers = [Worker(q,s,T) for i in range(N_w)] # all identical workers to start

        # initial matching of firms and workers (which sets their initial beliefs)
        self._hire_initial_workers(t_f_m,np.split(np.asarray(self.workers), N_f))

        initial_worker_pool = set(gen.permutation(self.workers)[:int(N_w * i_p)]) # grab i_p proportion of initial workers randomly for MVPT,could correlate with pay distribution of current market
        # [w for w in self.workers if w.fairness_uncertainty >= 0.4] pull from workers with higher fairness uncertainty first
        for w in initial_worker_pool:
            w.in_initial_pool = 1 # mark that they always share data 
        self.mvpt = MVPT(initial_worker_pool,s_e)
        self.mvpt.update_mvpt()

        self.num_successes = []
        
    
    def _hire_initial_workers(self, t_f_m, worker_assignment):
        '''
        '''
        for i,f in enumerate(self.firms):
            for w in worker_assignment[i]:
                # generate initial wage
                sigma_f = 0.3
                if f.dispersion_multiplier == t_f_m:
                    sigma_f = 0.05          
                wage = gen.normal(0.5,sigma_f)
                
                # enforce constraints:
                wage = max(wage,0)
                wage = min(wage,1)

                w.wage = wage

                # link worker and employee together
                w.employer = f
                f.workers.add(w)

                # update worker belief
                w.belief_update()
            
            # update firm belief (no benchmark)
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
    
    
    def perform_benchmark(self):
        ''''''
        # sample firms
        number_firms_to_check = int(self.benchmark_proportion * len(self.firms))
        sampled_firms = gen.choice(self.firms,number_firms_to_check)

        # compute summary statistics
        all_wages = []
        for f in sampled_firms:
            for w in f.workers:
                all_wages.append(w.wage)
        
        return [np.min(all_wages), np.median(all_wages), np.max(all_wages)]

    def information_seeking(self):
        '''
        '''
        new_applicants = [[] for i in self.firms] # tracking which workers are trying to negotiate with new firms

        # worker seeking
        for w in self.workers:
            w.information_seeking(self.mvpt)

            if w.market_pay_belief[1] >= (1+w.search_threshold)*w.wage: # requires m_hat to be  >  wage
                new_firm = gen.choice(self.firms,1)[0] # randomly choose a firm (possibly including current firm)
                new_applicants[self.firms.index(new_firm)].append(w)
                w.negotiating_with = new_firm
            # elif w.market_pay_belief[2] >= (1+w.search_threshold)*w.wage:
            #     new_firm = w.employer # renegotiate with current firm if upper bound from colleague search is sufficiently high
            #     new_applicants[self.firms.index(new_firm)].append(w)
            #     w.negotiating_with = new_firm
            else:
                w.negotiating_with = None
        
        # firms always get benchmark
        for i,f in enumerate(self.firms):
            benchmark_for_f = self.perform_benchmark()
            # N_a = len(new_applicants[i])
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

        num_successes = 0

        for w in gen.permutation(self.workers): # randomize sequence of negotiations to prevent small group of workers consistently switching
            
            firm = w.negotiating_with # firm that worker is potentially negotiating with

            if firm == None:
                continue # skip workers not renegotiating

            # special cases
            if len(firm.workers)>= firm.capacity:
                continue # firm already at capacity
            
            if len(firm.workers) == 0:
                w.wage = w.market_pay_belief[1] # accept any wage offer 
                w.employer.workers.remove(w) # remove from previous firm 
                w.employer = firm # worker has firm as employer
                firm.workers.add(w) # firm has worker as employee
                w.num_successes = w.num_successes + 1
                num_successes = num_successes + 1
                continue # successful switch

            outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1],firm.market_pay_belief[2], w.market_pay_belief[1] - self.mvpt.error,w.market_pay_belief[0])
            
            if outcome == -1:
                w.num_failures = w.num_failures + 1
                # w.stop_sharing = (w.num_failures >= w.failure_threshold)
            else:
                w.wage = outcome
                w.employer.workers.remove(w) # remove from previous firm 
                w.employer = firm # worker has firm as employer
                firm.workers.add(w) # firm has worker as employee
                w.num_successes = w.num_successes + 1
                num_successes = num_successes +1

            # worker updates beliefs
            w.belief_update() 
        
        # firms update beliefs after all negotiations done
        for f in self.firms:
            f.belief_update()
        
        self.num_successes.append(num_successes) # tracking how many successful negotiations there are in the market


def get_wage_distribution_within_firm(firm):
    wages = []
    
    for w in firm.workers:
        wages.append(w.wage)
    counts, bins = np.histogram(wages, range=(0,1))

    return counts, bins

def plot_attribute_distribution_within_firm(f_idx,firm,attr, c, extra_counts = None, extra_bins = None):
    
    attribute_values = []
    
    for w in firm.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,range=(0,1))
    plt.stairs(counts,bins,label="Final Distribution")
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed",label="Initial Distribution")
    plt.title(f"Distribution of {attr} of Workers at Firm {f_idx}")
    plt.legend()
    plt.ylim((0,c))
    plt.savefig(f"simulation_results/seed={seed}/distribution_of_{attr}_at_firm_{f_idx}")
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


def check_convergence(market, stop_sharing_threshold=0.9, successes_threshold=50):

    num_stop_sharing = 0

    for w in market.workers:
        num_stop_sharing = num_stop_sharing + w.stop_sharing
    
    if num_stop_sharing >= int(stop_sharing_threshold*len(market.workers)):
        print(f"At least {stop_sharing_threshold} proportion of workers stopped sharing data, tool too unreliable.")
        return True

    if len(market.num_successes) >= successes_threshold and all([s == 0 for s in market.num_successes[-successes_threshold:]]):
        print(f"No successful negotiations for {successes_threshold} time steps.")
        return True
    
    return False




if __name__ == "__main__":

    # labor market parameters
    N_firms = 10
    N_workers = 50 # evenly split among firms initially
    firm_capacity = 20 # total number of employees a firm can have
    worker_productivity =  1 # "quality" of worker -- ASSUME wages in [0,1], assume wages always below productivity
    failure_threshold = 50 # tolerance of workers for number of failed negotiations
    search_threshold = 0.15 # threshold for workers to renegotiate with their current firm or seek out new firm, i.e., if they could make 1.s x their curent pay
    transparent_firm_multiplier = 0.4 # relatively reduces the chance of workers in transparent firms of sharing data
    benchmark_proportion = 1 # proportion of firms that get added to random sample, 1=> firms perfectly know wage distributions in market, shared beliefs
    sigma_e = 0.1
    initial_pool_proportion = 0.2


    # run simulation
    market = Market(N_f = N_firms, N_w=N_workers,C=firm_capacity,q=worker_productivity, T=failure_threshold, s=search_threshold, t_f_m = transparent_firm_multiplier,b_k = benchmark_proportion,s_e=sigma_e,i_p=initial_pool_proportion)
    
    initial_counts_bins = [get_wage_distribution_within_firm(firm) for firm in market.firms]
    m_hat_over_time = [market.mvpt.m_hat]
    l_hat_over_time = [market.mvpt.l_hat]
    u_hat_over_time = [market.mvpt.u_hat]
    mvpt_pool_size = [len(market.mvpt.data_pool)]

    worker_mvpt_confidence = [[] for w in range(N_workers)]

    for t in tqdm(range(50000)):
        market.market_time_step()
        m_hat_over_time.append(market.mvpt.m_hat)
        l_hat_over_time.append(market.mvpt.l_hat)
        u_hat_over_time.append(market.mvpt.u_hat)
        mvpt_pool_size.append(len(market.mvpt.data_pool)) 
        for i in range(N_workers):
            worker_mvpt_confidence[i].append(market.workers[i].mvpt_confidence)
        if check_convergence(market, 1-initial_pool_proportion, 500):
            break


    # analyze results 
    # print("fairness uncertainty")
    # for k in range(0,N_workers-5,5):
    #     for i in range(k,k+5):
    #         plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].fairness_uncertainty}")
    #     plt.legend()
    #     plt.show()
    print("final wage")
    for k in range(0,N_workers-5,5):
        for i in range(k,k+5):
            plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].wage}")
        plt.legend()
        plt.savefig(f"simulation_results/seed={seed}/mvpt_confidence_worker_group_k={k}")
        plt.show()
    # print("num_failures")
    # for k in range(0,N_workers-5,5):
    #     for i in range(k,k+5):
    #         plt.plot(worker_mvpt_confidence[i],label=f"{market.workers[i].num_failures}")
    #     plt.legend()
    #     plt.show()

    plt.plot(m_hat_over_time)
    plt.plot(l_hat_over_time, color="red")
    plt.plot(u_hat_over_time, color="purple")
    plt.ylim((0,1))
    plt.title("m_hat values over time")
    plt.savefig(f"simulation_results/seed={seed}/m_hat_over_time_seed")
    plt.show()
    plt.plot(mvpt_pool_size)
    plt.ylim((0,N_workers))
    plt.title("data pool size of mvpt")
    plt.savefig(f"simulation_results/seed={seed}/mvpt_pool_size")
    plt.show()
    print(f"Final m_hat value: {market.mvpt.m_hat}")
 
    for i,f in enumerate(market.firms):
        print(f"Firm id: {f}")
        print(f"number of workers: {len(f.workers)}")
        print(f"market pay beliefs: {f.market_pay_belief}")
        plot_attribute_distribution_within_firm(i,f,"fairness_uncertainty",firm_capacity)
        plot_attribute_distribution_within_firm(i,f,"wage",firm_capacity,initial_counts_bins[i][0],initial_counts_bins[i][1])
        
    num_failures = []
    num_successes = []
    for w in market.workers:
        num_failures.append(w.num_failures)
        num_successes.append(w.num_successes)


    plt.bar(range(len(market.workers)), num_failures,color="red")
    plt.bar(range(len(market.workers)), num_successes,color="blue")
    plt.title("number of failed (red) and successful (blue) negotiations by worker index")
    plt.savefig(f"simulation_results/seed={seed}/failed_and_successful_negotiations")
    plt.show()
        
    