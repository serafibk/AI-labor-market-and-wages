import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

seed = 707202508312025 # use for reproducibility of initial simulations to debug 
gen = np.random.default_rng(seed=seed)

# agents and market running functions (which can be moved to a separate file)

class Worker:

    def __init__(self,q,s1,s2,T):
        '''initialization of Worker agent 

        params
            q (float):   
            s1 (float): 
            s2 (float): 
            T (int): max number of failed negotiations a worker is willing to tolerate from MVPT.
            firm (Firm): instance of Firm that is Worker's initial employer.
        '''
        self.fairness_uncertainty = gen.beta(a=4,b=5) # fairness uncertainty salience, slightly left skewed distribution
        self.productivity = q
        self.s1 = s1
        self.s2 = s2
        self.num_failures = 0 # counter to track failed negotiations
        self.num_successes = 0 # counter to track successful negotiations
        self.failure_threshold = T
        self.employer = None # initially unemployed
        self.wage = None # initially unemployed

        self.negotiating_with = None # firm that worker is trying to negotiate with

        self.market_pay_belief = [0,0,0]

    def information_seeking(self,mvpt):
        '''
        '''
        if self.num_failures == -1:
            mvpt.remove_worker(self) # make sure worker is not in next data pool
            return # do not seek info

        p_seek_info = (self.fairness_uncertainty + self.employer.transparency_multiplier)/2 # average instead of multiplication to prevent too small of probabilities
        seek_info = gen.choice([0,1],p=[1-p_seek_info, p_seek_info])

        if seek_info and self.num_failures>=0: # self.num_failures = -1 if worker decides to never share data again
            mvpt.add_worker(self) # share data (if not in data pool already)
            self.belief_update(m_hat = mvpt.m_hat) # (view m_hat and update median pay belief)
        else:
            mvpt.remove_worker(self) # try removing worker if they don't actively share
        

    def belief_update(self,anchor=0.01,m_hat = None):
        '''
        '''
        if self.wage is None:
            print("Error: No wage set.")
            return

        if m_hat is not None:
            self.market_pay_belief[1] = m_hat # set for negotiations
        else:
            self.market_pay_belief = [self.wage*(1-anchor), self.wage, self.wage*(1+anchor)] # anchors wage upper and lower bound around current pay



class Firm:

    def __init__(self,t,C,b):
        '''initialization of Firm agent

        params
            t (float): multiplier between 0 and 1 that increases or decreases the chance of employed workers to seek information from MVPT.
            C (int): maximum number of employees firm can support.
        '''
        self.transparency_multiplier = t
        self.capacity = C
        self.benchmark_cost = b
        self.workers = set([]) # list of employed workers
        self.market_pay_belief = [0, 0, 0]
        self.bought_benchmark = 0
  
    def information_seeking(self, mvpt, N_a, benchmark):
        '''
        '''
        if N_a == 0: # assuming not seeking benchmark info if no new applicants
            self.bought_benchmark = 0
            return

        m_hat = mvpt.m_hat
        error = mvpt.error

        if self.market_pay_belief[1] + self.benchmark_cost / N_a <= (m_hat-error): # benchmark cost is worth potential savings
            self.belief_update(benchmark=benchmark) # update belief with benchmark
            self.bought_benchmark = 1
        else:
            self.bought_benchmark = 0

    def belief_update(self, benchmark=None):
        '''
        '''
        if benchmark is not None:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]+[benchmark[0]]), benchmark[1], np.max([w.wage for w in self.workers]+[benchmark[2]])] # update w.r.t. provided benchmark
        else:
            self.market_pay_belief = [np.min([w.wage for w in self.workers]), np.median([w.wage for w in self.workers]), np.max([w.wage for w in self.workers])] # update w.r.t. current wage distribution


class MVPT:

    def __init__(self, worker_data_pool, sigma_e):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
            sigma_e (float): standard deviation parameter of error term of median calculation. positively correlated with len(worker_data_pool).
        '''
        self.data_pool = worker_data_pool # to generate median 
        self.sigma_e = sigma_e # to generate error
        self.error = 0
        self.m_hat = 0

    
    def add_worker(self, worker):
        '''adds specified Worker object to data pool. decreases sigma_e by a small amount. 

        params
            worker (Worker): Worker object to add. 
        '''
        if worker not in self.data_pool:
            self.data_pool.add(worker)
            self.sigma_e = max(self.sigma_e - 1e3,0) # arbitrary decrease / floor for now

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e by a small amount.

        params
            worker (Worker): Worker object to remove. 
        '''
        if worker in self.data_pool:
            self.data_pool.remove(worker)
            self.sigma_e = min(self.sigma_e + 1e3,1) # arbitrary increase / cap for now

    def update_m_hat(self):
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated
        self.error = abs(error) # record magnitude of error in either direction to report
        self.m_hat = np.median([w.wage for w in self.data_pool]) + error
        self.m_hat = min(self.m_hat,1)
        self.m_hat = max(self.m_hat,0)

class Market:
    '''Market class is used to create a single market instance to keep track of the connections between firms and workers and MVPT. Additionally, carries out each phase of the market evolution.'''

    def __init__(self, N_f, N_w, C,q,T,s1,s2,t_f_m,b,b_k,s_e):
        '''initializaiton of the market

        params:

        '''
        self.benchmark_cost = b
        self.benchmark_proportion = b_k

        self.firms = [Firm(t_f_m, C,b) for i in range(int(N_f / 2))] + [Firm(1-t_f_m,C,b) for i in range(int(N_f / 2))] # half firms transparent and half not
        self.workers = [Worker(q,s1,s2,T) for i in range(N_w)] # all identical workers to start

        # initial matching of firms and workers (which sets their initial beliefs)
        self._hire_initial_workers(t_f_m,np.split(np.asarray(self.workers), N_f))

        initial_worker_pool = set(gen.permutation(self.workers)[:int(N_w * 0.1)]) # grab 10% of initial workers randomly for MVPT
        self.mvpt = MVPT(initial_worker_pool,s_e)
        self.mvpt.update_m_hat()
        
    
    def _hire_initial_workers(self, t_f_m, worker_assignment):
        '''
        '''
        
        for i,f in enumerate(self.firms):
            for w in worker_assignment[i]:
                # generate initial wage
                sigma_f = 0.5
                if f.transparency_multiplier == t_f_m:
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

        self.mvpt.update_m_hat()
    
    
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

            if w.market_pay_belief[1] >= (1+w.s2)*w.wage:
                new_firm = gen.choice(list(set(self.firms) - set([w.employer])),1)[0] # randomly choose a different firm 
                new_applicants[self.firms.index(new_firm)].append(w)
                w.negotiating_with = new_firm
            elif w.market_pay_belief[1] >= (1+w.s1)*w.wage:
                w.negotiating_with = w.employer
            else:
                w.negotiating_with = None
        
        # firm seeking benchmark info based on how many applicants they get
        for i,f in enumerate(self.firms):
            benchmark_for_f = self.perform_benchmark()
            N_a = len(new_applicants[i])
            f.information_seeking(self.mvpt, N_a, benchmark_for_f)

    def negotiation_and_belief_update(self):
        '''
        '''
        def _bargaining_outcome(o1, at1, at2, o2, at1_c, benchmarked):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(a_t_1), counter offer cap (a_t_2), '''
            if benchmarked:
                if o1 <= at1: # opening offer lower than firm's median belief, accept
                    return o1
                elif o1 <= at2 and o2 >= at1_c: # opening offer lower than firm's upper belief, worker accepts reduced counteroffer
                    return o2
                else:
                    return -1
            else:
                if o2 >= at1_c: # worker accepts error-corrected mvpt offer if it is at least their belief lower bound
                    return o2
                else:
                    return -1

        m_hat = self.mvpt.m_hat
        error = self.mvpt.error

        for w in gen.permutation(self.workers): # randomize sequence of negotiations to prevent small group of workers consistently switching
            
            firm = w.negotiating_with # firm that worker is potentially negotiating with

            if firm == None:
                continue # skip workers not renegotiating

            if firm == w.employer: # renegotiation
                
                if firm.bought_benchmark:
                    outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1], firm.market_pay_belief[2], (w.market_pay_belief[1] + firm.market_pay_belief[1]) / 2,w.market_pay_belief[0],1)
                else:
                    outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1], firm.market_pay_belief[2], w.market_pay_belief[1] - error,w.market_pay_belief[0],0)
                
                if outcome == -1:
                    w.num_failures = w.num_failures + 1
                    stop_sharing = gen.choice([0,1], p=[(1-(w.num_failures)/(w.failure_threshold)),(w.num_failures)/(w.failure_threshold)])
                    if stop_sharing:
                        w.num_failures = -1 # will be removed from data pool next time step
                else:
                    w.wage = outcome
                    w.num_successes = w.num_successes + 1

            else: # negotiation with a new firm

                # special cases
                if len(firm.workers)>= firm.capacity:
                    continue # firm already at capacity
                
                if len(firm.workers) == 0:
                    # print("SWITCH")
                    w.wage = w.market_pay_belief[1] # accept any wage offer 
                    w.employer.workers.remove(w) # remove from previous firm 
                    w.employer = firm # worker has firm as employer
                    firm.workers.add(w) # firm has worker as employee
                    w.num_successes = w.num_successes + 1

                if firm.bought_benchmark:
                    outcome = outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1],firm.market_pay_belief[2], (w.market_pay_belief[1] +firm.market_pay_belief[1]) / 2,w.market_pay_belief[0],1)
                else:
                    outcome = _bargaining_outcome(w.market_pay_belief[1], firm.market_pay_belief[1], firm.market_pay_belief[2], w.market_pay_belief[1] - error,w.market_pay_belief[0],0)
                
                if outcome == -1:
                    w.num_failures = w.num_failures + 1
                    stop_sharing = gen.choice([0,1], p=[(1-(w.num_failures)/(w.failure_threshold)),(w.num_failures)/(w.failure_threshold)])
                    if stop_sharing:
                        w.num_failures = -1 # will be removed from data pool next time step
                else:
                    # print("SWITCH")
                    w.wage = outcome
                    w.employer.workers.remove(w) # remove from previous firm 
                    w.employer = firm # worker has firm as employer
                    firm.workers.add(w) # firm has worker as employee
                    w.num_successes = w.num_successes + 1

            # worker updates beliefs
            w.belief_update() 
        
        # firms update beliefs after all negotiations done
        for f in self.firms:
            f.belief_update()


def get_wage_distribution_within_firm(firm):
    wages = []
    
    for w in firm.workers:
        wages.append(w.wage)
    counts, bins = np.histogram(wages, range=(0,1))

    return counts, bins

def plot_attribute_distribution_within_firm(f_idx,firm,attr, extra_counts = None, extra_bins = None):
    
    attribute_values = []
    
    for w in firm.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,range=(0,1))
    plt.stairs(counts,bins)
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed")
    plt.title(f"Distribution of {attr} of Workers at Firm {firm}")
    plt.savefig(f"opt_in_simulation_results/distribution_of_{attr}_at_firm_{f_idx}_seed={seed}")
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


def check_for_convergence():
    #todo
    pass


if __name__ == "__main__":

    # labor market parameters
    N_firms = 10
    N_workers = 100 # evenly split among firms initially
    firm_capacity = 20 # total number of employees a firm can have
    worker_productivity =  1 # "quality" of worker -- ASSUME wages in [0,1], assume wages always below productivity
    failure_threshold = 10 # tolerance of workers for number of failed negotiations
    s1 = 0.05 # threshold for workers to renegotiate with their current firm
    s2 = 0.1 # threshold for workers to seek out a new firm 
    transparent_firm_multiplier = 0.1 # relatively reduces the chance of workers in transparent firms of sharing data
    benchmark_cost = 0 # cost of high quality benchmark, possibly leave it cost free
    benchmark_proportion = 0.6 # proportion of firms that get added to random sample
    sigma_e = 0.5


    # run simulation
    market = Market(N_f = N_firms, N_w=N_workers,C=firm_capacity,q=worker_productivity, T=failure_threshold, s1=s1,s2=s2, t_f_m = transparent_firm_multiplier,b=benchmark_cost,b_k = benchmark_proportion,s_e=sigma_e)
    
    initial_counts_bins = [get_wage_distribution_within_firm(firm) for firm in market.firms]
    m_hat_over_time = [market.mvpt.m_hat]
    mvpt_pool_size = [len(market.mvpt.data_pool)]

    for t in tqdm(range(1000)):
        market.market_time_step()
        m_hat_over_time.append(market.mvpt.m_hat)
        mvpt_pool_size.append(len(market.mvpt.data_pool))
        if t % 100 == 0:
            print(f"m_hat error: {market.mvpt.error}")
            print(f"benchmark choices: {[f.bought_benchmark for f in market.firms]}")

    # analyze results 
    plt.plot(m_hat_over_time)
    plt.title("m_hat values over time")
    plt.savefig(f"opt_in_simulation_results/m_hat_over_time_seed={seed}")
    plt.show()
    plt.plot(mvpt_pool_size)
    plt.title("data pool size of mvpt")
    plt.savefig(f"opt_in_simulation_results/mvpt_pool_size_seed={seed}")
    plt.show()
    print(f"Final m_hat value: {market.mvpt.m_hat}")
 
    for i,f in enumerate(market.firms):
        print(f"Firm id: {f}")
        print(f"number of workers: {len(f.workers)}")
        print(f"market pay beliefs: {f.market_pay_belief}")
        plot_attribute_distribution_within_firm(i,f,"fairness_uncertainty")
        plot_attribute_distribution_within_firm(i,f,"wage",initial_counts_bins[i][0],initial_counts_bins[i][1])
        
    num_failures = []
    num_successes = []
    for w in market.workers:
        num_failures.append(w.num_failures)
        num_successes.append(w.num_successes)


    plt.bar(range(len(market.workers)), num_successes,color="blue")
    plt.bar(range(len(market.workers)), num_failures,color="red")
    plt.title("number of failed (red) and successful (blue) negotiations by worker index")
    plt.savefig(f"opt_in_simulation_results/failed_and_successful_negotiations_seed={seed}")
    plt.show()
        
    