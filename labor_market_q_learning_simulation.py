'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 42
gen = np.random.default_rng(seed=seed)

update_T = 1 # only if needed


class MVPT:

    def __init__(self, worker_data_pool,W, N_f):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
        '''
        # data / noise parameter for predictions
        self.data_pool = [set() for f in range(N_f)]
        for w in worker_data_pool:
            self.data_pool[w.employer].add(w) 
        self.wages = W # possible wages to map onto 

        # information tool providees
        self.m_hat = 0 # overall 
        self.u_hat = [-1 for f in range(N_f)] # 90th percentile per firm 
        self.l_hat = [-1 for f in range(N_f)] # 10th percentile per firm 

    
    def add_worker(self, worker):
        '''adds specified Worker object to data pool. decreases sigma_e. 

        params
            worker (Worker): Worker object to add to data set. 
        '''
        if worker.employer == None:
            return # can't share data if you aren't employed!
        self.data_pool[worker.employer].add(worker) # track wage for their current employer

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e.

        params
            worker (Worker): Worker object to remove from data set. 
        '''
        for dataset in self.data_pool: # they may have a new employer now, so just loop over all since discard doesn't throw an error
            if worker in dataset:
                dataset.discard(worker)

    def get_mvpt_range(self, f_idx):
        '''return 10th and 90th percentile of firm, based on shared data'''
        return (self.l_hat[f_idx], self.u_hat[f_idx])
    
    def get_mvpt_median(self):
        '''if workers share, then they can see a market-wide median prediction for their role'''
        return self.m_hat

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''

        # update u_hat, l_hat
        for idx, pool in enumerate(self.data_pool):
            if len(pool) == 0:
                continue # can't update this firm 
            l_hat_f = np.percentile([w.wage for w in pool], 10)
            u_hat_f = np.percentile([w.wage for w in pool], 90)
            self.l_hat[idx] = min(self.wages, key = lambda x:abs(x-l_hat_f)) # map to closest value
            self.u_hat[idx] = min(self.wages, key = lambda x:abs(x-u_hat_f)) # map to closest value

        # update m_hat from all of the wages
        if len([w.wage for dataset in self.data_pool for w in dataset]) == 0:
            return
        estimated_median = np.percentile([w.wage for dataset in self.data_pool for w in dataset],50)
        self.m_hat = min(self.wages, key = lambda x:abs(x-estimated_median))# mapping to closest value in wage list


class SalaryBenchmark:
    '''salary benchmarking tool that firms have access to. They collect coarse salary information, and augment it for sharing firms with firm-reported wage data.    '''

    def __init__(self, firm_data_pool, all_firms, W):
        '''initialization of market value preictor tool (MVPT)
        
        params
            firm_data_pool (Set[Firm]): pool of initial firms who are sharing their wage information
        '''
        # data / noise parameter for predictions
        self.data_pool = firm_data_pool # to generate more granular estimates 
        self.wages = W # possible wages to map onto 

        # granular information  provided by the tool
        self.data_pool_10 = 0 # 10th percentile
        self.data_pool_50 = 0 # 50th percentile
        self.data_pool_90 = 0 # 90th percentile


    def add_firm(self, firm):
        '''adds specified Firm object to data pool.

        params
            firm (Firm): Firm object to add to data set. 
        '''
        self.data_pool.add(firm)

    
    def remove_firm(self, firm):
        '''removes specified Firm object from data pool.

        params
            firm (Firm): Firm object to remove from data set. 
        '''
        self.data_pool.discard(firm)
    
    def get_benchmark_range(self):
        '''just return upper and lower bound'''
        return (float(self.data_pool_10),float(self.data_pool_90))
    
    def get_benchmark_median(self):
        '''return median for those that choose to use the benchmarks range'''
        return float(self.data_pool_50)

    def update_benchmark(self): 
        '''updates firm data pool benchmark
        '''
        if len(self.data_pool) == 0:
            # print("Error, data pool empty")
            return 
        elif len([w for f in self.data_pool for w in f.workers]) == 0: # none of the participating firms have workers
            return 

        # update firm-shared benchmark 
        dp_10 = np.percentile([w.wage for f in self.data_pool for w in f.workers], 10)
        self.data_pool_10 = min(self.wages, key = lambda x:abs(x-dp_10))# mapping to closest value in wage list
        dp_50 = np.percentile([w.wage for f in self.data_pool for w in f.workers], 50)
        self.data_pool_50 = min(self.wages, key = lambda x:abs(x-dp_50))# mapping to closest value in wage list
        dp_90 = np.percentile([w.wage for f in self.data_pool for w in f.workers], 90)
        self.data_pool_90 = min(self.wages, key = lambda x:abs(x-dp_90))# mapping to closest value in wage list

class Market:
    '''Market class defines the environment that the workers are learning to share data in (or not). greatly simplified from social learning setting.
    '''

    def __init__(self, N_w, N_f, s_w_sharing, a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,s_f_benchmark,a_f_benchmark, s_f_negotiation,W,p,alpha,delta,initial_belief_strength,p_risky = 0,type_conditioning=False, p_reset= 0.5, beta = 6.91*10**(-4),use_mvpt = False, posting = False, mixed_posts = False):
        '''initializaiton of the market

        Parameters
            N_w: number of workers
        '''
        # setting parameters
        self.use_mvpt = use_mvpt 
        self.post_ranges = posting
        self.mixed_posts = mixed_posts # mvpt estimated and firm provided ranges
        self.wages = W   
        self.p_reset_wage = p_reset

        # explore epsilon
        self.beta = beta
        self.explore_epsilon = 1 # start at 1 

        # create Workers
        print("Initializing workers...")
        if p_risky >0:
            self.workers = [Worker(s_w_sharing,a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,gen.choice(self.wages,p=p),gen.choice(["risky","safe"],p=[p_risky,1-p_risky]),alpha,delta,initial_belief_strength) for i in range(N_w)] # workers risky or safe
        else:
            self.workers = [Worker(s_w_sharing,a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,gen.choice(self.wages,p=p),"symmetric",alpha,delta,initial_belief_strength) for i in range(N_w)] # symmetric up to initial wage
        print("Initializing firms...")
        self.firms = [Firm(s_f_benchmark,a_f_benchmark, s_f_negotiation,self.wages,type_conditioning,alpha,delta,initial_belief_strength) for i in range(N_f)]  ## assuming firms have enough capacity for all workers in our market and all have constant returns to scale from the set of workers that apply 
        worker_groups = np.split(np.array(self.workers), N_f)
        for i in range(N_f):
            for j in range(len(worker_groups[i])):
                self.firms[i].workers.append(worker_groups[i][j]) # connect worker
                worker_groups[i][j].employer = i # connect to employer

        # for tracking matches for negotiation
        self.negotiation_matches = [[] for i in range(N_f)] # a set of workers associated with the firm they will attempt to negotiate with

        # ensure at least one worker starts with MRPL
        expert_worker_idx = gen.integers(0,N_w)
        self.workers[expert_worker_idx].wage = float(1) 

        # CORRELATION TEST: bottom 1-p_risky percentile safe and p_risky percent risky
        if p_risky > 0:
            all_wages = [w.wage for w in self.workers]
            safe_threshold = np.percentile(all_wages,(1-p_risky)*100)
            num_safe = int(N_w * (1-p_risky))
            count_safe = 0
            for w in self.workers:
                if w.wage <=safe_threshold and count_safe < num_safe:
                    w.type = "safe"
                    count_safe = count_safe +1
                else:
                    w.type = "risky"
            # print(count_safe)

        print("Agents initialized.")

        # initializing salary benchmark (used in all settings)
        initial_firm_pool = set()
        for f in self.firms:
            if gen.binomial(1,0.5) >= 0.5: # fair coin toss to be in initial pool
                initial_firm_pool.add(f)
            
        self.salary_benchmark = SalaryBenchmark(initial_firm_pool, self.firms, self.wages)
        self.salary_benchmark.update_benchmark()

        # initializing mvpt after workers have an initial wage IF applicable to setting
        if self.use_mvpt:
            initial_worker_pool = set()
            for w in self.workers:
                if gen.binomial(1,0.5) >= 0.5:
                    initial_worker_pool.add(w) # initial pool generated with adding in workers uniform iid
            
            self.mvpt = MVPT(initial_worker_pool,self.wages,N_f) # initial worker pool is random
            self.mvpt.update_mvpt()

        # initializing data structure for posted ranges IF applicable to setting
        if posting:
            self.posted_ranges = [(0,0) for f in range(N_f)]
        
        for idx,f in enumerate(self.firms):
            f.update_individual_wage_distribution()
            if self.use_mvpt and -1 not in self.mvpt.get_mvpt_range(idx):
                f.state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range(), self.mvpt.get_mvpt_range(idx))
            else:
                f.state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range())
        
        # tracking market behavior for output graphs
        self.num_negotiations = []
        self.firm_information_use = [] # a tuple of (num sal bench, num independent) (mvpt use (if applicable) can be inferred from this)
        self.firm_information_source_values = [] # a tuple of (sal bench range, ind range 1, mvpt range 1, ind range 2, mvpt range 2,...)
        self.worker_mvpt_use = []
    
    def market_time_step(self,t):
        '''In a market time step, agents first seek information, then perform negotiations and update their beliefs, finally mvpt is updated for next round based on the data pool gathered this round

        Parameters
            T (int): total time in the simulation run, used in expectation calculations 
            t (int): current time step, used in expectation calculations
        '''
        # parameter  update
        self.explore_eps = 1*np.e**(-self.beta * t)

        # phase 1 - information seeking
        self.coarse_information_and_sharing_choice(t)  # firms and workers decide to share their private info

        # phase 2 - setting negotiation strategy
        self.negotiation_strategy_choice(t)

        # phase 3 - negotiation and belief updates    
        self.negotiation_and_state_transition(t)

    
    
    def _get_market_median(self):
        '''compute true median in market. I think this is only for tracking purposes now
        '''
        # compute summary statistics
        all_wages = []
        for w in self.workers:
            all_wages.append(w.wage)
        
        return min(W, key=lambda x:abs(x-np.median(all_wages)))

    def coarse_information_and_sharing_choice(self,t):
        ''' firms first choose a benchmark, then workers choose a firm to negotiate with 
        '''
        # fim seeking 

        # tracking info 
        using_bench = 0
        using_independent = 0
        ranges = (self.firms[0].state[1],) # sal benchmark recorded

        for idx,f in enumerate(self.firms):
            if t%update_T ==0:
                f.benchmark = f.action_decision(explore_eps=self.explore_eps,benchmark=True)
            ranges = ranges + (f.state[0],)
            if self.use_mvpt:
                ranges = ranges + (f.state[2],)
            
            if f.benchmark == "salary benchmark":
                self.salary_benchmark.add_firm(f) # for next round
                l_b, u_b= self.salary_benchmark.get_benchmark_range()
                next_state = (l_b, self.salary_benchmark.get_benchmark_median(), u_b)
                if t%update_T == 0:
                    f.update_q_vals(next_state,f.benchmark,benchmark=True)
                f.state = next_state
                using_bench = using_bench + 1
            else:
                self.salary_benchmark.remove_firm(f)
                if f.benchmark == "independent":
                    l_f, u_f = f.get_individual_range()
                    next_state = (l_f, f.get_individual_median(),u_f)
                    using_independent = using_independent + 1
                else:
                    l_m, u_m = self.mvpt.get_mvpt_range(idx)
                    next_state = (l_m, f.get_individual_median(),u_m)
                if t%update_T == 0:
                    f.update_q_vals(next_state,f.benchmark,benchmark=True)
                f.state = next_state
            
            # update range for tracking
            f.range = (f.state[0],f.state[2])

        self.firm_information_use.append((using_bench,using_independent))
        self.firm_information_source_values.append(ranges)

        # set worker's state with posted ranges 
        if self.post_ranges:
            min_val = 1.0
            max_val = 0.0
            for idx, f in enumerate(self.firms):
                if (self.use_mvpt and self.mixed_posts) or (not self.use_mvpt and not self.mixed_posts):
                    self.posted_ranges[idx] = (f.state[0],f.state[2])
                elif self.use_mvpt:
                    l_m, u_m = self.mvpt.get_mvpt_range(idx)
                    self.posted_ranges[idx] = (l_m,u_m)
                else:
                    print(f"Error: invalid setting flags self.post_ranges={self.post_ranges}, self.use_mvpt={self.use_mvpt}, self.mixed_posts={self.mixed_posts}.")

                if self.posted_ranges[idx][0] < min_val and self.posted_ranges[idx][0] >=0:
                    min_val = self.posted_ranges[idx][0]
                if self.posted_ranges[idx][1] > max_val:
                    max_val = self.posted_ranges[idx][1]
            
            for w in self.workers:
                if self.use_mvpt: # need to make a sharing decision first
                    next_state = (float(w.wage), min_val, max_val)
                else: # straight to firm choice
                    next_state = ()
                    for idx in range(len(self.posted_ranges)):
                        if self.posted_ranges[idx][1] > w.wage and -1 not in self.posted_ranges[idx]:
                            next_state = next_state + (min(self.wages, key = lambda x:abs(x-(self.posted_ranges[idx][1]-self.posted_ranges[idx][0]))),)
                        else:
                            next_state = next_state + (-1,) # range not applicable

                if t>0 and t%update_T == 0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt)  
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state
            
        else: # no posted ranges, workers have no information 
            for w in self.workers:
                next_state = (-1,-1,-1,-1,-1) # state shows that worker knows nothing before choosing a firm  
        
                if t>0 and t%update_T == 0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt)
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state


        # worker sharing decision if using MVPT 
        if self.use_mvpt:
            for w in self.workers:
                mvpt_share = 0
                if t%update_T == 0 or t ==0:
                    w.share = w.action_decision(explore_eps = self.explore_eps,sharing=True)
                
                if w.share == "share":
                    self.mvpt.add_worker(w) # add to data pool for next round
                    mvpt_share = mvpt_share + 1
                    res_wage = max(w.wage, self.mvpt.m_hat) # learned about m_hat
                else:
                    self.mvpt.remove_worker(w) # remove worker from data pool for next round
                    res_wage = w.wage

                # move to firm choice state
                next_state = ()
                for idx in range(len(self.posted_ranges)):
                    if self.posted_ranges[idx][1] > res_wage:
                        next_state = next_state + (min(self.wages, key = lambda x:abs(x-(self.posted_ranges[idx][1]-self.posted_ranges[idx][0]))),)
                    else:
                        next_state = next_state + (-1,) # range not applicable
                if t%update_T == 0 or t==0:
                    w.update_q_vals(next_state,w.share,sharing=True)
                w.state = next_state
             #tracking   
            self.worker_mvpt_use.append(mvpt_share)
    
    def negotiation_strategy_choice(self,t):
        '''
        '''
        # worker firm choice and offer decision
        for w in self.workers:
            # choose a firm to negotiate with based on range of posted wages (+ assumption that -1 => no knowledge or u<=res wage)
            if t%update_T == 0 or t==0:
                w.firm_negotiation_choice = int(w.action_decision(firm_choice = True,explore_eps=self.explore_eps))

            # determine next state
            if w.type == "symmetric":
                offer_low = w.wage
                if self.post_ranges:
                    if self.use_mvpt and w.share == "share":
                        offer_low = max(w.wage, self.mvpt.m_hat)
                    if self.posted_ranges[w.firm_negotiation_choice][1]>offer_low:
                        offer_high = self.posted_ranges[w.firm_negotiation_choice][1]
                    else:
                        offer_high = min(float(1), min(self.wages, key= lambda x: abs(x-(offer_low+self.wages[1])))) # high offer is one interval up, capped at 1
                else:
                    offer_high = min(float(1), min(self.wages, key= lambda x: abs(x-(offer_low+self.wages[1])))) # high offer is one interval up, capped at 1
            elif w.type == "risky":
                offer_low = w.wage
                if self.post_ranges:
                    if self.use_mvpt and w.share == "share":
                        offer_low = max(w.wage, self.mvpt.m_hat)
                    if self.posted_ranges[w.firm_negotiation_choice][1]>offer_low:
                        offer_high = self.posted_ranges[w.firm_negotiation_choice][1]
                    else: 
                        offer_high = min(float(1), min(self.wages, key= lambda x: abs(x-(offer_low+self.wages[1]))))#float(1)# high offer is 1 
                else:
                    offer_high = min(float(1), min(self.wages, key= lambda x: abs(x-(offer_low+self.wages[1]))))#float(1) # high offer is 1
            elif w.type == "safe":
                offer_low = w.wage
                offer_high = min(float(1), min(self.wages, key= lambda x: abs(x-(offer_low+self.wages[1])))) # high offer is one interval up, capped at 1, ignore ranges 
            else:
                print(f"Error: unrecognized worker type '{w.type}'.")
            
            next_state = (offer_low, offer_high)
            if t%update_T == 0 or t==0:
                if self.post_ranges:
                    w.update_q_vals(next_state, w.firm_negotiation_choice, firm_choice=True,firm_upper_bound=self.posted_ranges[w.firm_negotiation_choice][1])
                else:
                    w.update_q_vals(next_state, w.firm_negotiation_choice, firm_choice=True,firm_upper_bound=-1)
            w.state = next_state

            # offer decision
            if t%update_T == 0 or t==0:
                w.offer = w.action_decision(explore_eps=self.explore_eps)
            if w.offer != "no offer": # only if you make an offer do you get to move on
                w.offer = float(w.offer) # cast to float now
                self.negotiation_matches[w.firm_negotiation_choice].append(w)
        
        
        # firm acceptance threshold decision   
        if t%update_T == 0: 
            for f in self.firms:
                if f.type_conditioning:
                    f.acceptance_threshold = f.action_decision(explore_eps=self.explore_eps)
                    f.acceptance_threshold = (float(f.acceptance_threshold[0]),float(f.acceptance_threshold[1]))
                else:
                    f.acceptance_threshold = float(f.action_decision(explore_eps=self.explore_eps)) # have to make an acceptance threshold decision no matter what
        

    def negotiation_and_state_transition(self,t):
        '''
        '''
        def _bargaining_outcome(offer, acceptance_threshold):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(at1), counter offer cap (at2),  counter offer (o2), worker counter offer acceptance threshold (at1_c)'''
            if offer <= acceptance_threshold: # opening offer lower than firm's acceptance threshold, accept.
                return offer
            else: # opening offer too high, reject
                return -1

        # update mvpt based on workers' current wages, signals ready for next state
        # if t%2 == 0:
        if self.use_mvpt:
            self.mvpt.update_mvpt() 

        num_negotiations = 0
        accepted_offers = [[] for f in self.firms]

        for f_idx,f in enumerate(self.firms):
            num_offers = len(self.negotiation_matches[f_idx])
            if f.type_conditioning == True:
                AT_risky = f.acceptance_threshold[0]
                AT_safe = f.acceptance_threshold[1]
            else:
                AT = f.acceptance_threshold
            for w in self.negotiation_matches[f_idx]: # loop through each worker initiating negotiation
                num_negotiations = num_negotiations + 1 # tracking

                if w.type == "risky":
                    w.bargaining_outcome = _bargaining_outcome(w.offer, AT_risky)
                elif w.type == "safe":
                    w.bargaining_outcome = _bargaining_outcome(w.offer, AT_safe)
                elif w.type == "symmetric":
                    w.bargaining_outcome = _bargaining_outcome(w.offer, AT)
                else:
                    print(f"Error: invalid type cponditioning flag {f.type_conditioning} + worker type {w.type}")
                w.last_wage = w.wage # track wage before updating

                if w.bargaining_outcome < 0: # if negotiation fails, some small probability that worker's wage gets reset.
                    reset_wage = gen.binomial(1, self.p_reset_wage)
                    
                    if reset_wage:
                        w.wage = float(0) # their "leisure wage" or some notion of reservation utility -- should they still be employed?
                        if w.employer != None:
                            self.firms[w.employer].workers.remove(w) 
                        w.employer = None # they are unemployed
                else:# successful negotiation
                    w.wage = float(w.bargaining_outcome) # new wage acheived
                    accepted_offers[f_idx].append(w.bargaining_outcome)

                    # re-employ worker
                    if w.employer != None:
                        self.firms[w.employer].workers.remove(w)
                    w.employer = f_idx
                    self.firms[f_idx].workers.append(w)
                
                # next state of worker determined after next posted benchmark. reward update there.

        
        # update benchmarks based on new employment data
        if t%update_T ==0:
            self.salary_benchmark.update_benchmark()

        # rewards and state transitions for firms and workers, self.salary_benchmark.get_gov_data() 
        for idx, f in enumerate(self.firms): # all firms have a negotiation update
            if t%update_T ==0:
                f.update_individual_wage_distribution()
            if self.use_mvpt and -1 not in self.mvpt.get_mvpt_range(idx):
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range(),self.mvpt.get_mvpt_range(idx))
            else:
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range())
            if t%update_T == 0:
                f.update_q_vals(next_state,f.acceptance_threshold, accepted_offers = accepted_offers[idx], num_offers = len(self.negotiation_matches[idx]))
            f.state = next_state

        # for w in self.workers:
        #     next_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
        #     if len(w.state) == 3 and w.offer == None: # didn't share
        #         w.update_q_vals(next_state,"no share", sharing = True)
        #         w.state = next_state # update worker's new state
        #     elif  w.offer == "no offer": # shared, but didn't offer
        #         w.update_q_vals(next_state,w.offer)  
        #         w.state = next_state # update worker's new state
        
    
        self.num_negotiations.append(num_negotiations) # tracking
        self.negotiation_matches = [[] for i in range(N_f)] # reset negotitation matches for next round


class Worker:

    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, firm_choice_states,firm_choice_actions, initial_wage, worker_type, alpha = 0.5, delta = 0.8,initial_belief_strength=0.5):

        # environment parameters worker has access to
        self.share_states = share_states
        self.share_actions = share_actions # share or not share
        self.negotiation_states = negotiation_states
        self.negotiation_actions = negotiation_actions # offer choice
        self.firm_choice_states = firm_choice_states
        self.firm_choice_actions = firm_choice_actions

        # Q-learning params
        self.alpha = alpha
        self.delta = delta

        # dictionary of Q vals
        share_q_vals = dict({f"({s},{a})": 0 for s in self.share_states for a in self.share_actions}) # initial q-values
        negotiation_q_vals = dict({f"({s},{a})": 0 for s in self.negotiation_states for a in self.negotiation_actions}) # initial q-values
        firm_choice_q_vals = dict({f"({s},{a})": 0 for s in self.firm_choice_states for a in self.firm_choice_actions})
        share_and_negotiation = share_q_vals | negotiation_q_vals# merge dictionaries
        self.Q_vals = share_and_negotiation | firm_choice_q_vals # another merge

        # worker characteristics
        self.state = None
        self.offer = None
        self.share = None
        self.firm_negotiation_choice = None
        self.wage = initial_wage
        self.employer = None
        self.type = worker_type

        # bargaining outcome tracking
        self.last_wage = initial_wage
        self.bargaining_outcome = None

         # initial belief strength on sharing (possibly consider later)
        # for w,l,u in self.share_states:
        #     if w < u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = initial_belief_strength # upweight sharing if your wage is strictly below upper bound
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = -1*initial_belief_strength # downweight no sharing if your wage is strictly below upper bound
        #     elif w >= u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = -1*initial_belief_strength # downweight sharing if your wage is at least the upper bound of the range
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = initial_belief_strength # upweight sharing if your wage is at least the upper bound of the range
        
        # for w,l,m,u in self.negotiation_states:
            
        #     for o in self.negotiation_actions:
        #         if o == 'no offer':
        #             if w >= u:
        #                 self.Q_vals[f"({(w,l,m,u)},{'no offer'})"] = initial_belief_strength  # prefer not to offer when wage is above upper bound
        #             else:
        #                 self.Q_vals[f"({(w,l,m,u)},{'no offer'})"] = -1*initial_belief_strength  # prefer to offer when wage is below upper bound
        #         else:
        #             if o > w:
        #                 self.Q_vals[f"({(w,l,m,u)},{o})"] = initial_belief_strength # prefer to make larger offers
        #             else:
        #                  self.Q_vals[f"({(w,l,m,u)},{o})"] = -1*initial_belief_strength # prefer not to offer lower than current wage
                

    def get_reward(self, sharing = None, firm_choice = False, bargaining_outcome = None, last_wage=None, eps_share=1e-2,firm_upper_bound =None):
        '''sharing is None, True or False. bargaining_outcome is None, -1 or >0
        '''
        if sharing is not None:
            if not sharing:
                return 0
            else:
                return eps_share
        elif firm_choice:
            if firm_upper_bound != -1:
                return firm_upper_bound - self.wage
            else:
                return 0
        else:
            if bargaining_outcome == None: # some benefit to sharing w/ no negotiation
                return 0
            elif bargaining_outcome >= 0: # only if successful
                if last_wage == None:
                    print("Error: need to track last wage for this bargaining outcome.")
                    return 0
                return bargaining_outcome - last_wage
            else:
                return bargaining_outcome

    def action_decision(self, sharing = False, firm_choice = False, explore_eps = None):

        explore = gen.binomial(1,explore_eps) # decide to explore or exploit

        if not sharing and not firm_choice:
            negotiation_actions =  [a for a in self.negotiation_actions[1:] if (a>=self.state[0]) and (a<= self.state[1])]
            negotiation_actions.append("no offer")

        if explore: # choose an action U.A.R.
            if sharing:
                action = gen.choice(self.share_actions,1)[0] 
            elif firm_choice:
                action = gen.choice(self.firm_choice_actions,1)[0] 
            else:
                action = gen.choice(negotiation_actions,1)[0]
        else:
            if sharing:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.share_actions])
                action = self.share_actions[action_index]
            elif firm_choice:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.firm_choice_actions])
                action = self.firm_choice_actions[action_index]
            else:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in negotiation_actions])
                action = negotiation_actions[action_index]
        
        return action
    
    def update_q_vals(self, new_state, action, sharing = False, firm_choice=False, bargaining_outcome = None,last_wage=None,using_mvpt=False,firm_upper_bound =None):
        # update step: Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))

        ## get rewards
        if sharing:
            if action == "share":
                reward = self.get_reward(sharing=True)
            else:
                reward = self.get_reward(sharing=False)
        elif firm_choice:
            reward = self.get_reward(firm_choice = True,firm_upper_bound = firm_upper_bound)
            negotiation_actions =  [a for a in self.negotiation_actions[1:] if (a>=self.state[0]) and (a<= self.state[1])]
            negotiation_actions.append("no offer")
        else:
            reward = self.get_reward(bargaining_outcome=bargaining_outcome,last_wage=last_wage)


        ## q-value update
        if sharing: # moving to a firm choice state
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.firm_choice_actions]))
        elif firm_choice: # moving to a negotiation state
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in negotiation_actions]))
        else: # else, moving to a sharing state
            if using_mvpt:
                self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.share_actions]))
            else: # moving back to firm choice state
                self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.firm_choice_actions]))

    
class Firm:
    '''implementation of firms in this market who are now also q-learning their negotiation strategy under different information conditions.'''
    def __init__(self, benchmark_states, benchmark_actions, negotiation_states, W, type_conditioning, alpha = 0.5, delta = 0.8,initial_belief_strength=0.5):

        # environment parameters worker has access to
        self.benchmark_states = benchmark_states
        self.benchmark_actions = benchmark_actions # share or not share
        self.negotiation_states = negotiation_states
        self.wages = W

        # Q-learning params
        self.alpha = alpha
        self.delta = delta

        # dictionary of Q vals
        benchmark_q_vals = dict({f"({s},{a})": 0 for s in self.benchmark_states for a in self.benchmark_actions}) # initial q-values
        if type_conditioning:
            negotiation_q_vals = dict({f"({s},{(a,b)})": 0 for s in self.negotiation_states for a in s for b in s})
        else:
            negotiation_q_vals = dict({f"({s},{a})": 0 for s in self.negotiation_states for a in s}) # initial q-values [w for w in self.wages if w>=s[0] and w<=s[2]] +(float(1),) if we need more competition across all settings
        self.Q_vals = benchmark_q_vals | negotiation_q_vals # merge dictionaries
    
        # firm characteristics
        self.state = None
        self.acceptance_threshold = None
        self.range = None
        self.benchmark = None
        self.profit = None
        self.workers = [] # starting employees
        self.bot_10 = None
        self.bot_50 = None
        self.bot_90 = None
        self.type_conditioning = type_conditioning
    
    def get_individual_range(self):
        '''get individal wage distribution data'''
        
        return (self.bot_10, self.bot_90)
    
    def get_individual_median(self):
        '''get individual median of wage distribution'''

        return self.bot_50

    def update_individual_wage_distribution(self):
        '''update information firm has about their waeg distribution. If no workers, then use last available information.'''

        if len([w.wage for w in self.workers]) == 0:
            return 

        bot_50 = np.percentile([w.wage for w in self.workers],50)
        bot_50_mapped = min(self.wages, key = lambda x:abs(x-bot_50))

        bot_10 = np.percentile([w.wage for w in self.workers],10)
        bot_90 = np.percentile([w.wage for w in self.workers],90)

        bot_10_mapped =  min(self.wages, key = lambda x:abs(x-bot_10))
        bot_90_mapped =  min(self.wages, key = lambda x:abs(x-bot_90)) 

        self.bot_10 = float(bot_10_mapped)
        self.bot_50 = float(bot_50_mapped)
        self.bot_90 = float(bot_90_mapped)


    def get_reward(self,accepted_offers=[], num_offers=0, benchmark = None, eps_share=1e-2,eps_hire=1):
        ''' some reward for sharing, negotiation reward larger for 1) the more offers accepted and 2) the larger the value of the offers accepted
        sharing is None, True, or False. eps_hire is supposed to penalize firms if they get no offers whatsoever (hm but this really only makes sense if firm's strategy influences supply of workers)
        '''
        if benchmark is not None:
            if benchmark == "salary benchmark":
                return eps_share
            else:
                return 0
        else:
            if num_offers>0:
                reward = (len(accepted_offers) - sum(accepted_offers) - (num_offers - len(accepted_offers)))/num_offers 
                # if reward==1 and max(accepted_offers)>0:
                #     print(accepted_offers)
                #     print(num_offers)
                #     exit()
            else:
                reward = 0
            self.profit = reward # reward = that round's average profit 
            return reward

    def action_decision(self, benchmark = False, explore_eps=None):

        explore = gen.binomial(1,explore_eps) # decide to explore or exploit

        if not benchmark:
            if self.type_conditioning:
                negotiation_actions = [(a,b) for a  in self.state for b in self.state] #[w for w in self.wages if w >= self.state[0] and w <= self.state[2]]#   # bring back for more competition
            else:
                negotiation_actions = self.state 
       

        if explore: # choose an action U.A.R.
            if benchmark:
                if len(self.state) == 2:
                    action = gen.choice(self.benchmark_actions[:2],1)[0]
                else:
                    action = gen.choice(self.benchmark_actions,1)[0] 
            else:
                action = gen.choice(negotiation_actions,1)[0]
        else:
            if benchmark:
                if len(self.state) == 2:
                    action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.benchmark_actions[:2]])
                else:
                    action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.benchmark_actions])
                action = self.benchmark_actions[action_index]
            else:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in negotiation_actions])
                action = negotiation_actions[action_index]

        return action
    
    def update_q_vals(self, new_state, action, benchmark = False, accepted_offers = [], num_offers = 0):

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if benchmark:
            reward = self.get_reward(benchmark=action)
            if self.type_conditioning:
                negotiation_actions = [(a,b) for a in new_state for b in new_state] #[w for w in self.wages if w >= self.state[0] and w <= self.state[2]]#   # bring back for more competition
            else:
                negotiation_actions = new_state#[w for w in self.wages if w >= new_state[0] and w <= new_state[2]]##+(float(1),)# if needed
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in negotiation_actions]))
        else:
            reward = self.get_reward(accepted_offers=accepted_offers,num_offers=num_offers)
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.benchmark_actions]))



if __name__ == "__main__":


    # parameters
    N_w = 100 # small number of workers to start
    N_f = 5 # small number of firms
    k = 5 # number of intervals to break [0,1] up into
    W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
    ranges = W + [-1] # -1 indicates no range given 
    p_risky = 0# indicates probability of being a risky worker. 0 => symmetric, neutral worker case
    type_conditioning = False
    p_reset = 0.5 # probability of resetting wage, "riskiness" of market, robustness of this
    beta = 6.91*10**(-4) # slow 6.91*10**(-4), medium 9.21*10**(-4), fast 1.15*10**(-3)

    betas = [6.91*10**(-4),9.21*10**(-4),1.15*10**(-3)]
    beta_labels = ["slow", "medium","fast"]

    # setting flags
    use_mvpt = False
    posting = False
    mixed_posts = False

    settings = [(False,False, False),(False, True, False),(True, True, False), (True, True, True)] #
    setting_label = ["setting 1", "setting 2","setting 3", "setting 4"] #"setting 2",

    save = True
    folder = "experiment_1_test_3"

    # find values to fix these at to compare with previous work
    alpha = 0.3 # more weight on present rewards
    delta = 0.9 # more patient
    initial_belief_strength = 0.05

    T = 20000 # seems to be enough time to see "stable" behavior at the end 
    # T_negotiation = 1000 # tolerance for number of time steps with  no negotiations (what is necessary here?)

    ## initial distributions
    # skewed left (k-l):
    kl = [3/4,1/8,0.9-(3/4+1/8)]+[0.1/(k-2) for i in range(k-2)]
    # slightly skewed left (s-k-l):
    skl = [1/2,1/8,0.7-(1/2+1/8)]+[0.3/(k-2) for i in range(k-2)]
    # uniform (u): 
    u= [1/(k+1) for i in range(k+1)]
    # slightly skewed right (s-k-r):
    skr = [0.3/(k-2) for i in range(k-2)]+[1/2,1/8,0.7-(1/2+1/8)]
    # skewed right(k-r):
    kr = [0.1/(k-2) for i in range(k-2)]+[0.9-(3/4+1/8),1/8,3/4]
    # bimodal even (b-e):
    be = [0.4] + [0.1/2 for i in range(2)] + [0.1/2 for i in range(2)] + [0.4]
    # bimodal slightly left (b-l):
    bl =  [0.45] + [0.1/2 for i in range(2)]  + [0.1/2 for i in range(2)] + [0.35] 
    # bimodal slightly right (b-r):
    br =  [0.35] + [0.1/2 for i in range(2)]+ [0.1/2 for i in range(2)] + [0.45]

    p_settings = [kr,skr,skl,kl,u,be,bl,br]
    p_labels = ["k-r","s-k-r","s-k-l","k-l","u", "b-e", "b-l","b-r"] 

    for (use_mvpt, posting, mixed_posts), s_label in zip(settings,setting_label):

        # states and actions
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

        print(f"Information setting: {s_label}")
        print("size of worker state space")
        print(f"|S_w_sharing|= {len(S_w_sharing)}")
        print(f"|S_w_firm_choice|= {len(S_w_firm_choice)}")
        print(f"|S_w_negotiation|= {len(S_w_negotiation)}")

        print("size of firm state space")
        print(f"|S_f_benchmark|= {len(S_f_benchmark)}")
        print(f"|S_f_negotiation|= {len(S_f_negotiation)}")
        for beta, b_label in zip(betas, beta_labels):
            for p_s, p_l in zip(p_settings, p_labels):

                print(f"beta setting:{b_label}, distribution: {p_l}")
                market = Market(N_w,N_f,S_w_sharing,A_w_sharing,S_w_negotiation,A_w_negotiation,S_w_firm_choice, A_w_firm_choice,S_f_benchmark,A_f_benchmark,S_f_negotiation,W,p_s,alpha,delta,initial_belief_strength,p_risky=p_risky,type_conditioning=type_conditioning,p_reset=p_reset,beta=beta,use_mvpt=use_mvpt,posting=posting,mixed_posts=mixed_posts)

                # things to track through the market
                worker_wages = [[w.wage for w in market.workers]]
                if type_conditioning:
                    (initial_c_R,initial_b_R), (initial_c_S,initial_b_S) = get_wage_distribution_market_split_workers(market)
                else:
                    (initial_c, initial_b) = get_wage_distribution_market(market)
                m_hat_values = []
                true_median = []

                firms_ATs = [[]for f in market.firms]
                firms_ranges = [[] for f in market.firms]

                worker_offers = [[] for w in market.workers]
                worker_firm_choice = [[] for w in market.workers]

                firm_profits = [[] for f in market.firms]


                for t in tqdm(range(T)):
                    if use_mvpt:
                        m_hat_values.append(market.mvpt.m_hat)

                    true_median.append(market._get_market_median())

                    market.market_time_step(t)

                    for i,f in enumerate(market.firms):
                        firms_ATs[i].append(f.acceptance_threshold)
                        firms_ranges[i].append(f.range)
                        firm_profits[i].append(f.profit)

                    for i,w in enumerate(market.workers):
                        if w.offer == "no offer":
                            worker_offers[i].append(-1)
                            worker_firm_choice[i].append(-1)
                        else:
                            worker_offers[i].append(w.offer)
                            worker_firm_choice[i].append(w.firm_negotiation_choice)

                    all_wages = [w.wage for w in market.workers]
                    worker_wages.append(all_wages)
                    
                    # old "convergence" checks -- just wait the 20,000 time steps 
                    ## Convergence criteria
                    # if min(all_wages) == 1:
                    #     print("Converged to MRPL!")
                    #     break

                    # if max(all_wages) == 0:
                    #     print("Converged to firm capturing surplus!")
                    #     break
                
                    # if min(all_wages) == max(all_wages):
                    #     print(f"Converged to wage {min(all_wages)}")
                    #     break
                    # # need something like offer == worker's wage, so they don't negotiate (negotiations stop for X time steps or  negotiations stop AND eps<threshold)
                    # if t >= T_negotiation and sum(market.num_negotiations[-T_negotiation:]) == 0:
                    #     print(f"No negotiations for {T_negotiation} time steps")
                    #     break
                



                ### Analyzing

                ## Overall wage analysis 
                # Wage distribution final vs. initial
                plot_attribute_distribution_market(market,"wage",N_w=N_w,N_f=N_f,initial_counts = initial_c,initial_bins =initial_b,split_workers=type_conditioning,k=k,p_l=p_l,s_l = s_label,b_l=b_label,p_reset=p_reset,seed=seed,save=save,folder=folder)

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

                # Tracking possible dispersion
                all_wages_final = [w.wage for w in market.workers]
                all_wages_initial = worker_wages[0]
                bot_10 = np.percentile(all_wages_initial,10)
                bot_50 = np.percentile(all_wages_initial,50)
                bot_90 = np.percentile(all_wages_initial,90)

                bot_10_final = np.percentile(all_wages_final,10)
                bot_90_final = np.percentile(all_wages_final,90)
                bot_50_final = np.percentile(all_wages_final,50)

                print(f"initial percentiles: {bot_10}, {bot_50}, {bot_90}")
                print(f"final percentiles: {bot_10_final}, {bot_50_final}, {bot_90_final}")
                if bot_50 >0:
                    print(f"initial dispersion ratio: {(bot_90-bot_10)/bot_50}")
                if bot_50_final >0:
                    print(f"final dispersion ratio: {(bot_90_final-bot_10_final)/bot_50_final}")

                

                ## Firm analysis
                # firm profit in last 1000 time steps  
                tau = int(T/2)
                for i in range(N_f):
                    plt.plot(range(T-tau,T), firm_profits[i][T-tau:])
                    plt.title(f"Firm {i} average profits")
                    # plt.ylim((-1,1)) # normalize?
                    plt.xlabel(f"Last {tau} timesteps")
                    plt.ylabel("Firm profit")
                    if save:
                        plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_firm_{i}_profits_seed={seed}.png")
                    plt.clf()

                
                # information use numbers
                sal_bench_numbers = [info_use[0] for info_use in market.firm_information_use]
                independent_numbers = [info_use[1] for info_use in market.firm_information_use]
    
                plt.plot(range(len(market.firm_information_use)), sal_bench_numbers,label="Num. firms using Salary Benchmark")
                plt.plot(range(len(market.firm_information_use)), independent_numbers,label="Num. firms using Independent data")
                if use_mvpt:
                    mvpt_numbers = [N_f - iu_s - iu_i for iu_s,iu_i in zip(sal_bench_numbers,independent_numbers)]
                    plt.plot(range(len(market.firm_information_use)), mvpt_numbers,label="Num. firms using MVPT data")
                plt.title("Firm information use over time")
                plt.ylim((0,N_f))
                plt.xlabel("Time")
                plt.ylabel("Firm counts per source")
                plt.legend()
                if save:
                    plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_information_source_numbers_seed={seed}.png")
                plt.clf()

                # information source values
                # sal_bench_l = [info[0][1] for info in market.firm_information_source_values]
                # sal_bench_u = [info[0][1] for info in market.firm_information_source_values]
                # ind_range_l = [[]for i in range(N_f)]
                # ind_range_u = [[]for i in range(N_f)]
                # mvpt_range_l = [[]for i in range(N_f)]
                # mvpt_range_u = [[]for i in range(N_f)]
                # for t in range(len(market.firm_information_source_values)):
                #     for i in range(N_f):
                #         if use_mvpt:
                #             ind_range_l[i].append(market.firm_information_source_values[t][2*i+1][0])
                #             ind_range_u[i].append(market.firm_information_source_values[t][2*i+1][1])
                #             mvpt_range_l[i].append(market.firm_information_source_values[t][2*i+2][0])
                #             mvpt_range_u[i].append(market.firm_information_source_values[t][2*i+2][1])
                #         else:
                #             ind_range_l[i].append(market.firm_information_source_values[t][i+1][0])
                #             ind_range_u[i].append(market.firm_information_source_values[t][i+1][1])
                
                
                # find "stability" time to limit size of graph
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
                for i in range(len(market.firms)):
                    if type_conditioning:
                        plt.plot(range(stable_t), risky_ATs[i][:stable_t],color ="red",label="acceptance threshold for risky")
                        plt.plot(range(stable_t), safe_ATs[i][:stable_t],color ="blue",label="acceptance threshold for safe")
                    else:
                        plt.plot(range(stable_t), firms_ATs[i][:stable_t],color ="purple",label="acceptance threshold")
                        lower_bound = [r[0] for r in firms_ranges[i][:stable_t]]
                        upper_bound = [r[1] for r in firms_ranges[i][:stable_t]]
                        plt.plot(range(stable_t), lower_bound,color ="red",label="lower bound of range")
                        plt.plot(range(stable_t), upper_bound,color ="blue",label="upper bound of range")
                    plt.ylim((0,1))
                    plt.title(f"Firm index {i} acceptance thresholds over time")
                    plt.legend()
                    if save:
                        plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_firm_{i}_seed={seed}.png")
                    plt.clf()


                ## Worker analysis 
                # average worker surplus over last tau time steps
                avg_worker_surplus = []
                for t in range(T-tau,T):
                    avg_worker_surplus.append((1/N_w) * sum(worker_wages[t]))
                plt.plot(range(T-tau,T), avg_worker_surplus)
                plt.title("Average Worker Surplus")
                plt.ylim((0,1))
                plt.xlabel(f"Last {tau} timesteps")
                plt.ylabel("Worker Surplus")
                if save:
                    plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_average_worker_surplus_seed={seed}.png")
                plt.clf()




                # wage gap if type conditioning 
                if type_conditioning:
                    avg_worker_wage_risky = []
                    avg_worker_wage_safe = []

                    firm_set_risky_x = []
                    firm_set_risky_y = []
                    firm_set_safe_x = []
                    firm_set_safe_y = []

                    for t in range(T-tau,T):
                        worker_wages_risky = []
                        worker_wages_safe = []

                        chosen_firms_risky = set()
                        chosen_firms_safe = set()

                        for idx, w in enumerate(market.workers):
                            if w.type=="risky":
                                worker_wages_risky.append(worker_wages[t][idx])
                                if worker_firm_choice[idx][t] != -1:
                                    chosen_firms_risky.add(worker_firm_choice[idx][t])
                            else:
                                worker_wages_safe.append(worker_wages[t][idx])
                                if worker_firm_choice[idx][t] != -1:
                                    chosen_firms_safe.add(worker_firm_choice[idx][t])
                        avg_worker_wage_risky.append(np.mean(worker_wages_risky))
                        avg_worker_wage_safe.append(np.mean(worker_wages_safe))
                        for c_f in chosen_firms_risky:
                            firm_set_risky_x.append(t)
                            firm_set_risky_y.append(c_f)
                        for c_f in chosen_firms_safe:
                            firm_set_safe_x.append(t)
                            firm_set_safe_y.append(c_f)
                    
                    # plot wages
                    plt.plot(range(T-tau,T), avg_worker_wage_risky,label="Avg. Risky Worker Wages")
                    plt.plot(range(T-tau,T), avg_worker_wage_safe,label="Avg. Safe Worker Wages")
                    plt.title("Worker Wage Comparison")
                    plt.ylim((0,1))
                    plt.legend()
                    plt.xlabel(f"Last {tau} timesteps")
                    plt.ylabel("Average Worker Wage")
                    if save:
                        plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_worker_wage_comparison_seed={seed}.png")
                    plt.clf()

                    # plot firm choices
                    plt.scatter(firm_set_risky_x,firm_set_risky_y, label="Risky Workers",marker="o",color="red")
                    plt.scatter(firm_set_safe_x,firm_set_safe_y, label="Safe Workers",marker="x",color="blue")
                    plt.title("Worker Firm Negotiation Choices")
                    plt.ylim((0,4))
                    plt.legend()
                    plt.xlabel(f"Last {tau} timesteps")
                    plt.ylabel("Chosen Firms")
                    if save:
                        plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_worker_firm_choices_seed={seed}.png")
                    plt.clf()






                # mvpt use numbers over time
                if use_mvpt:
                    plt.plot(range(len(market.worker_mvpt_use)), market.worker_mvpt_use)
                    plt.ylim((0,N_w))
                    plt.title("Number of workers sharing with MVPT over time")
                    if save:
                        plt.savefig(f"{folder}/p={p_reset}_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_{s_label}_beta={b_label}_mvpt_sharing_numbers_seed={seed}.png")
                    plt.clf()

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
                
                plt.close() # close all figures between runs
                print() # line between runs


                # TODO -- further analysis? more on how workers use information?


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
