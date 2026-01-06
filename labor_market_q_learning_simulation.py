'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 42
gen = np.random.default_rng(seed=seed)

class MVPT:

    def __init__(self, worker_data_pool,W, N_f):
        '''initialization of market value preictor tool (MVPT). all workers can see 10th and 90th percentile estimates for each firm. sharing wrokers see m_hat (median estimate for whole market).
        
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
        '''adds specified Worker object to data pool.

        params
            worker (Worker): Worker object to add to data set. 
        '''
        if worker.employer == None:
            return # can't share data if you aren't employed!
        self.data_pool[worker.employer].add(worker) # track wage for their current employer

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool.

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
    '''salary benchmarking tool that firms have access to. Reports 10th and 90th percentile of all sharing firm's data to all firms. Sharing firms get 50th percentile as well. '''

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

    def __init__(self, N_w, N_f, s_w_sharing, a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,s_f_benchmark,a_f_benchmark, s_f_negotiation,W,p,alpha,delta,p_risky = 0,type_conditioning=False, p_reset= 0.5, beta = 6.91*10**(-4),use_mvpt = False, posting = False, mixed_posts = False,evaluate=False):
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

        self.eval = evaluate

        # explore epsilon
        self.beta = beta
        self.explore_eps = 1 # start at 1 

        # create Workers
        print("Initializing workers...")
        if p_risky >0:
            self.workers = [Worker(s_w_sharing,a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,gen.choice(self.wages,p=p),gen.choice(["risky","safe"],p=[p_risky,1-p_risky]),alpha,delta) for i in range(N_w)] # workers risky or safe
        else:
            self.workers = [Worker(s_w_sharing,a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,gen.choice(self.wages,p=p),"symmetric",alpha,delta) for i in range(N_w)] # symmetric up to initial wage
        print("Initializing firms...")
        self.firms = [Firm(s_f_benchmark,a_f_benchmark, s_f_negotiation,self.wages,type_conditioning,alpha,delta) for i in range(N_f)]  ## assuming firms have enough capacity for all workers in our market and all have constant returns to scale from the set of workers that apply 
        worker_groups = np.split(np.array(self.workers), N_f) # randomly and evenly split workers among firms to start
        for i in range(N_f):
            for j in range(len(worker_groups[i])):
                self.firms[i].workers.append(worker_groups[i][j]) # connect worker
                worker_groups[i][j].employer = i # connect to employer

        # for tracking matches for negotiation
        self.negotiation_matches = [[] for i in range(N_f)] # a set of workers associated with the firm they will attempt to negotiate with

        # ensure at least one worker starts with MRPL
        expert_worker_idx = gen.integers(0,N_w)
        self.workers[expert_worker_idx].wage = float(1) 

        # CORRELATION: bottom 1-p_risky percentile safe and p_risky percent risky
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
        # parameter update
        if self.eval==True:
            self.explore_eps = 0
        elif type(self.beta) == tuple:
            self.explore_eps = (1*np.e**(-self.beta[0] * t),1*np.e**(-self.beta[1] * t)) # (risky,safe) explore epsilon values
        else:
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

            if f.deviating == False or t > 0:
                f.benchmark = f.action_decision(explore_eps=self.explore_eps,benchmark=True)
            ranges = ranges + (f.state[0],)
            if self.use_mvpt: 
                if len(f.state)==3: 
                    ranges = ranges + (f.state[2],)
                else:
                    ranges = ranges + ((-1,-1),)
            
            if f.benchmark == "salary benchmark":
                self.salary_benchmark.add_firm(f) # for next round
                l_b, u_b= self.salary_benchmark.get_benchmark_range()
                next_state = (l_b, self.salary_benchmark.get_benchmark_median(), u_b)
                f.update_q_vals(next_state,f.benchmark,benchmark=True,evaluate=self.eval)
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
                f.update_q_vals(next_state,f.benchmark,benchmark=True,evaluate=self.eval)
                f.state = next_state

        self.firm_information_use.append((using_bench,using_independent))
        self.firm_information_source_values.append(ranges)

        # set worker's state with posted ranges 
        if self.post_ranges:
            min_val = 1.0
            max_val = 0.0
            for idx, f in enumerate(self.firms):
                if f.deviating ==True and t==0:
                    print(f"deviating firm posted range {f.state}")
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

                if t>0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt,evaluate=self.eval)  
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state
            
        else: # no posted ranges, workers have no information 
            for w in self.workers:
                next_state = (-1,-1,-1,-1,-1) # state shows that worker knows nothing before choosing a firm  
        
                if t>0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt,evaluate=self.eval)
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state


        # worker sharing decision if using MVPT 
        if self.use_mvpt:
            mvpt_share = 0
            for w in self.workers:
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
                w.update_q_vals(next_state,w.share,sharing=True,evaluate=self.eval)
                w.state = next_state
             #tracking   
            self.worker_mvpt_use.append(mvpt_share)
    
    def negotiation_strategy_choice(self,t):
        '''
        '''
        # worker firm choice and offer decision
        for w in self.workers:
            # choose a firm to negotiate with based on range of posted wages (+ assumption that -1 => no knowledge or u<=res wage)
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
            elif w.type == "risky" or w.type == "safe":
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
            else:
                print(f"Error: unrecognized worker type '{w.type}'.")
            
            next_state = (offer_low, offer_high)
            if self.post_ranges:
                w.update_q_vals(next_state, w.firm_negotiation_choice, firm_choice=True,firm_upper_bound=self.posted_ranges[w.firm_negotiation_choice][1],evaluate=self.eval)
            else:
                w.update_q_vals(next_state, w.firm_negotiation_choice, firm_choice=True,firm_upper_bound=-1,evaluate=self.eval)
            w.state = next_state

            # offer decision
            w.offer = w.action_decision(explore_eps=self.explore_eps)
            if w.offer != "no offer": # only if you make an offer do you get to move on
                w.offer = float(w.offer) # cast to float now
                self.negotiation_matches[w.firm_negotiation_choice].append(w)
        
        
        # firm acceptance threshold decision   
        for f in self.firms:
            if f.deviating == True and t == 0:
                continue

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
        if self.use_mvpt:
            self.mvpt.update_mvpt() 

        num_negotiations = 0
        accepted_offers = [[] for f in self.firms]

        for f_idx,f in enumerate(self.firms):
            num_offers = len(self.negotiation_matches[f_idx])
            # set firm acceptance threshold
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
                        w.wage = float(0) # their "leisure wage" or some notion of reservation utility
                        if w.employer != None:
                            self.firms[w.employer].workers.remove(w) 
                        w.employer = None # they are unemployed so firms cannot see 0 as a valid wage in this case
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
        self.salary_benchmark.update_benchmark()

        # rewards and state transitions for firms and workers, self.salary_benchmark.get_gov_data() 
        for idx, f in enumerate(self.firms): # all firms have a negotiation update
            f.update_individual_wage_distribution()
            if self.use_mvpt and -1 not in self.mvpt.get_mvpt_range(idx):
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range(),self.mvpt.get_mvpt_range(idx))
            else:
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range())
            f.update_q_vals(next_state,f.acceptance_threshold, accepted_offers = accepted_offers[idx], num_offers = len(self.negotiation_matches[idx]),evaluate=self.eval)
            f.state = next_state
    
        self.num_negotiations.append(num_negotiations) # tracking
        self.negotiation_matches = [[] for i in range(len(self.firms))] # reset negotitation matches for next round


class Worker:

    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, firm_choice_states,firm_choice_actions, initial_wage, worker_type, alpha = 0.5, delta = 0.8):

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

        if type(explore_eps) == tuple:
            if self.type == "risky":
                explore = gen.binomial(1,explore_eps[0]) # decide to explore or exploit
            else:
                explore = gen.binomial(1,explore_eps[1]) # decide to explore or exploit
        else:
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
    
    def update_q_vals(self, new_state, action, sharing = False, firm_choice=False, bargaining_outcome = None,last_wage=None,using_mvpt=False,firm_upper_bound =None,evaluate=False):
        # update step: Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if evaluate == True: # make sure no updating happens 
            return 

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
    def __init__(self, benchmark_states, benchmark_actions, negotiation_states, W, type_conditioning, alpha = 0.5, delta = 0.8):

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
        self.benchmark = None
        self.profit = None
        self.workers = [] # starting employees
        self.bot_10 = None
        self.bot_50 = None
        self.bot_90 = None
        self.type_conditioning = type_conditioning
        self.deviating = False
    
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
            else:
                reward = 0
            self.profit = reward # reward = that round's average profit 
            return reward

    def action_decision(self, benchmark = False, explore_eps=None):

        if type(explore_eps) == tuple and len(explore_eps) == 2:
            explore = gen.binomial(1,explore_eps[0]) # decide to explore or exploit
        else:
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
    
    def update_q_vals(self, new_state, action, benchmark = False, accepted_offers = [], num_offers = 0,evaluate=False):
        

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if benchmark:
            reward = self.get_reward(benchmark=action)
            if self.type_conditioning:
                negotiation_actions = [(a,b) for a in new_state for b in new_state] #[w for w in self.wages if w >= self.state[0] and w <= self.state[2]]#   # bring back for more competition
            else:
                negotiation_actions = new_state#[w for w in self.wages if w >= new_state[0] and w <= new_state[2]]##+(float(1),)# if needed
            if evaluate == True: # make sure no updating happens 
                return 
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in negotiation_actions]))
        else:
            reward = self.get_reward(accepted_offers=accepted_offers,num_offers=num_offers)
            if evaluate == True: # make sure no updating happens 
                return 
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.benchmark_actions]))



if __name__ == "__main__":


    # parameters
    N_w = 100 # small number of workers to start
    N_f = 5 # small number of firms
    k = 5 # number of intervals to break [0,1] up into
    W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
    ranges = W + [-1] # -1 indicates no range given 
    p_risky = 0# indicates probability of being a risky worker. 0 => symmetric, neutral worker case
    type_conditioning = True
    p_reset = 0.5 # probability of resetting wage, "riskiness" of market, robustness of this
    beta = 6.91*10**(-4) # slow 6.91*10**(-4), medium 9.21*10**(-4), fast 1.15*10**(-3)

    betas = [(6.91*10**(-4),1.15*10**(-3))]#9.21*10**(-4)(6.91*10**(-4),1.15*10**(-3))
    beta_labels = ["slow-fast"]#, "medium","fast"]

    settings = [(False,False, False),(False, True, False),(True, True, False), (True, True, True)] #
    setting_label = ["setting 1", "setting 2","setting 3","setting 4"] # 

    riskiness = [(0.5,0.5)]#,(0,0.5)
    riskiness_label = ["e-r"]#,"n-r"

    save = False
    # folder = "experiment_2_test_6"
    s_folder = "simulation_output_robustness_tests"

    # find values to fix these at to compare with previous work (also need to robustness test these)
    alpha = 0.3 # more weight on present rewards
    delta = 0.9 # more patient

    T = 500000#20000 # seems to be enough time to see "stable" behavior at the end 

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


    N = 10 # trials to run to average over

    setting_average_worker_surplus = [[[[] for p in range(8)] for s in range(4)] for r in range(2)]
    setting_wage_gap = [[[[] for p in range(8)] for s in range(4)] for r in range(2)]

    for n in range(N):
        print(f"RUN: {n}/{N-1}")
        for r, (p_risky, p_reset), r_label in zip(range(1),riskiness, riskiness_label):
            for s, (use_mvpt, posting, mixed_posts), s_label in zip(range(4),settings,setting_label):
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

                for beta, b_label in zip(betas, beta_labels):
                    for p, p_s, p_l in zip(range(8),p_settings, p_labels):
                        print("--Setting Details--")
                        print(f"information setting: {s_label}, market setting: {r_label}, beta setting:{b_label}, distribution: {p_l}")
                        market = Market(N_w,N_f,S_w_sharing,A_w_sharing,S_w_negotiation,A_w_negotiation,S_w_firm_choice, A_w_firm_choice,S_f_benchmark,A_f_benchmark,S_f_negotiation,W,p_s,alpha,delta,p_risky=p_risky,type_conditioning=type_conditioning,p_reset=p_reset,beta=beta,use_mvpt=use_mvpt,posting=posting,mixed_posts=mixed_posts)

                        # things to track through the market
                        if type_conditioning:
                            (initial_c_R,initial_b_R), (initial_c_S,initial_b_S) = get_wage_distribution_market_split_workers(market)
                        
                            worker_offers_R = [[] for w in market.workers if w.type == "risky"]
                            worker_firm_choice_R = [[] for w in market.workers if w.type == "risky"]
                            worker_wages_R = [[w.wage] for w in market.workers if w.type == "risky"]
                            worker_sharing_R = [[] for w in market.workers if w.type == "risky"]
                            
                            worker_offers_S = [[] for w in market.workers if w.type == "safe"]
                            worker_firm_choice_S = [[] for w in market.workers if w.type == "safe"]
                            worker_wages_S = [[w.wage] for w in market.workers if w.type == "safe"]
                            worker_sharing_S = [[] for w in market.workers if w.type == "safe"]
                        else:
                            (initial_c, initial_b) = get_wage_distribution_market(market)

                            worker_offers = [[] for w in market.workers]
                            worker_firm_choice = [[] for w in market.workers]
                            worker_wages = [[w.wage] for w in market.workers]
                            worker_sharing = [[] for w in market.workers]
                        
                        if use_mvpt:
                            m_hat_values = []
                        true_median = []

                        firms_ATs = [[]for f in market.firms]
                        firms_bm = [[] for f in market.firms]
                        firm_profits = [[] for f in market.firms]


                        for t in tqdm(range(T)):
                            # track median info
                            if use_mvpt:
                                m_hat_values.append(market.mvpt.m_hat)
                            true_median.append(market._get_market_median())

                            # market time step
                            market.market_time_step(t)

                            # track firm info
                            for i,f in enumerate(market.firms):
                                firms_ATs[i].append(f.acceptance_threshold)
                                firm_profits[i].append(f.profit)
                                if f.benchmark == "independent":
                                    firms_bm[i].append(0)
                                elif f.benchmark == "salary benchmark":
                                    firms_bm[i].append(1)
                                elif f.benchmark == "mvpt":
                                    firms_bm[i].append(2)
                                else:
                                    print(f"ERROR: unrecognized benchmark {f.benchmark}")
                                    exit()

                            # track worker info
                            if type_conditioning:
                                idx_S = 0
                                idx_R = 0
                                for w in market.workers:
                                    if w.type=="risky":
                                        worker_wages_R[idx_R].append(w.wage)
                                        if w.offer == "no offer":
                                            worker_offers_R[idx_R].append(-1)
                                            worker_firm_choice_R[idx_R].append(-1)
                                        else:
                                            worker_offers_R[idx_R].append(w.offer)
                                            worker_firm_choice_R[idx_R].append(w.firm_negotiation_choice)
                                        if use_mvpt:
                                            if w.share == "share":
                                                worker_sharing_R[idx_R].append(1)
                                            else:
                                                worker_sharing_R[idx_R].append(0)
                                        idx_R = idx_R + 1 # next index
                                    else:
                                        worker_wages_S[idx_S].append(w.wage)
                                        if w.offer == "no offer":
                                            worker_offers_S[idx_S].append(-1)
                                            worker_firm_choice_S[idx_S].append(-1)
                                        else:
                                            worker_offers_S[idx_S].append(w.offer)
                                            worker_firm_choice_S[idx_S].append(w.firm_negotiation_choice)
                                        if use_mvpt:
                                            if w.share == "share":
                                                worker_sharing_S[idx_S].append(1)
                                            else:
                                                worker_sharing_S[idx_S].append(0)
                                        idx_S = idx_S + 1 # next index
                            else:
                                for i,w in enumerate(market.workers):
                                    worker_wages[i].append(w.wage)
                                    if w.offer == "no offer":
                                        worker_offers[i].append(-1)
                                        worker_firm_choice[i].append(-1)
                                    else:
                                        worker_offers[i].append(w.offer)
                                        worker_firm_choice[i].append(w.firm_negotiation_choice)
                                    if w.share == "share":
                                        worker_sharing[i].append(1)
                                    else:
                                        worker_sharing[i].append(0)

                        # Tracking possible dispersion
                        all_wages_final = [w.wage for w in market.workers]
                       
                        # information source values
                        sal_bench_l = [info[0][0] for info in market.firm_information_source_values]
                        sal_bench_u = [info[0][1] for info in market.firm_information_source_values]
                        ind_range_l = [[]for i in range(N_f)]
                        ind_range_u = [[]for i in range(N_f)]
                        if use_mvpt:
                            mvpt_range_l = [[]for i in range(N_f)]
                            mvpt_range_u = [[]for i in range(N_f)]
                        for t in range(len(market.firm_information_source_values)):
                            for i in range(N_f):
                                if use_mvpt:
                                    ind_range_l[i].append(market.firm_information_source_values[t][2*i+1][0])
                                    ind_range_u[i].append(market.firm_information_source_values[t][2*i+1][1])
                                    mvpt_range_l[i].append(market.firm_information_source_values[t][2*i+2][0])
                                    mvpt_range_u[i].append(market.firm_information_source_values[t][2*i+2][1])
                                else:
                                    ind_range_l[i].append(market.firm_information_source_values[t][i+1][0])
                                    ind_range_u[i].append(market.firm_information_source_values[t][i+1][1])
                        
                        
                        
                        setting_average_worker_surplus[r][s][p].append((1/N_w) * sum(all_wages_final))



                        # wage gap if type conditioning 
                        if type_conditioning:
                                                        

                            worker_wages_R_T = [wage[T] for wage in worker_wages_R] # final wage of all risky workers
                            worker_wages_S_T = [wage[T] for wage in worker_wages_S] # final wage of all safe workers
                            
                            avg_wage_risky = (1/len(worker_wages_R_T)) * sum(worker_wages_R_T)
                            avg_wage_safe = (1/len(worker_wages_S_T)) * sum(worker_wages_S_T)
                            setting_wage_gap[r][s][p].append(avg_wage_risky - avg_wage_safe)
                        
                        
                        ### Saving Data
                        print("Saving data...")
                        # create folder
                        run_folder = f"N_w={N_w}_N_f={N_f}_k={k}_{s_label}_initial_dist={p_l}_beta={b_label}_risk={r_label}_seed={seed}_N={n}"
                        Path(f"{s_folder}/{run_folder}").mkdir(parents=True,exist_ok=True)

                        # dump data
                        with open(f"{s_folder}/{run_folder}/median_values.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/sal_bench_l.pkl", 'wb') as pkl_fl2, open(f"{s_folder}/{run_folder}/sal_bench_u.pkl", 'wb') as pkl_fl3:
                            pickle.dump(true_median,file = pkl_fl1)
                            pickle.dump(sal_bench_l, file = pkl_fl2)
                            pickle.dump(sal_bench_u, file = pkl_fl3)
                        if use_mvpt:
                            with open(f"{s_folder}/{run_folder}/mvpt_values.pkl",'wb') as pkl_fl:
                                pickle.dump(m_hat_values,file = pkl_fl)
                        
                        for idx_f in range(len(market.firms)):
                            with open(f"{s_folder}/{run_folder}/firm_{idx_f}_ATs.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/firm_{idx_f}_profits.pkl", 'wb') as pkl_fl2, open(f"{s_folder}/{run_folder}/firm_{idx_f}_benchmarks.pkl", 'wb') as pkl_fl3:
                                pickle.dump(firms_ATs[idx_f], file = pkl_fl1)
                                pickle.dump(firm_profits[idx_f], file = pkl_fl2)
                                pickle.dump(firms_bm[idx_f], file = pkl_fl3)
                            with open(f"{s_folder}/{run_folder}/ind_bench_l_{idx_f}.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/ind_bench_u_{idx_f}.pkl", 'wb') as pkl_fl2:# open(f"{s_folder}/{run_folder}/q_vals_firm_{idx_f}.pkl", 'wb') as pkl_fl3:
                                pickle.dump(ind_range_l[idx_f], file = pkl_fl1)
                                pickle.dump(ind_range_u[idx_f], file = pkl_fl2)
                                # pickle.dump(market.firms[idx_f].Q_vals, file=pkl_fl3) don't save q-vals for now!
                            if use_mvpt:
                                with open(f"{s_folder}/{run_folder}/mvpt_bench_l_{idx_f}.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/mvpt_bench_u_{idx_f}.pkl", 'wb') as pkl_fl2:
                                    pickle.dump(mvpt_range_l[idx_f], file = pkl_fl1)
                                    pickle.dump(mvpt_range_u[idx_f], file = pkl_fl2)
                            
                            
                        # don't save q-vals until I have a good test for them
                        # for idx_w,w in enumerate(market.workers):
                        #     if w.type == "risky":
                        #         with open(f"{s_folder}/{run_folder}/q_vals_worker_{idx_w}_R.pkl",'wb') as pkl_fl:
                        #             pickle.dump(w.Q_vals, file=pkl_fl)
                        #     elif w.type == "safe":
                        #         with open(f"{s_folder}/{run_folder}/q_vals_worker_{idx_w}_S.pkl",'wb') as pkl_fl:
                        #             pickle.dump(w.Q_vals, file=pkl_fl)
                        #     else:
                        #         with open(f"{s_folder}/{run_folder}/q_vals_worker_{idx_w}.pkl",'wb') as pkl_fl:
                        #             pickle.dump(w.Q_vals, file=pkl_fl)

                        if type_conditioning:
                            for idx_w in range(len(worker_offers_R)):
                                with open(f"{s_folder}/{run_folder}/worker_R_{idx_w}_offer.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/worker_R_{idx_w}_firm_choice.pkl", 'wb') as pkl_fl2, open(f"{s_folder}/{run_folder}/worker_R_{idx_w}_wage.pkl", 'wb') as pkl_fl3, open(f"{s_folder}/{run_folder}/worker_R_{idx_w}_sharing.pkl", 'wb') as pkl_fl4:
                                    pickle.dump(worker_offers_R[idx_w], file=pkl_fl1)
                                    pickle.dump(worker_firm_choice_R[idx_w], file=pkl_fl2)
                                    pickle.dump(worker_wages_R[idx_w], file=pkl_fl3)
                                    pickle.dump(worker_sharing_R[idx_w], file=pkl_fl4)
                            
                            for idx_w in range(len(worker_offers_S)):
                               with open(f"{s_folder}/{run_folder}/worker_S_{idx_w}_offer.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/worker_S_{idx_w}_firm_choice.pkl", 'wb') as pkl_fl2, open(f"{s_folder}/{run_folder}/worker_S_{idx_w}_wage.pkl", 'wb') as pkl_fl3, open(f"{s_folder}/{run_folder}/worker_S_{idx_w}_sharing.pkl", 'wb') as pkl_fl4:
                                    pickle.dump(worker_offers_S[idx_w], file=pkl_fl1)
                                    pickle.dump(worker_firm_choice_S[idx_w], file=pkl_fl2)
                                    pickle.dump(worker_wages_S[idx_w], file=pkl_fl3)
                                    pickle.dump(worker_sharing_S[idx_w], file=pkl_fl4)

                        else:
                            for idx_w in range(len(market.workers)):
                                with open(f"{s_folder}/{run_folder}/worker_{idx_w}_offer.pkl", 'wb') as pkl_fl1, open(f"{s_folder}/{run_folder}/worker_{idx_w}_firm_choice.pkl", 'wb') as pkl_fl2, open(f"{s_folder}/{run_folder}/worker_{idx_w}_wage.pkl", 'wb') as pkl_fl3:
                                    pickle.dump(worker_offers[idx_w], file=pkl_fl1)
                                    pickle.dump(worker_firm_choice[idx_w], file=pkl_fl2)
                                    pickle.dump(worker_wages[idx_w], file=pkl_fl3)

                        print("Data saved.")
                        print() # line between runs

                    
    print("----FINAL RESULTS----")
    with open(f"{s_folder}/average_worker_surplus_seed={seed}",'wb') as pkl_fl1, open(f"{s_folder}/average_wage_gap_seed={seed}",'wb') as pkl_fl2:
        pickle.dump(setting_average_worker_surplus, file=pkl_fl1)
        pickle.dump(setting_wage_gap, file=pkl_fl2)
    # print(f"Setting avg worker surpluses overall: {setting_average_worker_surplus}")


    for r in range(2):
        for s in range(4):
            print(f"riskiness {r}, setting {s+1}")
            print(f"Average worker surplus -- across all distributions {sum([sum(ws) for ws in setting_average_worker_surplus[r][s]])/(N*8)}")
            print(f"Average wage gap -- across all wage distributions {sum([sum(wg) for wg in setting_wage_gap[r][s]])/(N*8)}")
            print("Broken down by initial distribution")
            for p in range(8):
                print(f"Initial dist {p_labels[p]}")
                print(f"Average worker surplus {sum(setting_average_worker_surplus[r][s][p])/N}")
                print(f"Average wage gap {sum(setting_wage_gap[r][s][p])/N}")
                print()