'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 115
gen = np.random.default_rng(seed=seed)

update_T = 0 # only if needed


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

    def __init__(self, N_w, N_f, s_w_sharing, a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,s_f_benchmark,a_f_benchmark, s_f_negotiation,W,p,alpha,delta,initial_belief_strength, use_mvpt = False, posting = False, mixed_posts = False):
        '''initializaiton of the market

        Parameters
            N_w: number of workers
        '''
        # setting parameters
        self.use_mvpt = use_mvpt 
        self.post_ranges = posting
        self.mixed_posts = mixed_posts # mvpt estimated and firm provided ranges
        self.wages = W   

        # explore epsilon
        self.beta = 2.76 * 10**(-5)
        self.explore_epsilon = 1 # start at 1 

        # create Workers
        print("Initializing workers...")
        self.workers = [Worker(s_w_sharing,a_w_sharing,s_w_negotiation,a_w_negotiation,s_w_firm_choice, a_w_firm_choice,gen.choice(self.wages,p=p),alpha,delta,initial_belief_strength) for i in range(N_w)] # workers identical up to initial wage so far
        print("Initializing firms...")
        self.firms = [Firm(s_f_benchmark,a_f_benchmark, s_f_negotiation,self.wages,alpha,delta,initial_belief_strength) for i in range(N_f)]  ## assuming firms have enough capacity for all workers in our market and all have constant returns to scale from the set of workers that apply 
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
        self.num_firms_using_sal_bench = []
    
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
        using_bench = 0
        for idx,f in enumerate(self.firms):
            f.benchmark = f.action_decision(explore_eps=self.explore_eps,benchmark=True)
            
            if f.benchmark == "salary benchmark":
                self.salary_benchmark.add_firm(f) # for next round
                l_b, u_b= self.salary_benchmark.get_benchmark_range()
                next_state = (l_b, self.salary_benchmark.get_benchmark_median(), u_b)
                f.update_q_vals(next_state,f.benchmark,benchmark=True)
                f.state = next_state
                using_bench = using_bench + 1
            else:
                self.salary_benchmark.remove_firm(f)
                if f.benchmark == "independent":
                    l_f, u_f = f.get_individual_range()
                    next_state = (l_f, f.get_individual_median(),u_f)
                else:
                    l_m, u_m = self.mvpt.get_mvpt_range(idx)
                    next_state = (l_m, f.get_individual_median(),u_m)
                f.update_q_vals(next_state,f.benchmark,benchmark=True)
                f.state = next_state

        self.num_firms_using_sal_bench.append(using_bench)

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

                if t>0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt)  
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state
            
        else: # no posted ranges, workers have no information 
            for w in self.workers:
                next_state = (-1,-1,-1,-1,-1) # state shows that worker knows nothing before choosing a firm  
        
                if t>0: # need to update q vals from negotiation outcomes after the first time step
                    w.update_q_vals(next_state,w.offer,bargaining_outcome=w.bargaining_outcome,last_wage = w.last_wage,using_mvpt=self.use_mvpt)
                    w.bargaining_outcome = None # reset bargaining outcome since it is only set conditional on offer != no offer
                w.state = next_state


        # worker sharing decision if using MVPT 
        if self.use_mvpt:
            for w in self.workers:
                w.share = w.action_decision(explore_eps = self.explore_eps,sharing=True)
                
                if w.share == "share":
                    self.mvpt.add_worker(w) # add to data pool for next round
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
                w.update_q_vals(next_state,w.share,sharing=True)
                w.state = next_state
    
    def negotiation_strategy_choice(self,t):
        '''
        '''
        # worker firm choice and offer decision
        for w in self.workers:
            # choose a firm to negotiate with based on range of posted wages (+ assumption that -1 => no knowledge or u<=res wage)
            w.firm_negotiation_choice = int(w.action_decision(firm_choice = True,explore_eps=self.explore_eps))

            # determine next state
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
            
            next_state = (offer_low, offer_high)
            w.update_q_vals(next_state, w.firm_negotiation_choice, firm_choice=True)
            w.state = next_state

            # offer decision
            w.offer = w.action_decision(explore_eps=self.explore_eps)
            if w.offer != "no offer": # only if you make an offer do you get to move on
                w.offer = float(w.offer) # cast to float now
                self.negotiation_matches[w.firm_negotiation_choice].append(w)
        
        
        # firm acceptance threshold decision    
        for f in self.firms:
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
            for w in self.negotiation_matches[f_idx]: # loop through each worker initiating negotiation
                num_negotiations = num_negotiations + 1 # tracking

                w.bargaining_outcome = _bargaining_outcome(w.offer, f.acceptance_threshold)
                w.last_wage = w.wage # track wage before updating

                if w.bargaining_outcome < 0: # if negotiation fails, some small probability that worker's wage gets reset.
                    p_reset_wage = 0.5 # robustness of this parameter?
                    reset_wage = gen.binomial(1, p_reset_wage)
                    
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
        self.salary_benchmark.update_benchmark()

        # rewards and state transitions for firms and workers, self.salary_benchmark.get_gov_data() 
        for idx, f in enumerate(self.firms): # all firms have a negotiation update
            f.update_individual_wage_distribution()
            if self.use_mvpt and -1 not in self.mvpt.get_mvpt_range(idx):
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range(),self.mvpt.get_mvpt_range(idx))
            else:
                next_state = (f.get_individual_range(),self.salary_benchmark.get_benchmark_range())
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

    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, firm_choice_states,firm_choice_actions, initial_wage, alpha = 0.5, delta = 0.8,initial_belief_strength=0.5):

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
                

    def get_reward(self, sharing = None, firm_choice = False, bargaining_outcome = None, last_wage=None, eps_share=1e-2):
        '''sharing is None, True or False. bargaining_outcome is None, -1 or >0
        '''
        if sharing is not None:
            if not sharing:
                return 0
            else:
                return eps_share
        elif firm_choice:
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
    
    def update_q_vals(self, new_state, action, sharing = False, firm_choice=False, bargaining_outcome = None,last_wage=None,using_mvpt=False):
        # update step: Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))

        ## get rewards
        if sharing:
            if action == "share":
                reward = self.get_reward(sharing=True)
            else:
                reward = self.get_reward(sharing=False)
        elif firm_choice:
            reward = self.get_reward(firm_choice = True)
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
    def __init__(self, benchmark_states, benchmark_actions, negotiation_states, W, alpha = 0.5, delta = 0.8,initial_belief_strength=0.5):

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
        negotiation_q_vals = dict({f"({s},{a})": 0 for s in self.negotiation_states for a in s}) # initial q-values +(float(1),) if we need more competition across all settings
        self.Q_vals = benchmark_q_vals | negotiation_q_vals # merge dictionaries
    
        # firm characteristics
        self.state = None
        self.acceptance_threshold = None
        self.benchmark = None
        self.workers = [] # starting employees
        self.bot_10 = None
        self.bot_50 = None
        self.bot_90 = None
    
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
            AT_reward = len(accepted_offers) - sum(accepted_offers) - (num_offers  - len(accepted_offers))
            return AT_reward

    def action_decision(self, benchmark = False, explore_eps=None):

        explore = gen.binomial(1,explore_eps) # decide to explore or exploit
        
        negotiation_actions = self.state # +(float(1),) bring back for more competition

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
            negotiation_actions = new_state # +(float(1),) if needed
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

    # setting flags
    use_mvpt = True
    posting = True
    mixed_posts = False

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
    if MVPT:
        S_f_benchmark = S_f_benchmark + [((l_b,u_b),(l_i,u_i),(l_m,u_m)) for l_b in W for u_b in W for l_i in W  for u_i in W  for l_m in W for u_m in W if l_b<= u_b and l_i <= u_i and l_m <= u_m] # salary benchmark data + individual data + MVPT data (probably too many states)
        A_f_benchmark = A_f_benchmark + ["mvpt"] 

    print("size of worker state space")
    print(f"|S_w_sharing|= {len(S_w_sharing)}")
    print(f"|S_w_firm_choice|= {len(S_w_firm_choice)}")
    print(f"|S_w_negotiation|= {len(S_w_negotiation)}")

    print("size of firm state space")
    print(f"|S_f_benchmark|= {len(S_f_benchmark)}")
    print(f"|S_f_negotiation|= {len(S_f_negotiation)}")

    # find values to fix these at to compare with previous work
    alpha = 0.3 # more weight on present rewards
    delta = 0.9 # more patient
    initial_belief_strength = 0.05

    T = 500000 # consistent with previous simulations TODO --  early stopping criteria?
    T_negotiation = 1000 # tolerance for number of time steps with  no negotiations (what is necessary here?)


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

    p_settings = [kr,skr,skl,u,be,bl,br]
    p_labels = ["k-r","s-k-r","s-k-l","u", "b-e", "b-l","b-r"] 
    

    save = False

    for p_s, p_l in zip(p_settings, p_labels):

        print(f"distribution: {p_l}")
        market = Market(N_w,N_f,S_w_sharing,A_w_sharing,S_w_negotiation,A_w_negotiation,S_w_firm_choice, A_w_firm_choice,S_f_benchmark,A_f_benchmark,S_f_negotiation,W,p_s,alpha,delta,initial_belief_strength,use_mvpt=use_mvpt,posting=posting,mixed_posts=mixed_posts)

        # things to track through the market
        worker_wages = [[w.wage for w in market.workers]]
        initial_counts_bins_market = get_wage_distribution_market(market)
        m_hat_values = []
        # l_hat_values = []
        # u_hat_values = []
        true_median = []

        firms_ATs = [[]for f in market.firms]
        worker_offers = [[] for w in market.workers]


        for t in tqdm(range(T)):
            # m_hat_values.append(market.mvpt.m_hat)
            # l_hat_values.append(market.mvpt.l_hat)
            # u_hat_values.append(market.mvpt.u_hat)
            true_median.append(market._get_market_median())

            market.market_time_step(t)
            # print(market.workers[0].explore_eps)
            # print(market.firms[0].explore_eps)

            for i,f in enumerate(market.firms):
                firms_ATs[i].append(f.acceptance_threshold)
            for i,w in enumerate(market.workers):

                if w.offer == "no offer":
                    worker_offers[i].append(-1)
                    # print("no offer")
                elif w.offer == None:
                    # print("didn't share")
                    worker_offers[i].append(-2)
                else:
                    worker_offers[i].append(w.offer)


            # simple convergence check
            all_wages = [w.wage for w in market.workers]
            worker_wages.append(all_wages)
            
            ## Convergence criteria
            if min(all_wages) == 1:
                print("Converged to MRPL!")
                break

            if max(all_wages) == 0:
                print("Converged to firm capturing surplus!")
                break
            # need something like offer == worker's wage, so they don't negotiate (negotiations stop for X time steps or  negotiations stop AND eps<threshold)
            if t >= T_negotiation and sum(market.num_negotiations[-T_negotiation:]) == 0:
                print(f"No negotiations for {T_negotiation} time steps")
                break
        



        ## analyzing, TODO -- clean up analysis.
        # print(market.workers[0].explore_eps)
        # print(market.firms[0].explore_eps)
        print(f"Final salary benchmark pool size: {len(market.salary_benchmark.data_pool)}")
        # print(f"Final MVPT pool size: {len(market.mvpt.data_pool)}")

        # for i in range(len(market.firms)):
        #     print(f"Firm {i} non-zero final Q values")
        #     for n in S_negotiation_f:
        #         nonzero_Qs = [(n,a,market.firms[i].Q_vals[f"({n},{a})"]) for a in n +(float(1),) if a>=0 and market.firms[i].Q_vals[f"({n},{a})"] !=0]
        #         if len(nonzero_Qs)>0:
        #             print(nonzero_Qs)
        #     plt.plot(range(len(firms_ATs[i])),firms_ATs[i])
        #     plt.ylim((0,1))
        #     plt.title(f"Firm index {i} acceptance threshold over time")
        #     plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_firm_{i}_seed={seed}.png")
        #     plt.clf()

        # tau = min(1000, len(worker_offers[0]))
        # for i in range(0,100,20):
        #     plt.plot(range(tau),worker_offers[i][-tau:])
        #     plt.ylim((-2,1))
        #     plt.title(f"Worker index {i} offer over last {tau} time steps. Final employer: {market.workers[i].employer}")
        #     plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_worker_{i}_seed={seed}.png")
        #     plt.clf()

        # # plt.plot(range(len(market.num_bad_offers)), market.num_bad_offers)
        # # plt.title("Number of bad offers made over time")
        # # plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_bad_offers_numbers_seed={seed}.png")
        # # plt.clf()

        # plt.plot(range(len(market.num_firms_using_sal_bench)), market.num_firms_using_sal_bench)
        # plt.title("Number of firms using salary benchmark over time")
        # plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_sal_benchmark_numbers_seed={seed}.png")
        # plt.clf()



        # # correlating initial wage to final wage
        # all_wages_final = [w.wage for w in market.workers]
        # all_wages_initial = worker_wages[0]
        # bot_10 = np.percentile(all_wages_initial,10)
        # bot_25 = np.percentile(all_wages_initial,25)
        # bot_50 = np.percentile(all_wages_initial,50)
        # bot_75 = np.percentile(all_wages_initial,75)
        # bot_90 = np.percentile(all_wages_initial,90)

        # bot_10_final = np.percentile(all_wages_final,10)
        # bot_90_final = np.percentile(all_wages_final,90)
        # bot_50_final = np.percentile(all_wages_final,50)

        # print(f"initial percentiles: {bot_10}, {bot_50}, {bot_90}")
        # print(f"final percentiles: {bot_10_final}, {bot_50_final}, {bot_90_final}")

        # print(f"initial dispersion ratio: {(bot_90-bot_10)/bot_50}")
        # print(f"final dispersion ratio: {(bot_90_final-bot_10_final)/bot_50_final}")


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
        time_window = len(true_median)# min(5000,max(len(true_median)-1000,0))
        # plt.plot(m_hat_values[time_window:], label="m_hat value (median + error)")
        # plt.plot(l_hat_values[time_window:], color="red",label="Min value in data pool")
        # plt.plot(u_hat_values[time_window:], color="purple",label="Max value in data pool")
        plt.plot(true_median[:time_window], color="orange",linestyle="dashed",label="True median")
        plt.ylim((0,1))
        plt.title(f"true median values over first {time_window} steps")
        plt.xlabel("Time")
        plt.ylabel("true median values")
        plt.legend()
        if save:
            plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p_l}_mvpt_values_seed={seed}.png")
        plt.show()

        # plot_attribute_distribution_market(market,"wage",N_w=N_w,N_f=N_f,extra_counts = initial_counts_bins_market[0],extra_bins =initial_counts_bins_market[1],k=k,p=p_l,seed=seed,save=save)

        
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


        # # worker_time_window = 50000#int(max([t[0] for t in t_last_share]))+100 # 100 time steps after the last share action
        # # # print(worker_time_window)

        # # for i in range(0,500,10):
        # #     worker_k_wages = [w[i] for w in worker_wages[:worker_time_window]]
        # #     worker_k_sharing_q_val = [w[i] for w in worker_share_given_wage[:worker_time_window]]
        # #     worker_k_no_sharing_q_val = [w[i] for w in worker_no_share_given_wage[:worker_time_window]]
        # #     plt.plot(worker_k_wages, label="Wage", linestyle="solid",color="purple")
        # #     plt.plot(worker_k_sharing_q_val, label="Q(state,share)",linestyle="dashed",color="blue")
        # #     plt.plot(worker_k_no_sharing_q_val, label="Q(state,no share)", linestyle="dashed",color="red")
        # #     plt.legend(title="Worker characteristics")
        # #     plt.xlabel("Time")
        # #     plt.ylabel("Worker wage")
        # #     plt.title(f"worker index {i} wage over time and action q values")
        # #     # plt.ylim((0,1))
        # #     if save:
        # #         plt.savefig(f"q_learning_simulation_results/p=0.1_N={N}_k={k}_initial_distribution={p_l}_worker_{i}_wages_seed={seed}.png")
        # #     plt.clf()






