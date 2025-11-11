'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 101
gen = np.random.default_rng(seed=seed)


class MVPT:

    def __init__(self, worker_data_pool,W):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
        '''
        # data / noise parameter for predictions
        self.data_pool = worker_data_pool # to generate median 
        self.wages = W # possible wages to map onto 

        # information tool providees
        self.m_hat = 0
        self.u_hat = 0 # max of data pool (no error)
        self.l_hat = 0 # min of data pool (no error)

    
    def add_worker(self, worker):
        '''adds specified Worker object to data pool. decreases sigma_e. 

        params
            worker (Worker): Worker object to add to data set. 
        '''
        self.data_pool.add(worker)

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e.

        params
            worker (Worker): Worker object to remove from data set. 
        '''
        self.data_pool.discard(worker)

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        if len(self.data_pool) == 0:
            # print("Error, data pool empty")
            return 

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within reasonable bounds of data pool
        noisy_median = np.median([w.wage for w in self.data_pool])
        self.m_hat = min(self.wages, key = lambda x:abs(x-noisy_median))# mapping to closest value in wage list
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)


class SalaryBenchmark:
    '''salary benchmarking tool that firms have access to. They collect coarse salary information, and augment it for sharing firms with firm-reported wage data.    '''

    def __init__(self, firm_data_pool, all_firms, W):
        '''initialization of market value preictor tool (MVPT)
        
        params
            firm_data_pool (Set[Firm]): pool of initial firms who are sharing their wage information
        '''
        # data / noise parameter for predictions
        self.data_pool = firm_data_pool # to generate more granular estimates 
        self.gov_data =  all_firms
        self.wages = W # possible wages to map onto 

        # coarse information tool providees
        # self.m_gov = 0 # median from gov data
        self.u_gov = 0 # upper bound from gov
        self.l_gov = 0 # lower bound from gov

        # granular information  provided by the tool (could be different from gov. benchmark)
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
    
    def get_gov_data(self):
        return (float(self.l_gov), float(self.u_gov))
        #float(self.m_gov),
    
    def get_firm_shared_data(self):
        return (float(self.data_pool_10),float(self.data_pool_50),float(self.data_pool_90))

    def update_benchmarks(self): 
        '''updates gov data benchmark and firm data pool benchmark
        '''
        if len(self.data_pool) == 0:
            # print("Error, data pool empty")
            return 
        elif len([w for f in self.data_pool for w in f.workers]) == 0: # none of the participating firms have workers
            return 

        # update gov benchmark
        self.u_gov = np.max([w.wage for f in self.gov_data for w in f.workers])
        self.l_gov = np.min([w.wage for f in self.gov_data for w in f.workers])
        # self.m_gov = np.percentile([w.wage  for f in self.gov_data for w in f.workers], 50)
        # self.m_gov = min(self.wages, key = lambda x:abs(x-self.m_gov))# mapping to closest value in wage list

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

    def __init__(self, N_w, N_f, s_sharing_w, s_negotiation_w, s_sharing_f,s_negotiation_f,a_sharing, a_negotiation_w,a_negotiation_f,W,p,alpha,delta,explore_eps,initial_belief_strength, information_setting = "MVPT"):
        '''initializaiton of the market

        Parameters
            N_w: number of workers
        '''

        # create Workers
        print("Initializing workers...")
        self.workers = [Worker(s_sharing_w,a_sharing,s_negotiation_w,a_negotiation_w,gen.choice(W,p=p),alpha,explore_eps,delta,initial_belief_strength) for i in range(N_w)] # workers identical up to initial wage so far
        print("Initializing firms...")
        self.firms = [Firm(s_sharing_f,a_sharing, s_negotiation_f,a_negotiation_f,alpha,explore_eps,delta,initial_belief_strength) for i in range(N_f)]  ## assuming firms have enough capacity for all workers in our market and all have constant returns to scale from the set of workers that apply 
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

        ## initializing information setting of the market
        # initializing mvpt after workers have an initial wage
        if information_setting == "MVPT":
            initial_worker_pool = set()
            for w in self.workers:
                if gen.binomial(1,0.5) >= 0.5:
                    initial_worker_pool.add(w) # initial pool generated with adding in workers uniform iid
            
            self.mvpt = MVPT(initial_worker_pool,W) # initial worker pool is random
            self.mvpt.update_mvpt()
        # initializing word-of-mouth parameters 
        elif information_setting == "Word-of-mouth":
            # TODO -- experiment 2
            pass
        # initializing firm-posting parameters
        else:
            # TODO -- experiment 3
            pass

        # initializing salary benchmark
        initial_firm_pool = set()
        for f in self.firms:
            if gen.binomial(1,0.5) >= 0.5: # fair coin toss to be in initial pool
                initial_firm_pool.add(f)
            
        self.salary_benchmark = SalaryBenchmark(initial_firm_pool, self.firms, W)
        self.salary_benchmark.update_benchmarks()


        for w in self.workers: # set initial state of all workers
            first_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
            w.state = first_state 
        
        gov_data_state = self.salary_benchmark.get_gov_data()
        for f in self.firms:
            f_median = float(min(W, key=lambda x:abs(x-np.median([w.wage for w in f.workers])))) # make sure it stays in W
            f.state = gov_data_state + (f_median,) # gov data plus true median of firm's worker set
        
        # tracking market behavior for output graphs
        self.num_negotiations = []
    
    def market_time_step(self):
        '''In a market time step, agents first seek information, then perform negotiations and update their beliefs, finally mvpt is updated for next round based on the data pool gathered this round

        Parameters
            T (int): total time in the simulation run, used in expectation calculations 
            t (int): current time step, used in expectation calculations
        '''
        # parameter reset
        for w in self.workers:
            w.offer = None # reset offer 
        for f in self.firms:
            f.acceptance_threshold = None # reset acceptance threshold

        # phase 1 - information seeking
        self.coarse_information_and_sharing_choice()  # firms and workers decide to share their private info

        # phase 2 - setting negotiation strategy
        self.negotiation_strategy_choice()

        # phase 3 - negotiation and belief updates    
        self.negotiation_and_state_transition()

    
    
    def _get_market_median(self):
        '''compute true median in market. I think this is only for tracking purposes now
        '''
        # compute summary statistics
        all_wages = []
        for w in self.workers:
            all_wages.append(w.wage)
        
        return min(W, key=lambda x:abs(x-np.median(all_wages)))

    def coarse_information_and_sharing_choice(self):
        '''
        '''

        # worker seeking
        for w in self.workers:
            sharing_decision = w.action_decision(sharing=True) # choose decions eps-greedily
            if sharing_decision == "share":
                self.mvpt.add_worker(w) # add to data pool for next round
                next_state = (float(w.wage), float(self.mvpt.l_hat), float(self.mvpt.m_hat),float(self.mvpt.u_hat))
                w.update_q_vals(next_state,sharing_decision,sharing=True)
                w.state = next_state
            else:
                self.mvpt.remove_worker(w) # remove worker from data pool for next round
                # delay reward until we know state after negotiations 
        
        # fim seeking 
        for f in self.firms:
            sharing_decision = f.action_decision(sharing=True)
            if sharing_decision == "share":
                self.salary_benchmark.add_firm(f) # for next round
                next_state = self.salary_benchmark.get_firm_shared_data() + (f.state[-1],) # add in pooled data (from who shared last round, but current wages)
                f.update_q_vals(next_state,sharing_decision,sharing=True)
                f.state = next_state
            else:
                self.salary_benchmark.remove_firm(f) 
                next_state = f.state + (-1,)
                f.update_q_vals(next_state,sharing_decision,sharing=True)
                f.state = next_state
    
    def negotiation_strategy_choice(self):
        '''
        '''
        # worker offer decision
        for w in self.workers:
            if len(w.state) == 4: # you only get more information if you share
                w.offer = w.action_decision()
                if w.offer != "no offer": # only if you make an offer do you get to move on
                    w.offer = float(w.offer) # cast to float for now
                    firm_idx = gen.choice(range(len(self.firms)),1)[0] # uniformly at random
                    self.negotiation_matches[firm_idx].append(w)
        
        # firm acceptance threshold decision
        for f in self.firms:
            f.acceptance_threshold = float(f.action_decision()) # have to make an acceptance threshold decision no matter what
    

    def negotiation_and_state_transition(self):
        '''
        '''
        def _bargaining_outcome(offer, acceptance_threshold):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(at1), counter offer cap (at2),  counter offer (o2), worker counter offer acceptance threshold (at1_c)'''
            if offer <= acceptance_threshold: # opening offer lower than firm's acceptance threshold, accept.
                return offer
            else: # opening offer too high, reject
                return -1

        # update mvpt based on workers' current wages, signals ready for next state
        self.mvpt.update_mvpt() 

        num_negotiations = 0
        accepted_offers = [[] for f in self.firms]
        for f_idx,f in enumerate(self.firms):
            num_offers = len(self.negotiation_matches[f_idx])
            for w in self.negotiation_matches[f_idx]: # loop through each worker initiating negotiation
                num_negotiations = num_negotiations + 1 # tracking

                outcome = _bargaining_outcome(w.offer, f.acceptance_threshold)

                if outcome < 0: # if negotiation fails, some small probability that worker's wage gets reset.
                    p_reset_wage = 0.01 # robustness of this parameter?
                    reset_wage = gen.binomial(1, p_reset_wage)
                    
                    if reset_wage:
                        w.wage = float(0) # their "leisure wage" or some notion of reservation utility -- should they still be employed?
                        if w.employer != None:
                            self.firms[w.employer].workers.remove(w) # they are unemployed
                        w.employer = None

                # reward and state transition -- sharing + offering workers covered here
                next_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
                w.update_q_vals(next_state,w.offer,bargaining_outcome=outcome)  
                w.state = next_state
                
                # updating market info
                if outcome >= 0:
                    w.wage = float(outcome) # new wage acheived
                    accepted_offers[f_idx].append(outcome)

                    # re-employ worker
                    if w.employer != None:
                        self.firms[w.employer].workers.remove(w)
                    w.employer = f_idx
                    self.firms[f_idx].workers.append(w)

        
        # update benchmarks based on new employment data
        self.salary_benchmark.update_benchmarks()

        # rewards and state transitions for firms and workers
        for f_idx, f in enumerate(self.firms): # all firms have a negotiation update
            new_median = float(min(W, key=lambda x:abs(x-np.median([w.wage for w in f.workers])))) # make sure it stays in W
            next_state = self.salary_benchmark.get_gov_data() + (new_median,)
            f.update_q_vals(next_state,f.acceptance_threshold, accepted_offers = accepted_offers[f_idx], num_offers = len(self.negotiation_matches[f_idx]))
            f.state = next_state

            # f.acceptance_threshold = None # reset acceptance threshold
    
        for w in self.workers:
            next_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
            if len(w.state) == 3 and w.offer == None: # didn't share
                w.update_q_vals(next_state,"no share", sharing = True)
                w.state = next_state # update worker's new state
            elif  w.offer == "no offer": # shared, but didn't offer
                w.update_q_vals(next_state,w.offer)  
                w.state = next_state # update worker's new state
            
            # w.offer = None # reset offer 
        
    
        self.num_negotiations.append(num_negotiations) # tracking
        self.negotiation_matches = [[] for i in range(N_f)] # reset negotitation matches for next round


class Worker:

    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, initial_wage, alpha = 0.5, explore_eps = 0.2, delta = 0.8,initial_belief_strength=0.5):

        # environment parameters worker has access to
        self.share_states = share_states
        self.share_actions = share_actions # share or not share
        self.negotiation_states = negotiation_states
        self.negotiation_actions = negotiation_actions # share or not share

        self.alpha = alpha
        self.explore_eps = explore_eps
        self.delta = delta

        # dictionary of Q vals
        share_q_vals = dict({f"({s},{a})": 0 for s in self.share_states for a in self.share_actions}) # initial q-values
        negotiation_q_vals = dict({f"({s},{a})": 0 for s in self.negotiation_states for a in self.negotiation_actions}) # initial q-values
        self.Q_vals = share_q_vals | negotiation_q_vals# merge dictionaries

        # worker characteristics
        self.state = None
        self.offer = None
        self.wage = initial_wage
        self.employer = None

         # initial belief strength on sharing (possibly consider later)
        # for w,l,u in self.share_states:
        #     if w < u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = initial_belief_strength # upweight sharing if your wage is strictly below upper bound
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = -1*initial_belief_strength # downweight no sharing if your wage is strictly below upper bound
        #     elif w >= u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = -1*initial_belief_strength # downweight sharing if your wage is at least the upper bound of the range
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = initial_belief_strength # upweight sharing if your wage is at least the upper bound of the range
        
        for w,l,m,u in self.negotiation_states:
            if w > m:
                self.Q_vals[f"({(w,l,m,u)},{'no offer'})"] = initial_belief_strength  # prefer not to offer when wage is above median

            for o in self.negotiation_actions:
                if o == "no offer":
                    continue
                if o > w:
                    self.Q_vals[f"({(w,l,m,u)},{o})"] = initial_belief_strength # prefer to make larger offers
                else:
                    self.Q_vals[f"({(w,l,m,u)},{o})"] = -1*initial_belief_strength # prefer to not make smaller offers


    def get_reward(self, sharing = None, bargaining_outcome = None, eps_share=1e-2):
        '''sharing is None, True or False. bargaining_outcome is None, -1 or >0
        '''
        if sharing is not None:
            if not sharing:
                return 0
            else:
                return eps_share
        else:
            if bargaining_outcome == None: # some benefit to sharing w/ no negotiation
                return 0
            elif bargaining_outcome > 0: # only if successful
                return bargaining_outcome - self.wage
            else:
                return bargaining_outcome

    def action_decision(self, sharing = False, beta = 1.38*10**(-5)):

        explore = gen.binomial(1,self.explore_eps) # decide to explore or exploit

        if explore: # choose an action U.A.R.
            if sharing:
                action = gen.choice(self.share_actions,1)[0] 
            else:
                action = gen.choice(self.negotiation_actions,1)[0]
        else:
            if sharing:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.share_actions])
                action = self.share_actions[action_index]
            else:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.negotiation_actions])
                action = self.negotiation_actions[action_index]
        
        # discount explore_eps
        self.explore_eps = self.explore_eps * np.e**(-beta)

        return action
    
    def update_q_vals(self, new_state, action, sharing = False, bargaining_outcome = None):
        # update step: Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))

        ## get rewards
        if sharing:
            if action == "share":
                reward = self.get_reward(sharing=True)
            else:
                reward = self.get_reward(sharing=False)
        else:
            reward = self.get_reward(bargaining_outcome)
        
        ## q-value update
        if len(new_state) == 3: # indicates moving to a sharing state
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.share_actions]))
        else: # else, moving to a negotiation state
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.negotiation_actions]))

    
class Firm:
    '''implementation of firms in this market who are now also q-learning their negotiation strategy under different information conditions.'''
    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, alpha = 0.5, explore_eps = 0.2, delta = 0.8,initial_belief_strength=0.5):

        # environment parameters worker has access to
        self.share_states = share_states
        self.share_actions = share_actions # share or not share
        self.negotiation_states = negotiation_states
        # self.negotiation_actions = negotiation_actions # acceptance thresholds -- now just use one of the values 

        self.alpha = alpha
        self.explore_eps = explore_eps
        self.delta = delta

        # dictionary of Q vals
        share_q_vals = dict({f"({s},{a})": 0 for s in self.share_states for a in self.share_actions}) # initial q-values
        negotiation_q_vals = dict({f"({s},{a})": 0 for s in self.negotiation_states for a in s  if a >= 0}) # initial q-values
        self.Q_vals = share_q_vals | negotiation_q_vals # merge dictionaries
    
        # firm characteristics
        self.state = None
        self.acceptance_threshold = None
        self.workers = [] # starting employees


    def get_reward(self,accepted_offers=[], num_offers=0, sharing = None, eps_share=1e-2,eps_hire=1):
        ''' some reward for sharing, negotiation reward larger for 1) the more offers accepted and 2) the larger the value of the offers accepted
        sharing is None, True, or False. eps_hire is supposed to penalize firms if they get no offers whatsoever (hm but this really only makes sense if firm's strategy influences supply of workers)
        '''
        if sharing is not None:
            if not sharing:
                return 0
            else:
                return eps_share
        else:
            AT_reward = len(accepted_offers) - sum(accepted_offers) - (num_offers  - len(accepted_offers))
            return AT_reward

    def action_decision(self, sharing = False, beta = 1.38*10**(-5)):

        explore = gen.binomial(1,self.explore_eps) # decide to explore or exploit

        if explore: # choose an action U.A.R.
            if sharing:
                action = gen.choice(self.share_actions,1)[0] 
            else:
                if -1 in self.state: # don't choose -1 
                    action = gen.choice(self.state[:-1],1)[0]
                else:
                    action = gen.choice(self.state,1)[0]
        else:
            if sharing:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.share_actions])
                action = self.share_actions[action_index]
            else:
                action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.state if a >= 0])
                action = self.state[action_index]
        
        # discount explore_eps
        self.explore_eps = self.explore_eps * np.e**(-beta)

        return action
    
    def update_q_vals(self, new_state, action, sharing = False, accepted_offers = [], num_offers = 0):

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if sharing:
            if action == "share":
                reward = self.get_reward(sharing=True)
            else:
                reward = self.get_reward(sharing=False)
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in new_state if a >= 0]))
        else:
            reward = self.get_reward(accepted_offers=accepted_offers,num_offers=num_offers)
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.share_actions]))



if __name__ == "__main__":


    # parameters
    N_w = 100 # small number of workers to start
    N_f = 5 # small number of firms
    k = 8 # number of intervals to break [0,1] up into
    W = [float(i/k) for i in range(k+1)] # k + 1 possible wages

    # states and actions
    S_sharing_w = [(w,l,u) for w in W for l in W for u in W if l<= u] 
    S_negotiation_w= [(w,l,m,u) for w in W for l in W for m in W for u in W if l <= m and m <= u]
    S_sharing_f = [(l_g,u_g, m_f) for l_g in W for u_g in W for m_f in W if l_g<= u_g] # very coarse data + own data
    S_negotiation_f = [(l_g,u_g,m_f,-1) for l_g in W for u_g in W for m_f in W if l_g<= u_g] + [(dp_10,dp_50,dp_90,m_f) for m_f in W for dp_10 in W for dp_50 in W for dp_90 in W if dp_10<= dp_50 and dp_50 <= dp_90] # use the granular benchmark (or your own median) if you choose it
    print(f"number of firm negotiation states: {len(S_negotiation_f)}")
    print(f"number of worker negotiation states: {len(S_negotiation_w)}")
    A_sharing = ["share", "no share"] # action set for each worker and firm deciding whether to share wage info. Worker shares current salary, firm "plugs in" HR software
    A_negotiation_w = ["no offer"] + W
    A_negotiation_f = W

    # find values to fix these at to compare with previous work
    alpha = 0.3 # more weight on present rewards
    delta = 0.9 # more patient
    explore_eps = 1 # discounts over time
    initial_belief_strength = 1

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
    be = [0.4] + [0.1/2 for i in range(2)] + [0 for i in range(3)] + [0.1/2 for i in range(2)] + [0.4]
    # bimodal slightly left (b-l):
    bl =  [0.45] + [0.1/2 for i in range(2)] + [0 for i in range(3)] + [0.1/2 for i in range(2)]   + [0.35] 
    # bimodal slightly right (b-r):
    br =  [0.35] + [0.1/2 for i in range(2)]+ [0 for i in range(3)] + [0.1/2 for i in range(2)] + [0.45]

    p_settings = [kr,skr,u,be, bl,br]
    p_labels = ["k-r","s-k-r","u", "b-e", "b-l","b-r"] 
    

    save = False

    for p_s, p_l in zip(p_settings, p_labels):

        print(f"distribution: {p_l}")

        market = Market(N_w,N_f,S_sharing_w,S_negotiation_w, S_sharing_f, S_negotiation_f,A_sharing,A_negotiation_w, A_negotiation_f,W,p_s,alpha,delta,explore_eps,initial_belief_strength)

        # things to track through the market
        worker_wages = [[w.wage for w in market.workers]]
        initial_counts_bins_market = get_wage_distribution_market(market)
        m_hat_values = []
        l_hat_values = []
        u_hat_values = []
        true_median = []

        firms_ATs = [[]for f in market.firms]
        worker_offers = [[] for w in market.workers]


        for t in tqdm(range(T)):
            m_hat_values.append(market.mvpt.m_hat)
            l_hat_values.append(market.mvpt.l_hat)
            u_hat_values.append(market.mvpt.u_hat)
            true_median.append(market._get_market_median())

            market.market_time_step()
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
            # need something like offer == worker's wage, so they don't negotiate (negotiations stop for X time steps or  negotiations stop AND eps<threshold)
            if t >= T_negotiation and sum(market.num_negotiations[-T_negotiation:]) == 0:
                print(f"No negotiations for {T_negotiation} time steps")
                break
        



        ## analyzing, TODO -- clean up analysis.

        print(f"Final salary benchmark pool size: {len(market.salary_benchmark.data_pool)}")
        print(f"Final MVPT pool size: {len(market.mvpt.data_pool)}")

        for i in range(len(market.firms)):
            plt.plot(range(len(firms_ATs[i])),firms_ATs[i])
            plt.ylim((0,1))
            plt.title(f"Firm index {i} acceptance threshold over time")
            plt.show()

        for i in range(0,100,5):
            plt.plot(range(100),worker_offers[i][-100:])
            plt.ylim((0,1))
            plt.title(f"Worker index {i} offer over last 100 time steps. Final employer: {market.workers[i].employer}")
            plt.show()




        # correlating initial wage to final wage
        all_wages_final = [w.wage for w in market.workers]
        all_wages_initial = worker_wages[0]
        bot_10 = np.percentile(all_wages_initial,10)
        bot_25 = np.percentile(all_wages_initial,25)
        bot_50 = np.percentile(all_wages_initial,50)
        bot_75 = np.percentile(all_wages_initial,75)
        bot_90 = np.percentile(all_wages_initial,90)

        bot_10_final = np.percentile(all_wages_final,10)
        bot_90_final = np.percentile(all_wages_final,90)
        bot_50_final = np.percentile(all_wages_final,50)

        print(f"initial percentiles: {bot_10}, {bot_50}, {bot_90}")
        print(f"final percentiles: {bot_10_final}, {bot_50_final}, {bot_90_final}")

        print(f"initial dispersion ratio: {(bot_90-bot_10)/bot_50}")
        print(f"final dispersion ratio: {(bot_90_final-bot_10_final)/bot_50_final}")


        if min(all_wages_final) < 1:
            print("All workers DID NOT end up at MRPL.")
            all_wages_less_than_1 = [w for w in all_wages_final if w <1]
            all_wages_eq_1 = [w for w in all_wages_final if w ==1]
            
            all_wages_less_than_b10 = [w for w in all_wages_initial if w <=bot_10]
            all_wages_less_than_b25 = [w for w in all_wages_initial if w <=bot_25]
            all_wages_less_than_b50 = [w for w in all_wages_initial if w <=bot_50]
            all_wages_less_than_b75 = [w for w in all_wages_initial if w <=bot_75]
            all_wages_less_than_b90 = [w for w in all_wages_initial if w <=bot_90 and w > bot_75]

            wages_at_i = [[i for i in range(len(all_wages_initial)) if all_wages_initial[i] == w] for w in W]
            wages_at_i_final_eq_1 = [[i for i in range(len(all_wages_final)) if all_wages_final[i] == 1 and all_wages_initial[i] == w] for w in W]
            wages_at_i_final_l_1 = [[i for i in range(len(all_wages_final)) if all_wages_final[i] < 1 and all_wages_initial[i] == w] for w in W]

            for i in range(len(W)):
                if len(wages_at_i[i])>0:
                    print(f"P(w_f <1 | w_i = {W[i]}) = {len(wages_at_i_final_l_1[i])/len(wages_at_i[i])}")
                    print(f"P(w_f ==1 | w_i = {W[i]}) = {len(wages_at_i_final_eq_1[i])/len(wages_at_i[i])}")
                else:
                    print(f"No initial wages at {W[i]}.")


            # all_wages_less_than_1_and_initial_bot_10_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_10]
            # all_wages_less_than_1_and_initial_bot_25_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_25]
            # all_wages_less_than_1_and_initial_bot_50_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_50] 
            # all_wages_less_than_1_and_initial_bot_75_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_75] 
            # all_wages_less_than_1_and_initial_bot_90_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]<= bot_90 and all_wages_initial[i] > bot_75] 
            # all_wages_less_than_1_and_initial_less_than_1 = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] <1 and all_wages_initial[i]< 1] 

            # all_wages_eq_1_and_initial_bot_75_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] ==1 and all_wages_initial[i]<= bot_75]
            # all_wages_eq_1_and_initial_bot_90_p = [all_wages_final[i] for i in range(len(all_wages_final)) if all_wages_final[i] ==1 and all_wages_initial[i]<= bot_90 and all_wages_initial[i] > bot_75] 

            # print(f"10th percentile={bot_10}, P(w_f <1 | w_i<=bot10) = {len(all_wages_less_than_1_and_initial_bot_10_p)/len(all_wages_less_than_b10)}")
            # print(f"25th percentile={bot_25}, P(w_f <1 | w_i<=bot25) = {len(all_wages_less_than_1_and_initial_bot_25_p)/len(all_wages_less_than_b25)}")
            # print(f"50th percentile={bot_50}, P(w_f <1 | w_i<=bot50) = {len(all_wages_less_than_1_and_initial_bot_50_p)/len(all_wages_less_than_b50)}")

            # if len(all_wages_less_than_b75) >0:
            #     print(f"75th percentile={bot_75}, P(w_f <1 | w_i<=bot75) = {len(all_wages_less_than_1_and_initial_bot_75_p)/len(all_wages_less_than_b75)}")
            #     print(f"P(w_f==1| w_i<=bot75) = {len(all_wages_eq_1_and_initial_bot_75_p)/len(all_wages_less_than_b75)}")

            # if len(all_wages_less_than_b90) >0:
            #     print(f"90th percentile={bot_90}, P(w_f <1 | w_i<=bot90) = {len(all_wages_less_than_1_and_initial_bot_90_p)/len(all_wages_less_than_b90)}")
            #     print(f"P(w_f==1| w_i<=bot90) = {len(all_wages_eq_1_and_initial_bot_90_p)/len(all_wages_less_than_b90)}")
        else:
            print("All workers DID end up at MRPL.")

        m_hat_l_true_med = 0
        m_hat_g_true_med = 0
        m_hat_eq_true_med = 0
        t_last_m_l_t = 0
        t_last_m_g_t = 0
        t_last_m_eq_t = 0

        for m_hat,tm,t in zip(m_hat_values,true_median,range(T)):
            if m_hat < tm:
                m_hat_l_true_med  = m_hat_l_true_med  +1
                t_last_m_l_t = t
            elif m_hat == tm:
                m_hat_eq_true_med = m_hat_eq_true_med +1
                t_last_m_eq_t = t
            else:
                m_hat_g_true_med = m_hat_g_true_med +1
                t_last_m_g_t = t
        
        print("M_hat vs. true median statistics")
        print(f"P(m_hat < tm) = {m_hat_l_true_med / T} and time of last instance {t_last_m_l_t}")
        print(f"P(m_hat == tm) = {m_hat_eq_true_med / T} and time of last instance {t_last_m_eq_t}")
        print(f"P(m_hat > tm) = {m_hat_g_true_med / T} and time of last instance {t_last_m_g_t}")


        # graphs
        time_window = T
        plt.plot(m_hat_values[:time_window], label="m_hat value (median + error)")
        plt.plot(l_hat_values[:time_window], color="red",label="Min value in data pool")
        plt.plot(u_hat_values[:time_window], color="purple",label="Max value in data pool")
        plt.plot(true_median[:time_window], color="orange",linestyle="dashed",label="True median")
        plt.ylim((0,1))
        plt.title(f"MVPT summary statistics values over first {time_window} steps")
        plt.xlabel("Time")
        plt.ylabel("MVPT summary statistics values")
        plt.legend()
        if save:
            plt.savefig(f"q_learning_simulation_results/p=0.1_N={N}_k={k}_initial_distribution={p_l}_mvpt_values_seed={seed}.png")
        plt.show()

        plot_attribute_distribution_market(market,"wage",N_w,extra_counts = initial_counts_bins_market[0],extra_bins =initial_counts_bins_market[1],k=k,p=p_l,seed=seed,save=save)

        
        # worker attributes 
        # t_last_share  = []
        # for i in range(N):
        #     workers_actions[i].reverse()
        #     tls = len(workers_actions[i])/2 + (len(workers_actions[i])/2 - workers_actions[i].index(1)-1)

        #     t_last_share.append((tls,worker_wages[0][i],worker_wages[-1][i])) # (tls, initial wage, last wage)
        
        # last_share = max([t[0] for t in t_last_share])

        # ts_w_i_le_bot_75_eq_1 = [t[0] for t in t_last_share if t[1]<=bot_75 and t[2]==1]
        # ts_w_i_le_bot_75_l_1 = [t[0] for t in t_last_share if t[1]<=bot_75 and t[2]<1]
        # ts_w_i_g_bot_75_le_bot_90_eq_1 = [t[0] for t in t_last_share if t[1]>bot_75 and t[1]<=bot_90 and t[2]==1]
        # ts_w_i_g_bot_75_le_bot_90_l_1 = [t[0] for t in t_last_share if t[1]>bot_75 and t[1]<=bot_90 and t[2] <1]
        # ts_w_i_g_bot_90_eq_1 = [t[0] for t in t_last_share if t[1]>bot_90 and t[2] ==1]
        # ts_w_i_g_bot_90_l_1 = [t[0] for t in t_last_share if t[1]>bot_90 and t[2]<1]

        # ts_w_init_eq_i_f_eq_1 = [[t[0] for t in t_last_share if t[1] == w and t[2] == 1] for w in W]
        # ts_w_init_eq_i_f_l_1 = [[t[0] for t in t_last_share if t[1] == w and t[2] <1] for w in W]

        # colors = ["blue", "red", "black", "orange", "pink", "purple","green", "cyan", "grey", "brown","gold"]
        # for i in range(len(W)):
        #     counts_a, bins_a = np.histogram(ts_w_init_eq_i_f_eq_1[i],bins=50,range=(0,last_share))
        #     counts_b, bins_b = np.histogram(ts_w_init_eq_i_f_l_1[i],bins=50,range=(0,last_share))
        #     plt.stairs(counts_a,bins_a,label=f"initial wage = {W[i]} and final wage=1",color=colors[i])
        #     plt.stairs(counts_b,bins_b,label=f"initial wage = {W[i]} and final wage<1",color=colors[i],linestyle="dashed")
        # # counts_1a, bins_1a = np.histogram(ts_w_i_le_bot_75_eq_1,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # counts_1b, bins_1b = np.histogram(ts_w_i_le_bot_75_l_1,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # counts_2a, bins_2a = np.histogram(ts_w_i_g_bot_75_le_bot_90_eq_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # counts_2b, bins_2b = np.histogram(ts_w_i_g_bot_75_le_bot_90_l_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # counts_3a, bins_3a = np.histogram(ts_w_i_g_bot_90_eq_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # counts_3b, bins_3b = np.histogram(ts_w_i_g_bot_90_l_1 ,bins=50,range=(0,max([t[0] for t in t_last_share])))
        # # plt.stairs(counts_2a,bins_2a,label=f"initial wage>bot75,<=bot90 and final wage=1",color="purple")
        # # plt.stairs(counts_2b,bins_2b,label=f"initial wage>bot75,<=bot90 and final wage<1",color="purple",linestyle="dashed")
        # # plt.stairs(counts_3a,bins_3a,label=f"initial wage>bot90 and final wage=1",color="blue")
        # # plt.stairs(counts_3b,bins_3b,label=f"initial wage>bot90 and final wage<1",color="blue",linestyle="dashed")
        # plt.title(f"Distribution of time of last share of Workers throughout market")
        # plt.legend()
        # plt.ylim((0,N))
        # plt.xlabel(f"time of last share, up to t={T}")
        # plt.ylabel("Density of time of last share throughout market")
        # if save:
        #     plt.savefig(f"q_learning_simulation_results/time_of_last_share/p=0.1_N={N}_k={k}_initial_distribution={p_l}_tls_distribution_seed={seed}.png")
        # plt.show()


        # worker_time_window = 50000#int(max([t[0] for t in t_last_share]))+100 # 100 time steps after the last share action
        # # print(worker_time_window)

        # for i in range(0,500,10):
        #     worker_k_wages = [w[i] for w in worker_wages[:worker_time_window]]
        #     worker_k_sharing_q_val = [w[i] for w in worker_share_given_wage[:worker_time_window]]
        #     worker_k_no_sharing_q_val = [w[i] for w in worker_no_share_given_wage[:worker_time_window]]
        #     plt.plot(worker_k_wages, label="Wage", linestyle="solid",color="purple")
        #     plt.plot(worker_k_sharing_q_val, label="Q(state,share)",linestyle="dashed",color="blue")
        #     plt.plot(worker_k_no_sharing_q_val, label="Q(state,no share)", linestyle="dashed",color="red")
        #     plt.legend(title="Worker characteristics")
        #     plt.xlabel("Time")
        #     plt.ylabel("Worker wage")
        #     plt.title(f"worker index {i} wage over time and action q values")
        #     # plt.ylim((0,1))
        #     if save:
        #         plt.savefig(f"q_learning_simulation_results/p=0.1_N={N}_k={k}_initial_distribution={p_l}_worker_{i}_wages_seed={seed}.png")
        #     plt.clf()






