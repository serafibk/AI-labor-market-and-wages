'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 37
gen = np.random.default_rng(seed=seed)


class MVPT:

    def __init__(self, worker_data_pool, N_w,W, sd_cap=0.1):
        '''initialization of market value preictor tool (MVPT)
        
        params
            worker_data_pool (Set[Worker]): pool of initial workers who are sharing their wage information
            N_w (int): total number of workers in the market, tool is 100% accurate when all workers share data.
            sd_cap(float): ceiliing of standard deviation of normally distributed error, set to keep median prediction reasonably stable
        '''
        # data / noise parameter for predictions
        self.data_pool = worker_data_pool # to generate median 
        self.SD_cap = sd_cap # cap on error for stability, i.e., third-party will not release the tool if it has a SD of > SD_cap around true median.
        self.N_w = N_w # to track how much of the user base is currently in data pool
        self.sigma_e = 0.1 #(1-len(worker_data_pool)/(self.N_w)) * self.SD_cap # initial variance tied to how much of the data pool we have
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
        # self.sigma_e = (1-len(self.data_pool)/(self.N_w)) * self.SD_cap # variance tied to how much of the data pool we have

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e.

        params
            worker (Worker): Worker object to remove from data set. 
        '''
        self.data_pool.discard(worker)
        # self.sigma_e = (1-len(self.data_pool)/(self.N_w)) * self.SD_cap # variance tied to how much of the data pool we have

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        if len(self.data_pool) == 0:
            # print("Error, data pool empty")
            return 

        error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated, simulates inherent unpredictability of ML tools (over traditional statisical methods with valid confidence intervals)

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within reasonable bounds of data pool
        noisy_median = np.median([w.wage for w in self.data_pool]) + error
        self.m_hat = min(W, key = lambda x:abs(x-noisy_median))# mapping to closest value in wage list
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)




class Market:
    '''Market class defines the environment that the workers are learning to share data in (or not). greatly simplified from social learning setting.
    '''

    def __init__(self, N, states,actions,W,p,alpha,delta,explore_eps):
        '''initializaiton of the market

        Parameters
            N: number of workers
        '''

        # create Workers
        self.workers = [Worker(states,actions,gen.choice(W,p=p),alpha,explore_eps,delta) for i in range(N)] # workers identical up to initial wage and initial optimism (to be set after wage set)

        # initial matching of firms and workers (which sets their initial beliefs)
        expert_worker_idx = gen.integers(0,N)
        self.workers[expert_worker_idx].wage = float(1) # ensure at least one worker starts with MRPL

        # initializing mvpt after workers have an initial wage
        initial_worker_pool = set()
        for w in self.workers:
            if gen.binomial(1,0.5) >= 0.5:
                initial_worker_pool.add(w) # initial action is share w.p. 0.5
        
        self.mvpt = MVPT(initial_worker_pool,N,W) # initial worker pool is random
        self.mvpt.update_mvpt()
    
    def market_time_step(self):
        '''In a market time step, agents first seek information, then perform negotiations and update their beliefs, finally mvpt is updated for next round based on the data pool gathered this round

        Parameters
            T (int): total time in the simulation run, used in expectation calculations 
            t (int): current time step, used in expectation calculations
        '''
        # phase 1 - information seeking
        self.state_observation_and_action_choice() # worker now has an action

        # phase 2 - negotiation and belief updates    
        self.negotiation_and_state_transition()

    
    
    def _get_market_median(self):
        '''compute true median in market for hypothetical "firm". 
        '''
        # compute summary statistics
        all_wages = []
        for w in self.workers:
            all_wages.append(w.wage)
        
        return min(W, key=lambda x:abs(x-np.median(all_wages)))

    def state_observation_and_action_choice(self):
        '''
        '''

        # worker seeking
        for w in self.workers:
            w.state = (float(w.wage), float(self.mvpt.l_hat), float(self.mvpt.u_hat)) # observe state first
            w.action = w.share_decision() # choose decions eps-greedily
            if w.action == "share":
                self.mvpt.add_worker(w) # add to data pool for next round
            else:
                self.mvpt.remove_worker(w) # remove worker from data pool for next round




    def negotiation_and_state_transition(self):
        '''
        '''
        def _bargaining_outcome(o1, at1, at2, o2, at1_c):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(at1), counter offer cap (at2),  counter offer (o2), worker counter offer acceptance threshold (at1_c)'''
            if o1 <= at1: # opening offer lower than firm's median belief, accept
                return o1
            elif (o1 <= at2) and (o2 > at1_c): # opening offer lower than firm's upper belief, worker accepts reduced counteroffer if it is above their counter acceptance threshold
                return o2 
            else: # opening offer too high or counter offer too low, reject
                return -1

        # firms always see benchmark of whole market
        med = self._get_market_median()
        worker_offer = self.mvpt.m_hat # fixed for all workers


        # update mvpt based on workers CURRENT wage, m_hat already stored in worker_offer
        self.mvpt.update_mvpt() # signals for next state ready

        for w in self.workers: 
            
            if w.action == "share" and worker_offer > w.wage: # only negotiate if sharing and m_hat is greater than current wage
                p_accept_high = 0.1 # some realistic friction for getting high offers accepted #1 - len(self.mvpt.data_pool)/len(self.workers) # accept offers above median with probability proportional to number of workers in data pool
                accept_high = gen.binomial(1, p_accept_high)

                if accept_high:
                    firm_counter = worker_offer # accept offer if above lower end, but below high end of belief
                else:
                    firm_counter = med # counter offer median pay (todo, should this change a worker's belief about the validity of m_hat?)

                # bargaining outcome
                outcome = _bargaining_outcome(worker_offer, med, 1, firm_counter, w.wage)
                w.reward = w.get_reward(w.action,outcome)
                if outcome > 0:
                    w.wage = float(outcome) # new wage acheived
            else:
                w.reward = w.get_reward(w.action)
            
            new_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
            w.update_q_vals(new_state,w.action)  
            



class Worker:

    def __init__(self, states, actions, initial_wage, alpha = 0.5, explore_eps = 0.2, delta = 0.8):

        # environment parameters worker has access to
        self.states = states
        self.actions = actions # share or not share
        self.alpha = alpha
        self.explore_eps = explore_eps
        self.delta = delta

        # dictionary of Q vals
        self.Q_vals = dict({f"({s},{a})": 0 for s in self.states for a in self.actions}) # initial q-values

        # current values
        self.state = None
        self.action = None
        self.reward = None # current reward as a result of the previous state/action pair 
        self.wage = initial_wage


    def get_reward(self, action, bargaining_outcome = None, eps=1e-2):
        ''' if action == share, then the bargaining outcome is -1 if failed negotiation, 0 if no negotiation, and v (new wage) if successful negotiation
        '''
        if action == "no share":
            return 0
        else:
            if bargaining_outcome == None: # some benefit to sharing w/ no negotiation
                return eps
            elif bargaining_outcome > 0: # only if successful
                return bargaining_outcome - self.wage + eps
            else:
                return bargaining_outcome + eps

    def share_decision(self, beta = 0.5):

        explore = gen.binomial(1,self.explore_eps) # decide to explore or exploit

        if explore:
            action = gen.choice(self.actions,1)[0] # choose an action U.A.R.
        else:
            action_index = np.argmax([self.Q_vals[f"({self.state},{a})"] for a in self.actions])
            action = self.actions[action_index]
        
        # discount explore_eps
        self.explore_eps = self.explore_eps * np.e**(-beta)

        return action
    
    def update_q_vals(self, new_state, action):

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))

        self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (self.reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.actions]))







if __name__ == "__main__":


    # parameters
    N = 50 # small number of workers to start
    k = 10 # number of intervals to break [0,1] up into
    W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
    p = [1/11 for i in range(11)]# + [0 for j in range(3)]
    S = [(w,l,u) for w in W for l in W for u in W if l<= u] # (k+1)**3 states 
    A = ["share", "no share"] # action set for each worker

    # find values to fix these at to compare with previous work
    alpha = 0.3 # equal weight of current and future rewards
    delta = 0.95 # more patient
    explore_eps = 0.1 # low value since only two actions, could discount this over time

    save = False
    ## assuming any worker that wants to renegotiate will be able to find a firm with capacity that acts as we've assumed. no need for a firm class then.
    market = Market(N,S,A,W,p,alpha,delta,explore_eps)

    initial_counts_bins_market = get_wage_distribution_market(market)
    m_hat_values = []
    l_hat_values = []
    u_hat_values = []

    T = 500000 # consistent with previous simulations
    for t in tqdm(range(T)):
        m_hat_values.append(market.mvpt.m_hat)
        l_hat_values.append(market.mvpt.l_hat)
        u_hat_values.append(market.mvpt.u_hat)

        market.market_time_step()

        if t>499990:
            for w in market.workers:
                if w.wage < 1:
                    print(f"Worker: {w} with wage {w.wage}")
                    for s in S:
                        for a in A:
                            q_val = market.workers[1].Q_vals[f"({s},{a})"]
                            if q_val <0:
                                print(f"state, action: ({s},{a})")
                                print(f"q value: {q_val}")
        # print(f"m_hat: {market.mvpt.m_hat}")
        # for s in S:
        #     for a in A:
        #         q_val = market.workers[1].Q_vals[f"({s},{a})"]
        #         if q_val <0:
        #             print(f"state, action: ({s},{a})")
        #             print(f"q value: {q_val}")

    plt.plot(m_hat_values, label="m_hat value (median + error)")
    plt.plot(l_hat_values, color="red",label="Min value in data pool")
    plt.plot(u_hat_values, color="purple",label="Max value in data pool")
    plt.ylim((0,1))
    plt.title("MVPT summary statistics values over time")
    plt.xlabel("Time")
    plt.ylabel("MVPT summary statistics values")
    plt.legend()
    if save:
        plt.savefig(f"q_learning_simulation_results/N={N}_k={k}_initial_distribution=UNIFORM_mvpt_values_seed={seed}.png")
    plt.show()
    plot_attribute_distribution_market(market,"wage",N,extra_counts = initial_counts_bins_market[0],extra_bins =initial_counts_bins_market[1],k=k,p=p,seed=seed,save=save)






