'''
This file is meant to implement a simple epsilon-greedy Q-learning dynamic programming algoirthm 
'''
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

from market_outcome_analysis import get_wage_distribution_market, plot_attribute_distribution_market, get_wage_distribution_market_split_workers


seed = 101
gen = np.random.default_rng()



# need word-of-mouth and firm-posting functions to be alternative "information sources" from the MVPT. Maybe market should have an information source attribute?
# need to assign workers to firms again, but no capacity needed. firms are also restrictive or not, and this only affects the reward workers get from sharing with colleagues vs. MVPT (probabilities should still be epsilon-greedy)


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
        self.N_w = N_w # to track how much of the user base is currently in data pool
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

        # error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated, simulates inherent unpredictability of ML tools (over traditional statisical methods with valid confidence intervals)

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within reasonable bounds of data pool
        noisy_median = np.median([w.wage for w in self.data_pool])
        self.m_hat = min(W, key = lambda x:abs(x-noisy_median))# mapping to closest value in wage list
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)


class SalaryBenchmark:
    '''salary benchmarking tool that firms have access to. They collect coarse salary information, and augment it for sharing firms with firm-reported wage data.
    '''

    def __init__(self):
        # initialization of tool. What constitutes the coarse data? what constitutes the granular data?
        pass




class Market:
    '''Market class defines the environment that the workers are learning to share data in (or not). greatly simplified from social learning setting.
    '''

    def __init__(self, N_w, N_f, states,actions,W,p,alpha,delta,explore_eps,initial_belief_strength, information_setting = "MVPT"):
        '''initializaiton of the market

        Parameters
            N_w: number of workers
        '''

        # create Workers
        self.workers = [Worker(states,actions,gen.choice(W,p=p),alpha,explore_eps,delta,initial_belief_strength) for i in range(N_w)] # workers identical up to initial wage so far
        self.firms = [Firm() for i in range(N_f)]  ## assuming firms have enough capacity for all workers in our market and all have constant returns to scale from the set of workers that apply 
        self.negotiation_matches = [[] for i in range(N_f)] # a set of workers associated with the firm they will attempt to negotiate with

        # initial matching of firms and workers (which sets their initial beliefs)

        # ensure at least one worker starts with MRPL
        expert_worker_idx = gen.integers(0,N)
        self.workers[expert_worker_idx].wage = float(1) 

        ## initializing information setting of the market
        # initializing mvpt after workers have an initial wage
        if information_setting == "MVPT":
            initial_worker_pool = set()
            for w in self.workers:
                if gen.binomial(1,0.5) >= 0.5:
                    initial_worker_pool.add(w) # initial pool generated with adding in workers uniform iid
            
            self.mvpt = MVPT(initial_worker_pool,N_w,W) # initial worker pool is random
            self.mvpt.update_mvpt()
        # initializing word-of-mouth parameters 
        elif information_setting == "Word-of-mouth"
            pass
        # initializing firm-posting parameters
        else:
            pass


        for w in self.workers: # set initial state of all workers
            first_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
            w.state = first_state 
        
        # tracking market behavior for output graphs
        self.num_negotiations = []
    
    def market_time_step(self):
        '''In a market time step, agents first seek information, then perform negotiations and update their beliefs, finally mvpt is updated for next round based on the data pool gathered this round

        Parameters
            T (int): total time in the simulation run, used in expectation calculations 
            t (int): current time step, used in expectation calculations
        '''
        # phase 1 - information seeking
        self.coarse_information_and_sharing_choice()  # firms and workers decide to share their private info

        # phase 2 - setting negotiation strategy
        self.negotiation_strategy_choice()

        # phase 3 - negotiation and belief updates    
        self.negotiation_and_state_transition()

    
    
    def _get_market_median(self):
        '''compute true median in market. 
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
            w.action = w.action_decision(sharing=True) # choose decions eps-greedily
            if w.action == "share":
                self.mvpt.add_worker(w) # add to data pool for next round
                w.reward = w.get_reward(w.action, sharing=True)
                next_state = (float(w.wage), float(self.mvpt.l_hat), float(self.mvpt.m_hat),float(self.mvpt.u_hat))
                w.update_q_vals(next_state,w.action,sharing=True)
                w.state = next_state
            else:
                self.mvpt.remove_worker(w) # remove worker from data pool for next round
                # delay reward until we know state after negotiations
        
        # fim seeking 
        for f in self.firms:
            pass
    
    def negotiation_strategy_choice(self):
        '''
        '''

        # worker offer decision
        for w in self.workers:
            w.offer = w.action_decision()
            if w.offer is not "No offer":
                firm_idx = gen.choice(range(len(self.firms)),1)[0] # uniformly at random
                self.negotiation_matches[firm_idx].append(w)
        
        # firm acceptance threshold decision
        for f in self.firms:
            f.acceptance_threshold = f.action_decision()
    

    def negotiation_and_state_transition(self):
        '''
        '''
        def _bargaining_outcome(offer, acceptance_threshold):
            '''bargaining outcomes parameterized by initial offers by workers (o1), immediate acceptance cap(at1), counter offer cap (at2),  counter offer (o2), worker counter offer acceptance threshold (at1_c)'''
            if offer <= acceptance_threshold: # opening offer lower than firm's acceptance threshold, accept.
                return offer
            else: # opening offer too high, reject
                return -1

            #elif (o1 <= at2) and (o2 > at1_c): # opening offer lower than firm's upper belief, worker accepts reduced counteroffer if it is above their counter acceptance threshold
                #return o2, , at2, o2, at1_c

        # med = self._get_market_median()
        # worker_offer =  #self.mvpt.m_hat # fixed for all workers

        # update mvpt based on workers CURRENT wage, m_hat already stored in worker_offer
        self.mvpt.update_mvpt() # signals for next state ready

        num_negotiations = 0
        for f_idx,f in enumerate(self.firms):
            num_offers = len(self.negotiation_matches[f_idx])


        for w in self.workers: 
            worker_offer = w.offer 

            if worker_offer is not "No offer": # only negotiate if sharing and m_hat is greater than current wage
                num_negotiations = num_negotiations + 1
                p_accept_high = 0.1 # some realistic friction for getting high offers accepted #1 - len(self.mvpt.data_pool)/len(self.workers) # accept offers above median with probability proportional to number of workers in data pool
                accept_high = gen.binomial(1, p_accept_high)

                if accept_high:
                    firm_AT = worker_offer # accept any worker offer
                else:
                    firm_AT = firm.acceptance_threshold # only accept the firm's set acceptance threshold

                # bargaining outcome
                outcome = _bargaining_outcome(worker_offer, firm_AT)
                w.reward = w.get_reward(w.offer,outcome)
                if outcome > 0:
                    w.wage = float(outcome) # new wage acheived
                elif outcome < 0: # if negotiation fails, some small probability that worker's wage gets reset.
                    p_reset_wage = 0.1 # robustness of this parameter?
                    reset_wage = gen.binomial(1, p_reset_wage)
                    
                    if reset_wage:
                        w.wage = float(0) # their "leisure wage" or some notion of reservation utility
            else:
                w.reward = w.get_reward(w.action)
            

            
            next_state = (float(w.wage), float(self.mvpt.l_hat),float(self.mvpt.u_hat))
            w.update_q_vals(next_state,w.offer)  
            w.state = next_state # update worker's new state
    
        self.num_negotiations.append(num_negotiations)
            



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
        self.Q_vals = share_q_vals | negotiation_q_vals # merge dictionaries
    
        
        # # initial belief strength
        # for w,l,u in self.states:
        #     if w < u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = initial_belief_strength # upweight sharing if your wage is strictly below upper bound
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = -1*initial_belief_strength # downweight no sharing if your wage is strictly below upper bound
        #     elif w >= u:
        #         self.Q_vals[f"({(w,l,u)},{'share'})"] = -1*initial_belief_strength # downweight sharing if your wage is at least the upper bound of the range
        #         self.Q_vals[f"({(w,l,u)},{'no share'})"] = initial_belief_strength # upweight sharing if your wage is at least the upper bound of the range


        # current values
        self.state = None
        self.share_action = None
        self.offer = None
        self.reward = None # current reward as a result of the previous state/action pair 
        self.wage = initial_wage


    def get_reward(self, action, sharing = False, bargaining_outcome = None, eps_share=1e-2):
        ''' if action == share, then the bargaining outcome is -1 if failed negotiation, 0 if no negotiation, and v (new wage) if successful negotiation
        '''
        if sharing:
            if action == "no share":
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
                action = self.share_actions[action_index]
        
        # discount explore_eps
        self.explore_eps = self.explore_eps * np.e**(-beta)

        return action
    
    def update_q_vals(self, new_state, action, sharing = False):

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if sharing:
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (self.reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.negotiation_actions]))
        else:
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (self.reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.share_actions]))

    
class Firm:
    '''implementation of firms in this market who are now also q-learning their negotiation strategy under different information conditions.'''
    def __init__(self, share_states, share_actions, negotiation_states, negotiation_actions, worker_wages, alpha = 0.5, explore_eps = 0.2, delta = 0.8,initial_belief_strength=0.5):

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
        self.Q_vals = share_q_vals | negotiation_q_vals # merge dictionaries
    
        # current values
        self.state = None
        self.share_action = None
        self.acceptance_threshold = None
        self.reward = None # current reward as a result of the previous state/action pair 
        self.worker_wages = worker_wages # starting employees


    def get_reward(self, action, sharing = False, eps_share=1e-2):
        ''' if action == share, then the bargaining outcome is -1 if failed negotiation, 0 if no negotiation, and v (new wage) if successful negotiation
        '''
        if sharing:
            if action == "no share":
                return 0
            else:
                return eps_share
        else:
            pass
            # num_successful - sum(new wages) - (num_initiated - num_successful)

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
                action = self.share_actions[action_index]
        
        # discount explore_eps
        self.explore_eps = self.explore_eps * np.e**(-beta)

        return action
    
    def update_q_vals(self, new_state, action, sharing = False):

        # Q(s,a) = (1-alpha) Q(s,a) + alpha * (current_reward + delta*max_a Q(new_state,a))
        if sharing:
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (self.reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.negotiation_actions]))
        else:
            self.Q_vals[f"({self.state},{action})"] = (1-self.alpha)*self.Q_vals[f"({self.state},{action})"] + self.alpha * (self.reward + self.delta * max([self.Q_vals[f"({new_state},{a})"] for a in self.share_actions]))







if __name__ == "__main__":


    # parameters
    N = 500 # small number of workers to start
    k = 10 # number of intervals to break [0,1] up into
    W = [float(i/k) for i in range(k+1)] # k + 1 possible wages
    # W[0] = 0.01 # cannot have wage = 0
    # p = [1/11 for i in range(11)]#+ [0 for j in range(3)]

    # states and actions
    # worker
    S_sharing_worker = [(w,l,u) for w in W for l in W for u in W if l<= u] # (k+1)**3 states 
    S_negotiation_worker = [(w,l,m,u) for w in W for l in W for m in W for u in W if l <= u and m>= l and m <= u]
    A_sharing = ["share", "no share"] # action set for each worker
    A_negotiation = ["No offer"] + W
    # firm 


    # find values to fix these at to compare with previous work
    alpha = 0.3 # more weight on present rewards
    delta = 0.95 # more patient
    explore_eps = 1 # low value since only two actions, discounts over time
    initial_belief_strength = 0

    T = 500000 # consistent with previous simulations TODO, early stopping criteria?
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
    
    # wages = [t10 for i in range(int(N_workers*0.1))] + [t25 for i in range(int(N_workers*0.15))] + [t50 for i in range(int(N_workers*0.25))] + [t75 for i in range(int(N_workers*0.25))] + [t90 for i in range(int(N_workers*0.15))]
    # counts, bins = np.histogram(wages,bins=10,range=(0,1))
    # plt.stairs(counts, bins)
    # plt.clf()
    # exit()"bimodal-left-skewed","bimodal-slightly-left-skewed",


    ## initial distributions
    # skewed left (k-l):
    kl = [3/4,1/16,1/32,1/64,0.9-(3/4+1/16+1/32+1/64)]+[0.1/(k-4) for i in range(k-4)]
    # slightly skewed left (s-k-l):
    skl = [1/2,1/16,1/32,1/64,0.7-(1/2+1/16+1/32+1/64)]+[0.3/(k-4) for i in range(k-4)]
    # uniform (u): 
    u= [1/(k+1) for i in range(k+1)]
    # slightly skewed right (s-k-r):
    skr = [0.3/(k-4) for i in range(k-4)]+[1/2,1/16,1/32,1/64,0.7-(1/2+1/16+1/32+1/64)]
    # skewed right(k-r):
    kr = [0.1/(k-4) for i in range(k-4)]+[0.9-(3/4+1/16+1/32+1/64),1/64,1/32,1/16,3/4]
    # bimodal even (b-e):
    be = [0.4] + [0.1/3 for i in range(3)] + [0 for i in range(k-7)] + [0.1/3 for i in range(3)] + [0.4]
    # bimodal slightly left (b-l):
    bl =  [0.45] + [0.1/3 for i in range(3)]  + [0 for i in range(k-7)]  + [0.1/3 for i in range(3)]   + [0.35] 
    # bimodal slightly right (b-r):
    br =  [0.35] + [0.1/3 for i in range(3)] + [0 for i in range(k-7)] + [0.1/3 for i in range(3)] + [0.45]

    p_settings = [u, bl]#kl,skl,u,skr,kr,
    p_labels = ["u", "b-l"] #"k-l","s-k-l","u","s-k-r","k-r" ,
    

    save = False

    for p_s, p_l in zip(p_settings, p_labels):

        print(f"distribution: {p_l}")

        ## assuming any worker that wants to renegotiate will be able to find a firm with capacity that acts as we've assumed. no need for a firm class then.
        market = Market(N,S,A,W,p_s,alpha,delta,explore_eps,initial_belief_strength)

        worker_wages = [[w.wage for w in market.workers]]
        # all_wages_initial= worker_wages[0]
        # bot_10 = np.percentile(all_wages_initial,10)
        # bot_25 = np.percentile(all_wages_initial,25)
        # bot_50 = np.percentile(all_wages_initial,50)
        # bot_75 = np.percentile(all_wages_initial,75)
        # bot_90 = np.percentile(all_wages_initial,90)
        # print(bot_10)
        # print(bot_25)
        # print(bot_50)
        # print(bot_75)
        # print(bot_90)
        # exit()



        worker_share_given_wage = [[w.Q_vals[f"({w.state},share)"] for w in market.workers]]
        worker_no_share_given_wage = [[w.Q_vals[f"({w.state},no share)"] for w in market.workers]]
        workers_actions = [[1] for w in market.workers]


        initial_counts_bins_market = get_wage_distribution_market(market)
        m_hat_values = []
        l_hat_values = []
        u_hat_values = []
        true_median = []

        for t in tqdm(range(T)):
            m_hat_values.append(market.mvpt.m_hat)
            l_hat_values.append(market.mvpt.l_hat)
            u_hat_values.append(market.mvpt.u_hat)
            true_median.append(market._get_market_median())

            market.market_time_step()

            # simple convergence check
            all_wages = [w.wage for w in market.workers]
            worker_wages.append(all_wages)

            all_sharing_q_vals = [w.Q_vals[f"({w.state},share)"] for w in market.workers]
            worker_share_given_wage.append(all_sharing_q_vals)

            all_no_sharing_q_vals = [w.Q_vals[f"({w.state},no share)"] for w in market.workers]
            worker_no_share_given_wage.append(all_no_sharing_q_vals)

            # all_action = [0 if w.action=="no share" else 1 for w in market.workers]
            for i in range(N):
                if market.workers[i].action=="no share":
                    action = 0 
                else: 
                    action =1
                workers_actions[i].append(action)
            
            ## Convergence criteria
            if min(all_wages) == 1:
                print("Converged to MRPL!")
                break
            # need something like offer == worker's wage, so they don't negotiate (negotiations stop for X time steps or  negotiations stop AND eps<threshold)
            if t >= T_negotiation and sum(market.num_negotiations[-T_negotiation:]) == 0:
                print(f"No negotiations for {T_negotiation} time steps")
                break
        
        ## analyzing
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
        # time_window = T
        # plt.plot(m_hat_values[:time_window], label="m_hat value (median + error)")
        # plt.plot(l_hat_values[:time_window], color="red",label="Min value in data pool")
        # plt.plot(u_hat_values[:time_window], color="purple",label="Max value in data pool")
        # plt.plot(true_median[:time_window], color="orange",linestyle="dashed",label="True median")
        # plt.ylim((0,1))
        # plt.title(f"MVPT summary statistics values over first {time_window} steps")
        # plt.xlabel("Time")
        # plt.ylabel("MVPT summary statistics values")
        # plt.legend()
        # if save:
        #     plt.savefig(f"q_learning_simulation_results/p=0.1_N={N}_k={k}_initial_distribution={p_l}_mvpt_values_seed={seed}.png")
        # plt.clf()

        plot_attribute_distribution_market(market,"wage",N,extra_counts = initial_counts_bins_market[0],extra_bins =initial_counts_bins_market[1],k=k,p=p_l,seed=seed,save=save)

        
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






