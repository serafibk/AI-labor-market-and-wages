import numpy as np
import math

seed = 1 # use for reproducibility of initial simulations to debug 
gen = np.random.default_rng()

class Worker:

    def __init__(self,s,f,a,u,mass_le_m_hat, mass_g_m_hat, mass_ambiguous):
        '''initialization of Worker agent 

        Parameters 
            s (float): search threshold for worker such that they will renegotiate their pay with own firm or another firm if their median pay belief is >(1+s)*current wage. s>=0.
            f (float): confidence level, in range [0,1], in mvpt at which worker stops sharing data for all future time steps
            a (float): confidence level, in range [0,1], in mvpt at and above which worker shares data and uses mvpt tool w.p. 1
            u (float): fairness uncertainty salience in range [0,1] which determines how likely they are to share data regardless of the perceived utiltiy of the tool 
            c (float): initial confidence in MVPT
         '''
        # worker attributes: fairness uncertainty and outside opp rate possibly heterogenous, search threshold and failure/acceptance threshold homogenous
        self.fairness_uncertainty = u # gen.beta(a=4,b=5) # fairness uncertainty salience, slightly left skewed distribution
        self.outside_opp_rate = None # initialized after worker gets initial wage
        self.search_threshold = s # premium over current wage required for woker to negotiate and switch
        self.failure_threshold = f # discard the tool when mvpt confidence falls below this threshold
        self.acceptance_threshold = a # always use the tool when mvpt confidence exceeds this threshold
        
        # worker employment attributes
        self.employer = None # initially unemployed
        self.wage = None # initially unemployed
        self.job_switches = 0 # tracking how many job switches this worker makes
        self.time_since_last_switch = 0 # tracking time since last switch (visible to firm)

        # worker negotiation attributes
        self.negotiating_with = None # firm that worker is trying to negotiate with
        self.reservation_wage = None # wage + search threshold 
        self.offer = None # wage offer that worker opens with 

        # counters for  mass of different outcomes
        self.mass_mvpt_useful = mass_le_m_hat # initial mass
        self.mass_mvpt_not_useful = mass_g_m_hat # initial mass
        self.mass_mvpt_ambiguous = mass_ambiguous # initial mass
        self.n_seeks = 0
        # self.seek_mvpt_mass = [self.i_w_le_m_hat, self.i_w_g_m_hat,self.i_w_ambiguous]
        # self.mass_successful_negotiations = mass_successful_negotiation
        # self.mass_failed_negotiations = mass_failed_negotiation
        # self.n_negotiations = 0
        # self.use_mvpt_mass = [self.i_successful_negotiations,self.i_failed_negotiaions]
       
        # # flags used in updates
        # self.in_initial_pool = 0 # whether worker is in initial data pool or not, always share data if so 
        # self.always_share = 0 # track if they should share data deterministically
        # self.stop_sharing = 0 # track if they ever stop sharing
        self.optimistic = 0
        self.pessimistic = 0
    
    def _p_seek_info(self):
        '''P(seek info) = 1-P(certain m_hat < wage)
        '''
        if self.optimistic == 1: # count ambiguous mass towards usefulness of the tool
            return 1-((self.mass_mvpt_not_useful)/(self.mass_mvpt_useful  + self.mass_mvpt_not_useful + self.mass_mvpt_ambiguous))
        else: # count ambiguous mass towards tool not being useful
            return 1-((self.mass_mvpt_not_useful+self.mass_mvpt_ambiguous)/(self.mass_mvpt_useful  + self.mass_mvpt_not_useful + self.mass_mvpt_ambiguous))

    def _p_trust_m_hat(self):
        ''' offer = p(success)*m_hat + (1-p(success))*(wage+s.c.) ARCHIVED'''

        return self.mass_successful_negotiations / (self.mass_successful_negotiations + self.mass_failed_negotiations)
    
    def _use_mvpt_mass_unbiased_update(self, evidence):
        '''evidence = success or fail ARCHIVED
        '''

        self.n_negotiations = self.n_negotiations + 1

        if evidence == "success":
            self.mass_successful_negotiations = (self.n_negotiations*self.mass_successful_negotiations + 1) / (self.n_negotiations + 1)
            self.mass_failed_negotiations = (self.n_negotiations*self.mass_failed_negotiations + 0) / (self.n_negotiations + 1)
        elif evidence == "fail":
            self.mass_successful_negotiations = (self.n_negotiations*self.mass_successful_negotiations + 0) / (self.n_negotiations + 1)
            self.mass_failed_negotiations = (self.n_negotiations*self.mass_failed_negotiations + 1) / (self.n_negotiations + 1)
        else:
            print(f"Error: unknown evidence {evidence}")
            return 
    
    
    
    def _seek_mass_biased_update(self, evidence):
        '''evidence = not useful, useful, or ambiguous
        '''
        self.n_seeks = self.n_seeks + 1 # tracks 't'

        if evidence == "not useful":
            self.mass_mvpt_not_useful = (self.n_seeks*self.mass_mvpt_not_useful + 1) / (self.n_seeks + 1)
            self.mass_mvpt_useful = (self.n_seeks*self.mass_mvpt_useful + 0) / (self.n_seeks + 1)
            self.mass_mvpt_ambiguous = (self.n_seeks*self.mass_mvpt_ambiguous  + 0) / (self.n_seeks + 1)
        elif evidence == "useful":
            self.mass_mvpt_useful = (self.n_seeks*self.mass_mvpt_useful + 1) / (self.n_seeks + 1)
            self.mass_mvpt_not_useful = (self.n_seeks*self.mass_mvpt_not_useful + 0) / (self.n_seeks + 1)
            self.mass_mvpt_ambiguous = (self.n_seeks*self.mass_mvpt_ambiguous  + 0) / (self.n_seeks + 1)
        elif evidence == "ambiguous":
            self.mass_mvpt_not_useful = (self.n_seeks * self.mass_mvpt_not_useful + self.mass_mvpt_not_useful*(1-self.mass_mvpt_ambiguous ))/(self.n_seeks + 1)
            self.mass_mvpt_useful = (self.n_seeks * self.mass_mvpt_useful + self.mass_mvpt_useful*(1-self.mass_mvpt_ambiguous ))/(self.n_seeks + 1)
            self.mass_mvpt_ambiguous = self.mass_mvpt_ambiguous  * (self.n_seeks + 2 - self.mass_mvpt_ambiguous)/(self.n_seeks+1)
        else:
            print(f"Error: unknown evidence {evidence}")
            return


    def _seek_mass_distributed_update(self, evidence):
        '''TODO -- other kinds of updates'''
        pass 

    def _seek_mass_unbiased_update(self, evidence):
        '''TODO -- other kinds of updates'''
        pass 

        

    def mvpt_information_seeking(self,mvpt,max_switches):
        '''
        Worker possibly shares wage data and seeks information from the MVPT to possibly initiate a negotiation.

        Parameters
            mvpt (MVPT): instance of mvpt for worker to possibly share data with and seek data from.
        '''

        # everyone else gets to check upper bounds for perceived usefulness (lower than lower bound means usefulness of the tool is ambiguous)
        if self.wage >= mvpt.u_hat: # if you don't see any possible improvement, tool is definitely not useful
            self._seek_mass_biased_update("not useful")
        else:
            ## wage > l_hat and <= u_hat 
            self._seek_mass_biased_update("ambiguous")

        p_share = self._p_seek_info() # sharing is totally tied to belief in usefulness now, cleaned up edge cases

         # do not share if confidence below failure threshold (reasonable buffer for creating deterministic behavior)
        if p_share <= self.failure_threshold: # or self.job_switches >= max_switches:
            mvpt.remove_worker(self) # worker leaves data pool
            self.offer_update() # offer is none
            return # do not seek info

        # if in the initial pool or confidence exceeds acceptance threshold, share w.p. 1
        if p_share >= self.acceptance_threshold: #or self.in_initial_pool == 1: 
            mvpt.add_worker(self)
            self.offer_update(m_hat = mvpt.m_hat)
            return # done

        seek_info = gen.choice([0,1],p=[1-p_share, p_share])

        if seek_info: # share data, belief update
            mvpt.add_worker(self) # share data 
            self.offer_update(m_hat = mvpt.m_hat) # update belief based on mvpt information
        else: # don't share data
            mvpt.remove_worker(self) 
            self.offer_update() # offer is none
        

    def offer_update(self,m_hat = None):
        '''
        Worker updates belief about reservation wage, mvpt confidence, and offer if they have mvpt m_hat information 

        Parameters
            m_hat (Float): median pay prediction from MVPT 
        '''
        # set reservation wage
        self.reservation_wage = (1+self.search_threshold)*self.wage

        mvpt_trust = 1 #self._p_trust_m_hat() -- I think this just creates unnecessary noise

        if m_hat is not None and m_hat > self.reservation_wage: 
            # create an offer using mvpt only if it exceeds reservation wage
            self.offer = mvpt_trust*m_hat + (1-mvpt_trust)*self.reservation_wage
            self._seek_mass_biased_update("ambiguous") # still not certain your offer will result in a successful negotiation, but looks promising
        elif m_hat is not None and m_hat <= self.reservation_wage:
            self._seek_mass_biased_update("not useful") # m_hat certaintly does not increase wage
        else:
            self.offer = None # no offer if m_hat not sought, no further updates to pmf of seeking info 



class Firm:

    def __init__(self,C):
        '''initialization of Firm agent

        Parameters
            C (int): capacity, maximum number of employees firm can support.
            c_t (int): binary indicator for whether firm counters conservatively and shades down high bids (0) or competitively and accepts high bids (1)
        '''
        # firm attributes: capacity homogenous
        self.capacity = C
        self.workers = set([]) # list of employed workers
        self.market_pay_belief = [0, 0, 0] # min, med, and max pay believed to be in the market, set by benchmark + worker pool characteristics
    

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
            N_w (int): total number of workers in the market, tool is 100% accurate when all workers share data.
            sd_cap(float): ceiliing of standard deviation of normally distributed error, set to keep median prediction reasonably stable
        '''
        # data / noise parameter for predictions
        self.data_pool = worker_data_pool # to generate median 
        self.SD_cap = sd_cap # cap on error for stability, i.e., third-party will not release the tool if it has a SD of > SD_cap around true median.
        self.N_w = N_w # to track how much of the user base is currently in data pool
        self.sigma_e = (1-len(worker_data_pool)/(self.N_w)) * self.SD_cap # initial variance tied to how much of the data pool we have

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
        self.sigma_e = (1-len(self.data_pool)/(self.N_w)) * self.SD_cap # variance tied to how much of the data pool we have

    
    def remove_worker(self, worker):
        '''removes specified Worker object from data pool. increases sigma_e.

        params
            worker (Worker): Worker object to remove from data set. 
        '''
        self.data_pool.discard(worker)
        self.sigma_e = (1-len(self.data_pool)/(self.N_w)) * self.SD_cap # variance tied to how much of the data pool we have

    def update_mvpt(self): # could have workers see some coarser information too to start with 
        '''updates m_hat prediction based on current data pool and sigma_e 
        '''
        if len(self.data_pool) == 0:
            print("Error, data pool empty")
            return 

        error = gen.normal(0,self.sigma_e) # regenerate error each time tool is updated, simulates inherent unpredictability of ML tools (over traditional statisical methods with valid confidence intervals)

        # update accurate u_hat, l_hat
        self.u_hat = np.max([w.wage for w in self.data_pool])
        self.l_hat = np.min([w.wage for w in self.data_pool])

        # update m_hat within reasonable bounds of data pool
        self.m_hat = np.median([w.wage for w in self.data_pool]) + error
        self.m_hat = min(self.m_hat,self.u_hat) 
        self.m_hat = max(self.m_hat,self.l_hat)

       

class Market:
    '''Market class is used to create a single market instance to keep track of the connections between firms and workers and MVPT.
     Additionally, carries out each phase of the market evolution.
    '''

    def __init__(self, N_f, N_w, C, f, a, s, b_k, sd_cap, i_p,o_o_c, lam_H, lam_L, J,i_p_w_c):
        '''initializaiton of the market

        Parameters
            N_f (int): Number of Firms
            N_w (int): Number of Workers
            C (int): capacity of each firm, max. number of workers they can hire
            f (float): failure threshold for worker's confidence in mvpt
            a (float): acceptance threshold for worker's confidence in mvpt
            s (float): search threshold for workers, i.e., premium added to their wage to make job switching worth it
            b_k (float): proportion of firms to include in benchmark information that firms see
            sd_cap (float): cap on the standard deviation of normally distributed error of mvpt tool m_hat prediction.
            i_p (float): initial pool proportion of workers sharing data with mvpt
            o_o_c (float): wage threshold such that workers have lam_H and lam_L opportunity rates above and below the threshold, respectively
            lam_H (float): rate of high wage offers from firms for high wage workers
            lam_L (float): rate of high wage offers from firms for low wage workers
            J (int):  total number of job switches each worker can make
            i_p_w_c (float): wage cap on workers in the initial pool
        '''
        # market attributes 
        self.benchmark_proportion = b_k
        self.outside_offer_cutoff = o_o_c
        self.high_lambda = lam_H
        self.low_lambda = lam_L
        self.max_switches = J

        # create Firms and Workers
        self.firms = [Firm(C) for i in range(N_f)] # firms identical up to default counter offer type 
        self.workers = [Worker(s,f,a,None,mass_le_m_hat=1, mass_g_m_hat=1, mass_ambiguous=1) for i in range(N_w)] # workers identical up to initial wage and initial optimism (to be set after wage set)

        # initial matching of firms and workers (which sets their initial beliefs)
        self._hire_initial_workers(np.split(np.asarray(self.workers), N_f)) # evenly split workers into firms
        expert_worker_idx = gen.integers(0,N_w)
        self.workers[expert_worker_idx].wage = 1 # ensure at least one worker starts with MRPL 

        # initializing mvpt after workers have an initial wage
        optimistic_worker_pool = set(gen.permutation([w for w in self.workers if w.wage <= i_p_w_c])[:int(N_w *i_p)]) # grab i_p proportion of initial workers randomly for MVPT OR all workers with wage less than threshold to possibly depend on wage
        pessimistic_worker_pool = set(gen.permutation([w for w in self.workers if (w not in optimistic_worker_pool)])) #[:int(N_w * 0.3)]) # rest of the workers pessimistic, could have some neutral
        for w in optimistic_worker_pool:
            w.mass_mvpt_useful = 100 # more certain tool will be useful, variable
            w.optimistic = 1
        for w in pessimistic_worker_pool:
            w.mass_mvpt_not_useful = 5 # more certain tool will not be useful, variable
            w.pessimistic = 1
        
        self.mvpt = MVPT(optimistic_worker_pool,N_w,sd_cap) # initial worker pool is exactly the optimistic pool
        self.mvpt.update_mvpt()

        # tracking data points during market evolution for analysis 
        self.num_successful_mvpt_negotiations = []
        self.num_failed_mvpt_negotiations = []

        self.num_successful_outside_negotiations = []
        self.num_failed_outside_negotiations = []

        self.num_better_to_wait_H = []
        self.num_better_to_wait_L = []


    
    def _hire_initial_workers(self, worker_assignment):
        '''initial wage generation and firm worker matches based on worker assignment 

        Parameters
            worker_assignment (np.array(Worker)): array of Workers of length len(self.firms) to split workers into firms
        '''
        for i,f in enumerate(self.firms):
            for j,w in enumerate(worker_assignment[i]):
                
                # generate initial wage in [0,1] according to beta distribution params
                wage = gen.beta(a=2,b=1) # somewhat skewed towards 1
                
                # outside opportunity rate determined by initial wage being relatively high or low
                if wage >= self.outside_offer_cutoff:
                    w.outside_opp_rate = self.high_lambda
                else:
                    w.outside_opp_rate = self.low_lambda

                # set worker wage
                w.wage = wage

                # link worker and employee together
                w.employer = f
                f.workers.add(w)

                # update worker belief to set res. wage 
                w.offer_update()
            
            # update firm belief based on initial worker pool
            f.belief_update()
    
    def market_time_step(self, T, t):
        '''In a market time step, agents first seek information, then perform negotiations and update their beliefs, finally mvpt is updated for next round based on the data pool gathered this round

        Parameters
            T (int): total time in the simulation run, used in expectation calculations 
            t (int): current time step, used in expectation calculations
        '''
        # phase 1 - information seeking
        self.information_seeking(T,t)

        # phase 2 - negotiation and belief updates    
        self.negotiation_and_belief_update()

        # phase 3 - update mvpt
        self.mvpt.update_mvpt()
    
    
    def _perform_benchmark(self):
        '''Benchmark independently sampled for each firm in the market. 
        '''
        # sample firms
        number_firms_to_check = int(self.benchmark_proportion * len(self.firms))
        sampled_firms = gen.choice(self.firms,number_firms_to_check,replace=False)

        # compute summary statistics
        all_wages = []
        for f in sampled_firms:
            for w in f.workers:
                all_wages.append(w.wage)
        
        return [np.min(all_wages), np.median(all_wages), np.max(all_wages)]
    
    def _job_switch(self, accepted_wage, firm, worker):
        '''Updates worker and firm when a wage is accepted and worker makes a job switch.

        Parameters
            accepted_wage (float): Worker's new wage at the firm they made a deal with
            firm (Firm): firm instance that is the worker's new employer
            worker (Worker): worker instance that is the firm's new employee
        '''
        # update connections and worker's wage
        worker.wage = accepted_wage
        worker.employer.workers.remove(worker) # remove from previous firm 
        worker.employer = firm # worker has firm as employer
        firm.workers.add(worker) # firm has worker as employee

        # update worker's trackers and possibly outside offer param
        worker.job_switches = worker.job_switches + 1
        worker.time_since_last_switch = -1 # will be corrected to 0
        if worker.wage >= self.outside_offer_cutoff: # check to see if they passed the threshold
            worker.outside_opp_rate = self.high_lambda

    def information_seeking(self, T, t):
        '''
        '''
        num_successful_outside = 0
        num_failed_outside = 0

        waiting_low = 0
        waiting_high = 0

        # firms always see benchmark first
        for i,f in enumerate(self.firms):
            benchmark_for_f = self._perform_benchmark()
            f.belief_update(benchmark_for_f)

        def _outside_offer_check(w):
            
            # Workers receive a true outside option according to a Poisson distribution parameterized by their rate
            outside_option = gen.poisson(w.outside_opp_rate) # 0 or 1 or >= 2 

            failed = 2
        
            if outside_option > 0: # anything non-zero counts as an outside option opportunity (can change this to binomial or can count 2 or more arrivals as the bertrand competition chance)
                # randomly sample a firm with a vacancy
                firm = gen.choice([f for f in self.firms if len(f.workers) < f.capacity],1)[0] 

                if outside_option >= 2: # perfect Bertrand competition with multiple firms giving offers 
                    offer = 1 # mrpl attained!
                else:
                    # print(f"[{firm.market_pay_belief[1]}, {firm.market_pay_belief[2]}]")
                    # o = gen.beta(a=1,b=2) # left skewed
                    # offer = (1-o)*firm.market_pay_belief[1]  + o*firm.market_pay_belief[2] 
                    offer = gen.uniform(firm.market_pay_belief[1], firm.market_pay_belief[2]) # something in the upper end, but exclusive of upper end (strictly less than mrpl) 

                # worker accepts iff offer is > wage + search costs 
                if offer > w.reservation_wage and w.job_switches < self.max_switches:
                    self._job_switch(offer, firm, w)
                    w.negotiating_with = None
                    failed = 0
                    return 0, failed # only don't use MVPT if switch with true outside option already happened
                elif offer <= w.reservation_wage: # don't count at max switches as fail
                    failed = 1
            return 1, failed

        # worker seeking
        for w in self.workers:
            seek_mvpt, failed = _outside_offer_check(w) # 1 if no outside offer, else 0

            if failed < 2: # just tracking statistics for visualization
                num_successful_outside = num_successful_outside + (not failed)
                num_failed_outside = num_failed_outside + failed

            # check if worker has enough switches left to make it to MRPL BEFORE checking mvpt
            p_ge_2 = 1-np.exp(-1 *w.outside_opp_rate) * sum([w.outside_opp_rate**i  / math.factorial(i) for i in range(2)]) # P(k>=2) for k opportunities in the next time step 
            better_to_wait =  p_ge_2*(T-t) >= 1 # only wait if you expect at least one chance to get MRPL
            if better_to_wait:
                self.mvpt.remove_worker(w) # worker does not share data, and waits for outside option to arrive 

            if seek_mvpt and (not better_to_wait):
                w.mvpt_information_seeking(self.mvpt, self.max_switches)
                # only attempt negotiation if it seems not worth it to wait for true outside option
                if (w.offer is not None) and (w.job_switches < self.max_switches): # and w.offer > w.reservation_wage, should already be guaranteed
                    new_firm = gen.choice([f for f in self.firms if len(f.workers) < f.capacity],1)[0] # randomly choose a firm (possibly including current firm) that has vacancies
                    w.negotiating_with = new_firm
                else:
                    w.negotiating_with = None
            elif better_to_wait and w.outside_opp_rate == self.high_lambda:
                waiting_high = waiting_high + 1
                w.negotiating_with = None
            elif better_to_wait and w.outside_opp_rate == self.low_lambda:
                waiting_low = waiting_low + 1
                w.negotiating_with = None
            else:
                w.negotiating_with = None
            
    
        self.num_successful_outside_negotiations.append(num_successful_outside)
        self.num_failed_outside_negotiations.append(num_failed_outside)

        self.num_better_to_wait_H.append(waiting_high)
        self.num_better_to_wait_L.append(waiting_low)
                        

    def negotiation_and_belief_update(self):
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

        successful_mvpt_negotiations = 0
        failed_mvpt_negotiations = 0

        for w in gen.permutation(self.workers): # randomize sequence of negotiations so that the same workers are not getting consistenly filtered out due to capacity
            
            firm = w.negotiating_with # firm that worker is potentially negotiating with

            if firm == None:
                continue # skip workers not renegotiating, success or failure if they used the tool already tracked

            # special cases
            if len(firm.workers)>= firm.capacity:
                # failed_mvpt_negotiations = failed_mvpt_negotiations + 1 # failure to negotiate, but still attempted
                continue # firm already at capacity, learn nothing about the tool
            
            if len(firm.workers) == 0:
                self._job_switch(w.offer, firm, w)
                # w.num_successes = w.num_successes + 1
                # w._use_mvpt_mass_unbiased_update("success")
                w._seek_mass_biased_update("useful")
                successful_mvpt_negotiations = successful_mvpt_negotiations+ 1
                continue # successful switch
            
            # counter offer, can be generous or strict, ARCHIVED -- fixed firm behavior for both settings 

            # if self.high_lambda > 0: # flags if there are outside opportunities in the market
            #     ## FLAGGING: this could be changed to be less of a threshold, but want to convey that firms are more certain a high wage worker is offering if it has been longer since their last switch
            #     p_type_H = min(w.time_since_last_switch * self.high_lambda, 1) # probability is 1 if t >= 1/self.high_lambda, o.w. proportional
            #     compete = gen.binomial(1, p_type_H) # bernoulli 
            # else: # else, rely on firm's default counter type
            #     compete = firm.counter_type

            p_accept_high = 0.1 # some realistic friction for getting high offers accepted #1 - len(self.mvpt.data_pool)/len(self.workers) # accept offers above median with probability proportional to number of workers in data pool
            accept_high = gen.binomial(1, p_accept_high)

            if accept_high:
                firm_counter = w.offer # accept offer if above lower end, but below high end of belief
            else:
                firm_counter = firm.market_pay_belief[1] # counter offer median pay (todo, should this change a worker's belief about the validity of m_hat?)

            # bargaining outcome
            outcome = _bargaining_outcome(w.offer, firm.market_pay_belief[1],firm.market_pay_belief[2], firm_counter, w.reservation_wage)
            
            if outcome == -1:
                # w.num_failures = w.num_failures + 1
                # w._use_mvpt_mass_unbiased_update("fail")
                w._seek_mass_biased_update("not useful")
                failed_mvpt_negotiations = failed_mvpt_negotiations+ 1
            else:
                self._job_switch(outcome, firm, w)
                # w.num_successes = w.num_successes + 1
                # w._use_mvpt_mass_unbiased_update("success")
                w._seek_mass_biased_update("useful")
                successful_mvpt_negotiations = successful_mvpt_negotiations + 1
        
        # workers update beliefs after all negotiations done and time since last switch 
        # for w in self.workers:
        #     w.time_since_last_switch = w.time_since_last_switch + 1 # those at -1 will be corrected to 0
        #     w.belief_update()

        # # firms update beliefs after all negotiations done
        # for f in self.firms:
        #     f.belief_update()
        
        # tracking how many attempted negotiations there were, 0=> wage stability 
        self.num_successful_mvpt_negotiations.append(successful_mvpt_negotiations)
        self.num_failed_mvpt_negotiations.append(failed_mvpt_negotiations)






## Archived code in case we want to add these features back in 

# ADD BACK IN to change success of the tool to be confirming information even if it doesn't help you search
# elif w.offer >= w.wage and w.sought_info: #< (1+w.search_threshold)*w.wage and w.offer >= w.wage and w.sought_info:
#     # w.num_successes = w.num_successes + 1 # success in the sense of your confidence in the tool? (no longer tracking)
#     w.negotiating_with = None
#     # mvpt_success = mvpt_success + 1
# elif w.offer < w.wage: #* (1-0.01): # testing tolerance 
#     # w.num_failures = w.num_failures + 1    (no longer tracking)
#     w.negotiating_with = None
# else: # median belief not based on mvpt, w.sought_info == 0
#     w.negotiating_with = None

 # if firm.counter_type == 0:
#     firm_counter = (1/2) *(firm.market_pay_belief[2]) + (1/2) * (w.market_pay_belief[1]) # shade up
# elif firm.counter_type == 1:
#     firm_counter = (1/2) *(firm.market_pay_belief[1]) + (1/2) * (w.market_pay_belief[1]) # shade down
# else:
#     firm_counter = firm.market_pay_belief[1] # median


# def information_seeking(self, benchmark):
#     '''

#     notes
#     - this could better align with current information seeking practices (e.g., x% look at independent manual benchmarks, y% look at glass door estimates of competitors, z% have access to a real time tool with higher accuracy than worker tool, etc.)
#     - okay, going to assume cost-free benchmark not based on number of applicants 

#     possible information seeking mechanisms
#     -- real-time benchmarking
#     -- competing offers from candidates (salary history, true outside offers, etc.)
#     -- old benchmarking + individual process from tailoring 
#     -- mvpt & user reported salaries
#     -- posted salary ranges of other companies
#     -- some other unknown, individualized process
#     -- (speculative) AI to suggest which benchmarks to use or to predict gaps in salary data collection or to set the strategy for benchmarking
#     '''
#     self.belief_update(benchmark=benchmark) # update belief with benchmark, always, just use function below


# print(f"Type H: P(>=2 offers arrive in next time step) = 1-P(<=1 offers arrive in next time step) = {1-cdf_le_1(lam_high)}")
#     print(f"Type H: Expected number of time steps in {T}-{t} steps left = P(>=2 offers arrive in next time step)*(T-t) ={(1-cdf_le_1(lam_high))*(T-t)}")
#     print(f"Type L: P(>=2 offers arrive in next time step) = 1-P(<=1 offers arrive in next time step) = {1-cdf_le_1(lam_low)}")
#     print(f"Type L: Expected number of time steps in {T}-{t} steps left = P(>=2 offers arrive in next time step)*(T-t) ={(1-cdf_le_1(lam_low))*(T-t)}")
