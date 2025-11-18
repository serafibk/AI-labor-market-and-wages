import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


## Market Outcome Visualizations
def get_wage_distribution_within_firm(firm):
    wages = []
    for w in firm.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=500,range=(0,1))

    return counts, bins

def get_wage_distribution_market(market):
    wages = []
    for w in market.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=500,range=(0,1))

    return counts, bins

def get_wage_distribution_market_split_workers(market,lam_H = None):
    wages_H = []
    wages_L = []
    for w in market.workers:
        if w.outside_opp_rate == lam_H:
            wages_H.append(w.wage)
        else:
            wages_L.append(w.wage)

    counts_H, bins_H = np.histogram(wages_H,bins=500,range=(0,1))
    counts_L, bins_L = np.histogram(wages_L,bins=500,range=(0,1))

    return (counts_H, bins_H), (counts_L, bins_L)

def get_wage_distribution_market_optimistic_vs_pessimistic(market):
    wages_opt = []
    wages_pess = []
    wages_balanced = []
    for w in market.workers:
        if w.optimistic == 1:
            wages_opt.append(w.wage)
        elif w.pessimistic == 1:
            wages_pess.append(w.wage)
        else:
            wages_balanced.append(w.wage)

    counts_O, bins_O = np.histogram(wages_opt,bins=500,range=(0,1))
    counts_P, bins_P = np.histogram(wages_pess,bins=500,range=(0,1))
    counts_B, bins_B = np.histogram(wages_balanced,bins=500,range=(0,1))

    return (counts_O, bins_O), (counts_P, bins_P), (counts_B, bins_B)


def plot_attribute_distribution_market(market,attr, N_w,N_f, k=None,p=None,extra_counts = None, extra_bins = None,i_p_p=1,seed=0,save=False,n=0,i_p_w_c=1,o_o_c=None,f_t=None,l_H=None,l_L=None,J=None,c_b_H=None,c_b_L=None,c_b_O=None, c_b_P=None, c_b_B=None,m_u=None,m_n_u=None,m_a=None,i_w_l_p=None,i_w_u_p=None):
    
    attribute_values = []
    
    for w in market.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,bins=500,range=(0,1))
    plt.stairs(counts,bins,label=f"Final {attr} distribution")
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed",label=f"Initial {attr} distribution")
    plt.title(f"Distribution of {attr} of Workers throughout market")
    plt.legend()
    plt.ylim((0,N_w))
    plt.xlabel(f"{attr} value, between 0 and 1")
    plt.ylabel(f"Density of {attr} value throughout market")
    if save:
        # plt.savefig(f"simulation_results/seed=101/i_p_{i_p_p}_i_w_l_p_{i_w_l_p}_i_w_u_p_{i_w_u_p}_i_m<{m_u:.2f},{m_n_u:.2f},{m_a:.2f}>_market_wage_distribution_seed={seed}_n={n}.png")
        # plt.savefig(f"q_learning_simulation_results/p=0.1_N={c}_k={k}_initial_distribution={p}_market_wage_distribution_seed={seed}.png")
        plt.savefig(f"q_learning_sal_benchmarks_mvpt/p=0.01_N_w={N_w}_N_f={N_f}_k={k}_initial_distribution={p}_market_wage_distribution_seed={seed}.png")
        # plt.savefig(f"simulation_results/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_adj_{o_adj}_p_adj_{p_adj}_{n}_market_wage_distribution_seed={seed}.png")
        # plt.savefig(f"simulation_results/setting_2/seed={seed}_additional_settings_market_wage_distribution_graphs/job_switches_{J}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.png")
    plt.clf()


    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    def animate_H_L(t):
        ax.cla() # clear the previous image
        ax.stairs(c_b_H[t][0],c_b_H[t][1], label="Lam_H",color="blue")
        ax.stairs(c_b_L[t][0],c_b_L[t][1], label="Lam_L",color="red",linestyle="dashed") 
        ax.set_ylim((0,c))
        ax.set_title(f"Wage distribution of all workers at time {t}")
        ax.legend()
    
    def animate_O_P_B(t):
        ax.cla() # clear the previous image
        ax.stairs(c_b_O[t][0],c_b_O[t][1], label="Optimistic",color="blue")
        ax.stairs(c_b_P[t][0],c_b_P[t][1], label="Pessimistic",color="red",linestyle="dashed") 
        ax.stairs(c_b_B[t][0],c_b_B[t][1], label="Balanced",color="purple",linestyle="dotted") 
        ax.set_ylim((0,c))
        ax.set_title(f"Wage distribution of all workers at time {t}")
        ax.legend()

    if c_b_O is not None:
        if save and False:
            anim = animation.FuncAnimation(fig, animate_O_P_B, frames = len(c_b_O)-450, interval = 5, blit = False)
            anim.save(f"simulation_results/i_p_{i_p_p}_i_w_l_p_{i_w_l_p}_i_w_u_p_{i_w_u_p}_i_m<{m_u:.2f},{m_n_u:.2f},{m_a:.2f}>_market_wages_animations_seed={seed}_n={n}.gif")
            # anim.save(f"simulation_results/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_adj_{o_adj}_p_adj_{p_adj}_{n}_market_wages_animations_seed={seed}.gif")
            # anim.save(f'simulation_results/setting_2/seed={seed}_market_wages_animations_job_switches_{J}/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.gif', writer='Pillow', fps=30)
        else:
            anim = animation.FuncAnimation(fig, animate_O_P_B, frames = len(c_b_O)-450, interval = 5, blit = False)
            plt.show()
    if c_b_H is not None:
        anim = animation.FuncAnimation(fig, animate_H_L, frames = len(c_b_H), interval = 5, blit = False)
        if save:
            anim.save(f'simulation_results/setting_2/seed={seed}_market_wages_animations_job_switches_{J}/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.gif', writer='Pillow', fps=30)
        plt.clf()


def plot_attribute_distribution_within_firm(f_idx,firm,attr, c, extra_counts = None, extra_bins = None,i_p_p=1,seed=0,save=False,n=0,i_p_w_c=1):
    
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
        plt.savefig(f"simulation_results/seed={seed}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_{n}/distribution_of_{attr}_at_firm_{f_idx}")
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
