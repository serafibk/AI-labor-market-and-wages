import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


## Market Outcome Visualizations
def get_wage_distribution_within_firm(firm):
    wages = []
    for w in firm.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=100,range=(0,1))

    return counts, bins

def get_wage_distribution_market(market):
    wages = []
    for w in market.workers:
        wages.append(w.wage)

    counts, bins = np.histogram(wages,bins=100,range=(0,1))

    return counts, bins

def get_wage_distribution_market_split_workers(market,lam_H = None):
    wages_H = []
    wages_L = []
    for w in market.workers:
        if w.outside_opp_rate == lam_H:
            wages_H.append(w.wage)
        else:
            wages_L.append(w.wage)

    counts_H, bins_H = np.histogram(wages_H,bins=100,range=(0,1))
    counts_L, bins_L = np.histogram(wages_L,bins=100,range=(0,1))

    return (counts_H, bins_H), (counts_L, bins_L)


def plot_attribute_distribution_market(market,attr, c, extra_counts = None, extra_bins = None,i_p_p=1,seed=0,save=False,n=0,i_p_w_c=1,o_o_c=None,f_t=None,l_H=None,l_L=None,J=None,c_b_H=None,c_b_L=None):
    
    attribute_values = []
    
    for w in market.workers:
        attribute_values.append(getattr(w,attr))
    counts, bins = np.histogram(attribute_values,bins=100,range=(0,1))
    plt.stairs(counts,bins,label=f"Final {attr} distribution")
    if extra_counts is not None:
        plt.stairs(extra_counts, extra_bins, color="red",linestyle="dashed",label=f"Initial {attr} distribution")
    plt.title(f"Distribution of {attr} of Workers throughout market")
    plt.legend()
    plt.ylim((0,c))
    plt.xlabel(f"{attr} value, between 0 and 1")
    plt.ylabel(f"Density of {attr} value throughout market")
    if save:
        # plt.savefig(f"simulation_results/setting_2/seed={seed}_additional_settings_market_wage_distribution_graphs/job_switches_{J}_i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.png")
        # plt.clf()

        fig, ax = plt.subplots(1, 1, figsize = (6, 6))
        def animate(t):
            ax.cla() # clear the previous image
            # ax.set_title(f"Proposer initial={get_support(beta_f,S_f)}, Responder initial={get_support(beta_c,S_c)}, M={M}, T={T}")
            # ax.plot(S_c,all_cdfs[t], label="Responder",color="blue")
            # ax.stairs()
            ax.stairs(c_b_H[t][0],c_b_H[t][1], label="Lam_H",color="blue")
            ax.stairs(c_b_L[t][0],c_b_L[t][1], label="Lam_L",color="red",linestyle="dashed") # 0 == Accept condition
            ax.set_title(f"Wage distribution of all workers at time {t}")
            # ax.scatter([proposer_final_most_mass], [1], label=f"NE point: {proposer_final_most_mass}", color="black")
            ax.legend()

        anim = animation.FuncAnimation(fig, animate, frames = 2000, interval = 5, blit = False)
        anim.save(f'simulation_results/setting_2/seed={seed}_market_wages_animations_job_switches_{J}/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.gif', writer='Pillow', fps=30)
        plt.clf()
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
    plt.clf()

def print_wage_distribution_within_firm(firm):
    
    wages = []
    for w in firm.workers:
        wages.append(w.wage)
    
    print(f"Firm id: {firm}")
    print(f"min wage: {np.min(wages)}")
    print(f"med wage: {np.median(wages)}")
    print(f"max wage: {np.max(wages)}")
    print("--------------------------")
