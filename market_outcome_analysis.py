import matplotlib.pyplot as plt
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


def plot_attribute_distribution_market(market,attr, c, extra_counts = None, extra_bins = None,i_p_p=1,seed=0,save=False,n=0,i_p_w_c=1,o_o_c=None,f_t=None,l_H=None,l_L=None):
    
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
        plt.savefig(f"simulation_results/setting_2/seed={seed}_additional_settings_market_wage_distribution_graphs/i_p_{i_p_p}_i_p_w_c_{i_p_w_c}_o_o_c_{o_o_c}_f_t_{f_t}_l_H_{l_H}_l_L_{l_L}_{n}.png")
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
