import pickle

seed=42


N_w = 10 # small number of workers to start
N_f = 5 # small number of firms
k = 5 # number of intervals to break [0,1] up into

s_label = "setting 3"
p_l = "u"
r_label="e-r"
b_label= "slow-fast"

n=0


folder = "simulation_output_data"
run_folder= f"N_w={N_w}_N_f={N_f}_k={k}_{s_label}_initial_dist={p_l}_beta={b_label}_risk={r_label}_seed={seed}_N={n}"

with open(f"{folder}/{run_folder}/q_vals_firm_0","rb") as pkl_fl:
    firm_0_q_vals = dict(pickle.load(pkl_fl))

for key, val in firm_0_q_vals.items():
    if val <0:
        print(f"Key: {key}, q-val: {val}")


