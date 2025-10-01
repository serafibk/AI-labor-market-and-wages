#!/bin/bash

ipps=(1 0.9 0.75 0.5 0.25 0.1 0)
ipwc=(1 0.9 0.7)
o_adj=(5 10 100)
p_adj=(10)

for i in "${!ipps[@]}"
do 
	for j in "${!ipwc[@]}"
	do
		for k in "${!o_adj[@]}"
		do
			for l in "${!p_adj[@]}"
			do
				mv i_p_${ipps[i]}_i_p_w_c_${ipwc[j]}_o_adj_${o_adj[k]}_p_adj_${p_adj[l]}_* i_p_${ipps[i]}_i_p_w_c_${ipwc[j]}_o_adj_${o_adj[k]}_p_adj_${p_adj[l]} || echo "simulation not completed"
			done
		done
	done
done
