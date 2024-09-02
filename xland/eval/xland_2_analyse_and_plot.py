import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
from plot_utils import _annotate_and_decorate_axis, plot_mean_std_xy



groups = [
    "high-13-sfl-40K",
    # "high-13-plr-v1",
    # "high-13-dr-v1",
    "meta-task-high-13-sfl-linear-0.0",
    "meta-task-high-13-sfl-linear-1.0",
    "high-13-sfl-uniform-v1",
]

groups_to_labels = {
    "high-13-dr-v1": "DR",
    "high-13-plr-v1": "PLR",
    "high-13-sfl-40K": "SFL",
    "meta-task-high-13-sfl-linear-0.0": "Linear(0)",
    "meta-task-high-13-sfl-linear-1.0": "Linear(1)",
    "high-13-sfl-uniform-v1": "Uniform",
}

GROUP_TO_COLOUR = {
    "high-13-sfl-40K": '#0173b2', 
    "high-13-dr-v1": '#de8f05', 
    "high-13-plr-v1": '#d55e00', 
    "meta-task-high-13-sfl-linear-0.0": "#cc78bc",
    "meta-task-high-13-sfl-linear-1.0": "#de8f05",
    "high-13-sfl-uniform-v1": "#029e73",
    "PLR-PVL": '#029e73', 
    "ACCEL-MaxMC": '#cc78bc', 
    "PLR-L1VL": '#ece133', 
    "RobustPLR": '#fbafe4', 
    "RobustACCEL": '#949494', 
    "Perfect Regret (Oracle)": '#56b4e9',
    "SFL(Dec-Learn)": 'red',
    "PVL(SFL)": 'purple',
    "Learnability (OldParams)": '#d55e00',
    "SFL(Dec-LearnRatio)": "yellow",
    "SFL(LargeBatch)": "purple",
    "SFL(NormalBatch)": "green",
    "ACCEL-Learn": "#7BAE7F",
    "PLR-Learn": "brown",
    "PVL(SFLNewparams)": "black",
}


percentages = [
    1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100
]

SEEDS = 5
TOTAL = 10000
cvar_metric = "win-rates"

def get_vals(group, seed_num):
    all_vals = {
        'env-id': jnp.zeros((SEEDS, TOTAL), dtype=jnp.int32),
        'win-rates': jnp.zeros((SEEDS, TOTAL), dtype=jnp.float32),
        'returns': jnp.zeros((SEEDS, TOTAL), dtype=jnp.float32),
    }
    for seed in range(SEEDS):
        file = f'eval/data/results/eval_{TOTAL}_envs_seed_{seed_num}/{group}/{seed}.csv'
        
        # if not os.path.exists(file): continue
        df = pd.read_csv(file)
        # env-id,win-rates,returns
        id =        jnp.array(df['env-id'])
        win_rates = jnp.array(df['win-rates'])
        returns =   jnp.array(df['returns'])

        all_vals['env-id'] = all_vals['env-id'].at[seed].set(id)
        all_vals['win-rates'] = all_vals['win-rates'].at[seed].set(win_rates)
        all_vals['returns'] = all_vals['returns'].at[seed].set(returns)
    return all_vals

cvar_results_per_group = {}
for group in groups:
    
    all_vals = get_vals(group, 0)
    all_vals_for_bias_calc = get_vals(group, 1)
    
    # print(all_vals['env-id'].mean(axis=0))
    
    cvar_results = jnp.zeros((SEEDS, len(percentages)), dtype=jnp.float32)
    for seed in range(SEEDS):
        argsort_idxs = jnp.argsort(all_vals[cvar_metric][seed])

        envs_in_sorted_order = all_vals['env-id'][seed][argsort_idxs]
        win_rates_in_sorted_order = all_vals[cvar_metric][seed][argsort_idxs]

        unbiased_win_rates_sorted_order = all_vals_for_bias_calc[cvar_metric][seed][argsort_idxs]

        for perc_id, percentage in enumerate(percentages):
            n = int((percentage / 100) * TOTAL)
            bad_biased = win_rates_in_sorted_order[:n].mean()
            good_unbiased = unbiased_win_rates_sorted_order[:n].mean()
            cvar_results = cvar_results.at[seed, perc_id].set(good_unbiased)

            print("CVAR BIAS", group, seed, percentage, bad_biased, good_unbiased)
    
    cvar_results_per_group[group] = cvar_results
    
for group_name, cvar in cvar_results_per_group.items():
    plot_mean_std_xy(
        percentages, cvar.mean(axis=0) * 100, cvar.std(axis=0) / np.sqrt(SEEDS) * 100, plt.gca(), label=groups_to_labels[group_name], marker='o', linewidth=4,
        color=GROUP_TO_COLOUR[group_name]
    )
_annotate_and_decorate_axis(plt.gca(), xlabel=r'$\alpha$', ylabel=r'Avg Win Rate % on worst-case $\alpha$% levels', labelsize='x-large', ticklabelsize='x-large', wrect=10, hrect=10, legend=True)
plt.xscale('log')
plt.xticks([1, 10, 100], ['1%', '10%', '100%'])
plt.tight_layout()
plt.savefig(f'results/xland_cvar_line_{cvar_metric}_all.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)
plt.savefig(f'../multi-robot-neurips/images/xland_cvar_line_{cvar_metric}_all.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)