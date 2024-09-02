import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import jax.numpy as jnp
from sfl.deploy.plot_utils import _annotate_and_decorate_axis, plot_mean_std_xy


# "MA-Learnability",
#     # "MA-PLR-PVL",
#     # "MA-PLR-MaxMC",
#     # "MA-ACCEL-MaxMC",
#     # "MA-DR",
ORDER = [
    "Learnability (ours)",
    "Perfect Regret (Oracle)",
    "DR",
    "ACCEL-MaxMC",
    "PLR-PVL",
    "PLR-MaxMC",
    "PLR-L1VL",
    "RobustPLR",
    "RobustACCEL",
]
GROUP_TO_COLOUR = {
    "Learnability (ours)": '#0173b2', 
    "DR": '#de8f05', 
    "PLR-PVL": '#029e73', 
    "PLR-MaxMC": '#d55e00', 
    "ACCEL-MaxMC": '#cc78bc', 
    "PLR-L1VL": '#ece133', 
    "RobustPLR": '#fbafe4', 
    "RobustACCEL": '#949494', 
    "Perfect Regret (Oracle)": '#56b4e9',
    # '#ece133', 
    # '#56b4e9'
}
def do_main(env='jaxnav-multi'):
    if env == 'jaxnav-multi':
        CLEAN_NAMES = {
            "MA-ACCEL-MaxMC": "ACCEL-MaxMC",
            "MA-PLR-PVL": "PLR-PVL",
            "MA-PLR-MaxMC": "PLR-MaxMC",
            "MA-DR": "DR",
            "MA-Learnability": "Learnability (ours)",
        }

    else:
        CLEAN_NAMES = {
                # "minigrid_plr_l1vl_single_v1": "PLR-L1VL",
                "MA-DR": "DR",
                "MA-ACCEL-MaxMC": "ACCEL",
                "MA-PLR-PVL": "PLR-PVL",
                "MA-PLR-MaxMC": "PLR-MaxMC",
                # "minigrid_plr_maxmc_single_v1": "PLR-MAXMC",
                "MA-Learnability": "Learnability (Ours)",
        }
    LABEL_TO_GROUP = {
        v: k for k, v in CLEAN_NAMES.items()
    }
    # GROUPS = list(CLEAN_NAMES.keys())
    GROUPS = [LABEL_TO_GROUP[o] for o in ORDER if o in CLEAN_NAMES.values()]

    # GROUPS = [
    #     "rsim_accel_maxmc_fixed_single_v4",
    #     "rsim_plr_pvl_single_v4",
    #     "rsim_plr_maxmc_single_v4",
    #     "rsim_dr_single_v4",
    #     "rsim_learnability_single_v4",
    # ][:4]

    bar_x = []
    bar_y = []
    bar_std = []

    all_vals_per_group = {}
    cvar_results_per_group = {}

    SEEDS = 3
    TOTAL = 10_000  # Total number of environments used

    percentages = [
        0.1, 1, 2, 5, 10, 20, 50, 100
    ]

    def get_vals(group, seed_num):
        print("GRPU", group, seed_num)
        all_vals = {
            'env-id': jnp.zeros((SEEDS, TOTAL), dtype=jnp.int32),
            'win-rates': jnp.zeros((SEEDS, TOTAL), dtype=jnp.float32),
            'returns': jnp.zeros((SEEDS, TOTAL), dtype=jnp.float32),
        }
        for seed in range(SEEDS):
            file = f'sfl/data/eval/results/{env}/eval_{TOTAL}_envs_seed_{seed_num}/{group}/{seed}.csv'
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

    for group in GROUPS:
        all_vals               = get_vals(group, 0)
        all_vals_for_bias_calc = get_vals(group, 1) # TODO MIKE

        # all vals here has all of the data
        print(all_vals['env-id'].mean(axis=0))
        assert (all_vals['env-id'].mean(axis=0) == jnp.arange(TOTAL)).all()

        avg_win_rate = all_vals['win-rates'].mean(axis=-1)
        avg_return   = all_vals['returns'].mean(axis=-1)

        # This is things we plot
        winrate_meanstd = (avg_win_rate.mean(), avg_win_rate.std())
        return_meanstd = (avg_return.mean(), avg_return.std())


        bar_x.append(group)
        bar_y.append(winrate_meanstd[0])
        bar_std.append(winrate_meanstd[1])

        all_vals_per_group[group] = all_vals


        cvar_results = jnp.zeros((SEEDS, len(percentages)), dtype=jnp.float32)
        for seed in range(SEEDS):
            argsort_idxs = jnp.argsort(all_vals['win-rates'][seed])

            envs_in_sorted_order = all_vals['env-id'][seed][argsort_idxs]
            win_rates_in_sorted_order = all_vals['win-rates'][seed][argsort_idxs]

            unbiased_win_rates_sorted_order = all_vals_for_bias_calc['win-rates'][seed][argsort_idxs]

            for perc_id, percentage in enumerate(percentages):
                # TODO get unbiased win rates here
                n = int((percentage / 100) * TOTAL)
                bad_biased = win_rates_in_sorted_order[:n].mean()
                good_unbiased = unbiased_win_rates_sorted_order[:n].mean()
                cvar_results = cvar_results.at[seed, perc_id].set(good_unbiased)

                print("CVAR BIAS", group, seed, percentage, bad_biased, good_unbiased)
        
        cvar_results_per_group[group] = cvar_results


    # Bar chart of solve rates over 10k randomly-generated (but solvable) levels
    for i in range(len(bar_x)):
        plt.bar(bar_x[i], bar_y[i] * 100, yerr=bar_std[i] * 100, label=CLEAN_NAMES[bar_x[i]],
                color=GROUP_TO_COLOUR[CLEAN_NAMES[bar_x[i]]])
    _annotate_and_decorate_axis(plt.gca(), xlabel='Group', ylabel='Win Rate', labelsize='x-large', ticklabelsize='x-large', wrect=10, hrect=10, legend=True)
    plt.xticks([])
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig(f'results/{env}/winrates_bar.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    for group_name, cvar in cvar_results_per_group.items():
        plot_mean_std_xy(
            percentages, cvar.mean(axis=0) * 100, cvar.std(axis=0) * 100, plt.gca(), label=CLEAN_NAMES[group_name], marker='o', linewidth=4,
            color=GROUP_TO_COLOUR[CLEAN_NAMES[group_name]]
        )
    _annotate_and_decorate_axis(plt.gca(), xlabel=r'$\alpha$', ylabel=r'Avg Win Rate % on worst-case $\alpha$% levels', labelsize='x-large', ticklabelsize='x-large', wrect=10, hrect=10, legend=True)
    plt.xscale('log')
    plt.xticks([1, 10, 100], ['1%', '10%', '100%'])
    plt.tight_layout()
    plt.savefig(f'results/{env}/cvar_line.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # Now do the worst-case results domination stuff for each of them
    # all_vals_per_group 

    BASE_MODEL = 'MA-Learnability'  # if env == 'rsim' else 'minigrid_learnability_single_v1'
    BASE_VALS  = all_vals_per_group[BASE_MODEL]

    fig, axs = plt.subplots(1, len(all_vals_per_group) - 1, figsize=(20, 5))
    items = all_vals_per_group.items()
    items = [(k, v) for (k, v) in items if k != BASE_MODEL]
    fig, axs = plt.subplots(1, len(items), figsize=(20, 5))
    for idx, (group_name, all_vals) in enumerate(items):
        if group_name == BASE_MODEL: continue
        x_data = BASE_VALS['win-rates'].mean(axis=0)
        y_data = all_vals['win-rates'].mean(axis=0)

        heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=10, density=False)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # plt.scatter(x_data, y_data, label=CLEAN_NAMES[group_name])
        # plt.scatter(x_data, y_data, label=CLEAN_NAMES[group_name])
        axs[idx].imshow(np.log(heatmap.T + 1), extent=extent, origin='lower')

        
        for (j, i), label in np.ndenumerate(heatmap.T):
            axs[idx].text((i + 0.5) / heatmap.shape[0], (j + 0.5) / heatmap.shape[0], int(label), ha='center',va='center')

        # sns.heatmap(heatmap, annot=True, fmt='d', cmap='viridis',  origin='lower')

        print(np.round(heatmap).astype(np.int32))
        _annotate_and_decorate_axis(axs[idx], xlabel=CLEAN_NAMES[BASE_MODEL], ylabel=CLEAN_NAMES[group_name], labelsize='x-large', ticklabelsize='x-large', wrect=10, hrect=10, legend=False)
    plt.tight_layout()
    plt.savefig(f'results/{env}/compare_scatter.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)
    # exit()

do_main('jaxnav-multi')
