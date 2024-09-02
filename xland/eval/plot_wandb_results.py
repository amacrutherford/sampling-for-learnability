import wandb 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import plot_mean_std_xy, _annotate_and_decorate_axis

api = wandb.Api()
entity = "alex-plus"
project = "xminigrid"

# metric_name = {
#     "high-13-sfl-metrics": "mean_num_rules",
#     "high-13-ued-metrics": "ruleset_mean_num_rules",
#     "high-13-dr-metrics": "ruleset_mean_num_rules",
# }

def download_metric_for_group(group, metric="eval/returns_mean"):

    # metric = metric_name[group]
    print('group', group, 'metric', metric)
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    
    df = pd.DataFrame(columns=["_step"])
    for i, run in enumerate(runs):
        print(run.name)
        # get data for metric 
        data = run.history(keys=[metric])
        print(data.head())
        # rename eval/returns_mean col name to run name
        data = data.rename(columns={metric: f"{run.name}-{i}"})
        
        
        # join data along _step
        df = pd.merge(df, data, how="outer", on="_step")
    print(df.head())
    return df


groups = [
    "high-13-sfl-40k",
    "high-13-plr-v1",
    "high-13-dr-v1",
]



groups_to_labels = {
    "high-13-dr-v1": "DR",
    "high-13-plr-v1": "PLR",
    "high-13-sfl-40k": "SFL",
    # "high-13-dr-metrics": "DR",
    # "high-13-ued-metrics": "PLR",
    # "high-13-sfl-metrics": "SFL",
}

GROUP_TO_COLOUR = {
    "high-13-sfl-40k": '#0173b2', 
    "high-13-dr-v1": '#de8f05', 
    "high-13-plr-v1": '#d55e00', 
    "high-13-dr-metrics": "#de8f05",
    "high-13-ued-metrics": "#d55e00",
    "high-13-sfl-metrics": "#0173b2",
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

# fig, ax = plt.subplots(1, 1, figsize=(5, 4))

data = {}
for group in groups:
    data[group] = download_metric_for_group(group)
    
    mean = data[group].drop(columns='_step').mean(axis=1)
    df_std_error = data[group].drop(columns='_step').std(axis=1) / np.sqrt(5)
    
    plot_mean_std_xy(
        data[group]['_step'], mean, df_std_error, plt.gca(), label=groups_to_labels[group], color=GROUP_TO_COLOUR[group]
    )

_annotate_and_decorate_axis(plt.gca(), xlabel='Meta-RL Update Step', ylabel='Mean Return on Evaluation Set', labelsize='x-large', ticklabelsize='x-large', wrect=10, hrect=10, legend=True)

plt.tight_layout()
plt.savefig('xland_return3.pdf', dpi=200, bbox_inches='tight', pad_inches=0.0)