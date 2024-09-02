import os

import jax
import xminigrid
import pickle as pkl

benchmark_id = "high-3m"
seed = 0
NUM_TO_SAVE = 10000

benchmark = xminigrid.load_benchmark(benchmark_id)

rng = jax.random.key(seed)
rngs = jax.random.split(rng, NUM_TO_SAVE)

rulesets = jax.vmap(benchmark.sample_ruleset)(rngs)

# save rulesets
rulesets_dir = f"eval/data/rulesets/{benchmark_id}"
os.makedirs(rulesets_dir, exist_ok=True)

with open(f"{rulesets_dir}/{NUM_TO_SAVE}_rulesets.pkl", "wb") as f:
    pkl.dump(rulesets, f)

