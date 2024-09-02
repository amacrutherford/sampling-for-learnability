"""
Start either a new wandb sweep or start agents for an existing sweep within a tmux session.

Command line arguments:
    1. sweep_agent: file_name.yaml or entity/project/sweep_id 
    2. gpus_to_use: all, 0:4, 0,1,2,3 (default: all)
    3. agents_per_gpu: number of agents to start per gpu (default: 2)

"""

import jax
import yaml
import wandb
from os import system
import os
import sys

os.environ['WANDB_DISABLE_SERVICE']= "True"
wandb.login()

try:
    sweep_agent = sys.argv[1]
    if sweep_agent.endswith('.yaml'):
        # create a new sweep
        config_file = sweep_agent
        # check if config file exists
        assert os.path.exists(config_file), "Config file does not exist"
        print(f"Creating sweep with config from {config_file}")
        sweep_config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        project = sweep_config["project"]
        entity = os.environ.get("WANDB_ENTITY")
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"Created sweep with id: {sweep_id}")
    elif len(sweep_agent.split('/')) == 3:
        entity, project, sweep_id = sweep_agent.split('/')
    else: 
        raise ValueError("Invalid sweep agent")
except: 
    raise ValueError("No sweep agent")
    

try:
    gpus_to_use = sys.argv[2]
    if gpus_to_use == "all":
        gpus_to_use = list(range(len(jax.devices())))
    elif ":" in gpus_to_use:
        gpus_to_use = list(range(int(gpus_to_use.split(":")[0]), int(gpus_to_use.split(":")[1])))
    elif "," in gpus_to_use:
        gpus_to_use = list(map(int, gpus_to_use.split(",")))
    
except:
    # assumes all gpus
    gpus_to_use = list(range(len(jax.devices())))

assert len(gpus_to_use) <= len(jax.devices()), "More GPUs requested than available"

try:
    agents_per_gpu = int(sys.argv[3])
except:
    agents_per_gpu = 2

print('-- starting wandb sweep agents --')
print(f'entity: {entity}, project: {project}, sweep_id: {sweep_id}')
print(f'sweep command: wandb agent {entity}/{project}/{sweep_id}')
print(f'gpus_to_use: {gpus_to_use}, agents_per_gpu: {agents_per_gpu}, total_agents: {len(gpus_to_use) * agents_per_gpu}')

def tmux(command):
    system('tmux %s' % command)

def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)


# start sesstion
tmux('new-session -d -s sweep')

for gpu in gpus_to_use:
    for agent in range(agents_per_gpu):

        pane_name = f"{gpu}-{agent}"
        tmux(f'new-window -t sweep -n {pane_name}')
        command = f"CUDA_VISIBLE_DEVICES={gpu} XLA_PYTHON_CLIENT_PREALLOCATE=false wandb agent {entity}/{project}/{sweep_id}"

        tmux_shell(command)

print('view runs with: $ tmux a -t sweep')