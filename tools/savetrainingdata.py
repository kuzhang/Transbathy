import yaml
from data import load_data
from tqdm import tqdm
import pandas as pd
import os
import torch
import numpy as np


def seed(seed_value):
    """ Seed
    Arguments:
        seed_value {int} -- [description]
    """
    # Check if seed is default value
    if seed_value == -1:
        return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    if not config['Device'] == 'cpu':
        torch.backends.cudnn.deterministic = True

# Loading configurations in yaml file
with open('../config/config-gpu.yaml', 'r') as file:
    config = yaml.safe_load(file)
phase = config['Phase']
trn_dir = os.path.join(config['Output_dir'], 'train')
# build dataloader
dataloader = load_data(config)

observations = []
lons = []
lats = []
seed(config['Seed'])

dst = os.path.join(trn_dir, 'visual')
if not os.path.isdir(dst):
    os.makedirs(dst)



for data in tqdm(dataloader['train'], leave=False, total=len(dataloader['train'])):
    img = data['image']
    tgt = data['depth']
    lons.extend(data['lon'].tolist())
    lats.extend(data['lat'].tolist())
    observations.extend(data['depth'].squeeze(-1).tolist())



visual_data = {
    'lons': lons,
    'lats': lats,
    'observation': observations
}

df = pd.DataFrame(visual_data)
visual_save_file = 'train_sample_20240420.csv'
visual_file = os.path.join(dst, visual_save_file)
df.to_csv(visual_file, index=False, header=True)
print('saved')
