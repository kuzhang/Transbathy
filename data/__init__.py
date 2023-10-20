import time

from torch.utils.data import DataLoader
import numpy as np
from .dataset import BathyDataset
import yaml
from tqdm import tqdm

def load_data(config):
    splits = ['train', 'test', 'val']
    shuffle = {'train': True, 'test': False, 'val': False}
    drop_last_batch = {'train': True, 'test': False, 'val': False}

    dataset = {}
    dataset['train'] = BathyDataset(config, split='train')
    dataset['test'] = BathyDataset(config, split='test')
    dataset['val'] = BathyDataset(config, split='val')

    if not config['Phase'] == 'test':
        dataloader = {x: DataLoader(dataset=dataset[x],
                                     batch_size=config['Data']['batch_size'],
                                     shuffle=shuffle[x],
                                     num_workers=int(config['Data']['workers']),
                                     drop_last=drop_last_batch[x],
                                     worker_init_fn=(None if config['Seed'] == -1
                                                     else lambda x: np.random.seed(config['Seed'])))
                      for x in splits}
        return dataloader

    else:
        dataloader = {'test': DataLoader(dataset=dataset['test'],
                                    batch_size=config['Data']['batch_size'],
                                    shuffle=shuffle['test'],
                                    num_workers=int(config['Data']['workers']),
                                    drop_last=drop_last_batch['test'],
                                    worker_init_fn=(None if config['Seed'] == -1
                                                    else np.random.seed(config['Seed'])))}
        return dataloader




if __name__ == '__main__':
    # Loading configurations in yaml file
    with open('../config/config-cpu-test.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # build dataloader
    dataloader = load_data(config)
    observations = []
    lons = []
    lats = []
    for data in tqdm(dataloader['test'], leave=False, total=len(dataloader['test'])):
        observations.extend(data['depth'].squeeze(-1).tolist())
        lons.extend(data['lon'].tolist())
        lats.extend(data['lat'].tolist())

    print('done')