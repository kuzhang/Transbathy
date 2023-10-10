import time

from torch.utils.data import DataLoader
import numpy as np
from .dataset import BathyDataset
import yaml

def load_data(config):
    splits = ['train', 'test', 'val']
    shuffle = {'train': True, 'test': False, 'val': False}
    drop_last_batch = {'train': True, 'test': False, 'val': False}

    dataset = {}
    dataset['train'] = BathyDataset(config, split='train')
    dataset['test'] = BathyDataset(config, split='test')
    dataset['val'] = BathyDataset(config, split='val')

    dataloader = {x: DataLoader(dataset=dataset[x],
                                 batch_size=config['Data']['batch_size'],
                                 shuffle=shuffle[x],
                                 num_workers=int(config['Data']['workers']),
                                 drop_last=drop_last_batch[x],
                                 worker_init_fn=(None if config['Seed'] == -1
                                                 else lambda x: np.random.seed(config['Seed'])))
                  for x in splits}
    return dataloader


if __name__ == '__main__':
    # Loading configurations in yaml file
    with open('../config/config-cpu.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # build dataloader
    dataloader = load_data(config)
    #time_i = time.time()
    for data in dataloader['train']:
        print("data:{}".format(data))
        #time_o = time.time()
        #print('loading time:{}'.format(time_o - time_i))
        print('done')