from collections import OrderedDict
import os
import time
import torch
from torch import Tensor, nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from .transformer import Transformer
from .utils import weights_init


class BaseModel():
    """ Base Model for TransBath
    """
    def __init__(self, config, dataloader):
        """

        :param config: user defined model parameters, yaml file
        :param dataloader:
        """
        self.config = config
        # Seed for deterministic behavior
        self.seed(config['Seed'])

        # Initalize variables.
        self.dataloader = dataloader
        self.trn_dir = os.path.join(self.config['Output_dir'], 'train')
        self.tst_dir = os.path.join(self.config['Output_dir'], 'test')
        self.device = torch.device("cuda:0" if self.config['Device'] != 'cpu' else "cpu")


    def seed(self, seed_value):
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
        if not self.config['Device'] == 'cpu':
            torch.backends.cudnn.deterministic = True

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.trn_dir, 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.net.state_dict()},
                   '%s/net.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        self.net.train()
        step_loss = []
        epoch_iter = 0
        time_i = time.time()
        for data in tqdm(self.dataloader['train'], leave=False, total=len(self.dataloader['train'])):
            img = data['image'].to(self.device)
            tgt = data['depth'].to(self.device)
            epoch_iter += self.config['Data']['batch_size']

            self.optimizer.zero_grad()
            out = self.net(img)
            err = self.l_mse(out, tgt)
            err.backward()
            self.optimizer.step()

            rmse_err = torch.sqrt(err)
            step_loss.append(rmse_err.item())

        time_o = time.time()
        rmse = np.array(step_loss).mean()
        self.trainingEpoch_loss.append(rmse)

        train_time = time_o - time_i
        results = {
            "epoch": self.epoch,
            "rmse": rmse.item(),
            "training time": train_time
        }

        # Save test results
        dst = os.path.join(self.trn_dir, 'rmse')
        if not os.path.isdir(dst):
            os.makedirs(dst)

        train_save_file = self.config['Train']['save_file']
        results_file = os.path.join(dst, train_save_file)

        df = pd.DataFrame([results])
        if not os.path.exists(results_file):
            df.to_csv(results_file, index=False, header=True)
        else:
            df.to_csv(results_file, mode='a', header=False, index=False)

        print(">> Training model {}. Epoch {}/{} training rmse: {} \n" .
              format(self.name, self.epoch + 1, self.config['Train']['num_epoch'], rmse.item()))

    ##
    def train(self):
        """ Train the model
        """

        # TRAIN
        best_rmse = 100

        # Train for niter epochs.
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.config['Train']['start_epoch'], self.config['Train']['num_epoch']):
            # Train for one epoch
            self.train_one_epoch()
            res = self.test()
            if res[self.config['Metric']] <= best_rmse:
                best_rmse = res[self.config['Metric']]
                self.save_weights(self.epoch)

        print(">> Training model %s.[Done]" % self.name)

    ##
    def test(self):
        """ Test model
        """
        print(">> Testing model %s." % self.name)

        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.config['Test']['load_weights']:
                path = self.config['Test']['weight_dir']
                pretrained_dict = torch.load(path, map_location=self.device)['state_dict']

                try:
                    self.net.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("net weights not found")
                print('   Loaded weights.')

            predictions = []
            observations = []
            lons = []
            lats = []
            validationStep_loss = []
            time_i = time.time()
            for data in tqdm(self.dataloader['test'], leave=False, total=len(self.dataloader['test'])):
                img = data['image'].to(self.device)
                tgt = data['depth'].to(self.device)
                prediction = self.net(img)
                if self.config['Test']['visualize']:
                    lons.extend(data['lon'].tolist())
                    lats.extend(data['lat'].tolist())
                observations.extend(data['depth'].squeeze(-1).tolist())
                error = torch.sqrt(torch.mean(torch.pow((prediction - tgt), 2), dim=0))
                predictions.extend(prediction.to('cpu').squeeze(-1).tolist())
                validationStep_loss.append(error.item())

            time_o = time.time()
            rmse = np.array(validationStep_loss).mean()
            self.validationEpoch_loss.append(rmse)

            # Measure inference time.
            infer_time = time_o - time_i

            print('Testing time: {:.2f} s, and the rmse: {} \n' .format(infer_time, rmse.item()))

            # Calculate R-square
            corr_matrix = np.corrcoef(observations, predictions)
            corr = corr_matrix[0, 1]
            R_sq = corr ** 2
            print('Testing R square:{} \n'.format(R_sq.item()))

            results = {
                'epoch': self.epoch,
                'rmse': rmse.item(),
                'r2': R_sq.item(),
                'inference time': infer_time
            }

            # Save test rmse results
            if self.config['Test']['save_test']:
                dst = os.path.join(self.tst_dir, 'rmse')
                if not os.path.isdir(dst):
                    os.makedirs(dst)

                test_save_file = self.config['Test']['save_file']
                results_file = os.path.join(dst, test_save_file)

                df = pd.DataFrame([results])
                if not os.path.exists(results_file):
                    df.to_csv(results_file, index=False, header=True)
                else:
                    df.to_csv(results_file, mode='a', header=False, index=False)

            # save the predictions vs the observation
            if self.config['Test']['visualize']:
                visual_data ={
                    'lons': lons,
                    'lats': lats,
                    'predictions': predictions,
                    'observation': observations
                }

                df = pd.DataFrame(visual_data)
                visual_save_file = self.config['Test']['visual_save_file']
                visual_file = os.path.join(dst, visual_save_file)
                df.to_csv(visual_file, index=False, header=True)

            return results

class TransBathy(BaseModel):
    """
    TransBathy Model
    """

    @property
    def name(self): return 'TransBathy'

    def __init__(self, config, dataloader):

        super().__init__(config=config, dataloader=dataloader)

        num_features = config['Model']['num_features']
        num_encoder_layers = config['Model']['num_encoder_layers']
        num_decoder_layers = config['Model']['num_decoder_layers']
        num_heads = config['Model']['num_heads']
        dim_feedforward = config['Model']['dim_feedforward']
        dim_out = config['Model']['dim_out']
        resume = config['Train']['resume']
        span = config['Data']['span']

        kernel = ((2 * span) + 1) ** 2

        # -- Misc attributes
        self.epoch = 0

        # Create and initialize networks.
        self.net = Transformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_model=num_features,
            num_heads= num_heads,
            dim_feedforward=dim_feedforward,
            dim_out=dim_out,
            kernel=kernel,
            device=self.device
           )
        self.net.to(self.device)
        self.net.apply(weights_init)

        if resume:
            print("\nLoading pre-trained networks.")
            self.config['Train']['start_epoch'] = torch.load(os.path.join(self.trn_dir, 'weights/net.pth'))['epoch']
            self.net.load_state_dict(torch.load(os.path.join(self.trn_dir, 'weights/net.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_mse = nn.MSELoss()
        self.trainingEpoch_loss = []
        self.validationEpoch_loss = []

        if self.config['Phase'] == 'train':
            self.net.train()
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.config['Optimizer']['lr'], betas=(self.config['Optimizer']['beta'], 0.999))



    def reinit_net(self):
        """
        Re-initialize the weights of net
        """
        self.net.apply(weights_init)
        print('   Reloading net weight')
