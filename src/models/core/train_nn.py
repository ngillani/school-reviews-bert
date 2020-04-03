# train_nn.py


from __future__ import print_function  # for python2

from collections import defaultdict
import numpy as np
import os
import time

import torch
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from config import RUNS_PATH
import src.utils as utils


class TrainNN(nn.Module):

    """
    Base class for this project that covers train-eval loop.
    """

    def __init__(self, hp, save_dir=None):
        super(TrainNN, self).__init__()
        self.hp = hp
        self.save_dir = save_dir

        self.models = []
        self.optimizers = []

        self.tr_loader = None
        self.val_loader = None
        self.end_epoch_loader = None

        self.writer = None

    ##############################################################################
    #
    # Training
    #
    ##############################################################################
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def lr_decay(self, optimizer, min_lr, lr_decay):
        """
        Decay learning rate by a factor of lr_decay
        """
        for param_group in optimizer.param_groups:
            if param_group['lr'] > min_lr:
                param_group['lr'] *= lr_decay
        return optimizer

    def preprocess_batch_from_data_loader(self, batch):
        return batch

    def one_forward_pass(self, batch):
        """
        Return dict where values are float Tensors. Key 'loss' must exist.
        """
        pass

    def pre_forward_train_hook(self):
        """Called before one forward pass when training"""
        pass

    def dataset_loop(self, data_loader, epoch, is_train=True, writer=None, tb_tag='train', stdout_f=None):
        """
        Can be used with either different splits of the dataset (train, valid, test)
        :return: dict
            key: str
            value: float
        """
        losses = defaultdict(list)
        for i, batch in enumerate(data_loader):
            # set up optimizers
            if is_train:
                self.pre_forward_train_hook()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

            # forward pass
            batch = self.preprocess_batch_from_data_loader(batch)  # [max_len, bsz, 5];
            result = self.one_forward_pass(batch)
            for k, v in result.items():
                if k.startswith('loss'):
                    losses[k].append(v.item())

            # optimization
            if is_train:
                result['loss'].backward()
                # nn.utils.clip_grad_value_(self.parameters(), self.hp.grad_clip)
                nn.utils.clip_grad_norm_(self.parameters(), self.hp.grad_clip)
                for optimizer in self.optimizers:
                    optimizer.step()

            # Logging
            if i % 10 == 0:
                step = epoch * data_loader.__len__() + i
		curr_mean_losses = {k: np.mean(values) for k, values in losses.items()}
		log_str = self.get_log_str(epoch, 'step {}'.format(i), 'train: {}'.format(is_train), curr_mean_losses)
                print(log_str)
		print(log_str, file=stdout_f)
                for k, v in result.items():
                    if k.startswith('loss'):
                        writer.add_scalar('{}/{}'.format(tb_tag, k), v.item(), step)

        mean_losses = {k: np.mean(values) for k, values in losses.items()}
        return mean_losses

    def get_log_str(self, epoch, context, dataset_split, mean_losses, runtime=None):
        """
        Create string to log to stdout
        """
        log_str = 'Epoch {} {} -- {}:'.format(epoch, context, dataset_split)
        for k, v in mean_losses.items():
            log_str += ' {}={:.4f}'.format(k, v)
        if runtime is not None:
            log_str += ' minutes={:.1f}'.format(runtime)
        return log_str

    def train_loop(self):
        """Train and validate on multiple epochs"""
        tb_path = os.path.join(self.save_dir, 'tensorboard')
        outputs_path = os.path.join(self.save_dir, 'outputs')
        os.makedirs(outputs_path)
        self.writer = SummaryWriter(tb_path)
        stdout_fp = os.path.join(self.save_dir, 'stdout.txt')
        stdout_f = open(stdout_fp, 'w')
        model_fp = os.path.join(self.save_dir, 'model.pt')

        print('Number of trainable parameters: ', self.count_parameters())

        # Train
        val_losses = []  # used for early stopping
        min_val_loss = float('inf')  # used to save model
        last_model_fp = None
        for epoch in range(self.hp.max_epochs):

            # train
            start_time = time.time()
            for model in self.models:
                model.train()
            mean_losses = self.dataset_loop(self.tr_loader, epoch, is_train=True, writer=self.writer, tb_tag='train', stdout_f=stdout_f)
            end_time = time.time()
            min_elapsed = (end_time - start_time) / 60
            log_str = self.get_log_str(epoch, 'After full training loop', 'train', mean_losses, runtime=min_elapsed)
            print(log_str)
	    print(log_str, file=stdout_f)
            stdout_f.flush()

            for optimizer in self.optimizers:
                self.lr_decay(optimizer, self.hp.min_lr, self.hp.lr_decay)

            # validate
            for model in self.models:
                model.eval()
            mean_losses = self.dataset_loop(self.val_loader, epoch, is_train=False, writer=self.writer, tb_tag='valid', stdout_f=stdout_f)
            val_loss = mean_losses['loss']
            val_losses.append(val_loss)
            log_str = self.get_log_str(epoch, 'After full validation loop', 'valid', mean_losses)
            print(log_str)
	    print(log_str, file=stdout_f)
            stdout_f.flush()

            # 
            self.end_of_epoch_hook()

            # Save best model
            if val_loss < min_val_loss:
                #  remove old model and symlink
                if last_model_fp is not None:
                    os.remove(last_model_fp)
                if os.path.exists(model_fp):
                    os.remove(model_fp)

                # save a file with epoch and loss in name, as well as a file named model.pt
                cur_fn = 'e{}_loss{:.4f}.pt'.format(epoch, val_loss)  # val loss
                cur_fp = os.path.join(self.save_dir, cur_fn)
                torch.save(self.state_dict(), cur_fp)
                torch.save(self.state_dict(), model_fp)

                # update,
                min_val_loss = val_loss
                last_model_fp = cur_fp

            # Early stopping
            if min_val_loss not in val_losses[-3:]:  # hasn't been an improvement in last 5 epochs
                break

        stdout_f.close()

    def end_of_epoch_hook(self):
        pass


    ##############################################################################
    #
    # Testing / Inference
    #
    ##############################################################################

    def load_model(self, dir):
        """
        Args:
            dir: str (location of trained model)
        """

        # Load hyperparams used to train model
        # TODO: we may want to change certain hyperparams at inference time (e.g. decode_method)
        # Currently, the below just overwrites it
        model_hp = utils.load_file(os.path.join(dir, 'hp.json'))
        for key, value in model_hp.items():
            setattr(self.hp, key, value)
        # Also want the updated values and save it next to inference/train.json....

        # Load trained weights
        weights_fp = os.path.join(dir, 'model.pt')
        print('Loading model weights from: ', weights_fp)
        self.load_state_dict(torch.load(weights_fp))

    def save_inference_on_split(self, loader=None, dataset_split=None, dir=None, ext=None):
        """
        Args:
            loader: DataLoader
            dataset_split: str
            dir: str (location to save inference/<dataset_split>.pkl>
            ext: str (e.g. 'json', 'pkl'; extension of file)
        """
        if loader is None:
            loader = self.get_data_loader(dataset_split, self.hp.batch_size, shuffle=False)
        inference = self.inference_loop(loader)
        fp = os.path.join(dir, 'inference', '{}.{}'.format(dataset_split, ext))
        utils.save_file(inference, fp, verbose=True)

    def inference_loop(self, loader):
        pass
