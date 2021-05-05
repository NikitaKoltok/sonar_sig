import argparse
import itertools
import json
import logging
import os
from pathlib import Path

import math
import matplotlib.pyplot as plt
import neptune
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from src.data_process.load_data import DatasetCreator
from src.models.dist_lstm import ModelLSTM
from src.utils.general import (seed_everything, set_logging, check_git_status, increment_dir, check_file,
                               get_latest_run, plot_lr_scheduler)
#from src.utils.metrics import f1_score
from sklearn.metrics import accuracy_score  #for 1 class dataset
from src.utils.torch import ModelEMA, FocalCELoss, select_device

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../')

import random
random.seed(5)

logger = logging.getLogger(__name__)


class TrainProcess(object):
    """
    Обучение модели в режиме mode = 'train'
    или тестирование модели в режиме 'test'
    с подгрузкой весов из существующего файла формата .pth
    """
    def __init__(self):
        self.mode = 'train'  # режим обучения
        self.model = None  # модель нейронной сети
        self.optimizer = None  # оптимизатор параметров модели
        self.scheduler = None  # оптимизатор шага обучения
        self.criterion = None
        self.scaler = None

        self.train_loader = None  # torch.utils.data.DataLoader
        self.val_loader = None  # torch.utils.data.DataLoader
        self.train_data = None  # torch.utils.data.Dataset
        self.val_data = None  # torch.utils.data.Dataset

        # классы ЛА
        # self.models = ['Tomahawk_BGM', 'F35A', 'F22_raptor', 'Harpoon', 'AH-1Cobra', 'Aerob', 'AH-1WSuperCobra',
        #                'ExocetAM39', 'Dassaultrafale', 'F16', 'Jassm', 'Mig29', 'Orlan', 'EuroFighterTyphoon']

        self.models = ['object']

        # классы гражданских целей
        # self.models = [
        #                'Begemot', 'Byk', 'Chelovek',
        #                'DorozhnyZnak1', 'Barier', 'Kamen',
        #                'Skoray', 'FuraToplivo', 'FuraGruz',
        #                'Bike', 'Moped', 'QuadroBike',
        #                'Lada4x4', 'Mazda-v3', 'Vaz'
        # ]

        # параметры модели
        self.encoder_size = [32, 64, 128, 256, 512, 1024]
        self.encoder_dropout = 0.3
        self.lstm_layers = 2
        self.lstm_hidden_size = 64
        self.lstm_dropout = 0.1
        self.is_bilstm = True

    def neptune_plot_log(self, epoch):
        def plot_confusion_matrix(cm, normalize=False, cmap=plt.cm.Reds):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.rcParams.update({'font.size': 15})
            plt.rcParams["font.family"] = "Times New Roman"
            # plt.colorbar()

            fmt = '.2f'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         size=17)

            plt.tight_layout()
            plt.ylabel('Верный класс', size=19, labelpad=.2)
            plt.xlabel('Предсказанный класс', size=19, labelpad=.2)

        all_predicted, all_labels = [], []
        self.model.eval()
        pbar = tqdm(self.val_loader, total=len(self.val_loader), ascii=True, desc='plot validation')
        for RCS, labels in pbar:
            RCS, labels = RCS.to(device), labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = self.model(RCS)

            _, pred = outputs.sigmoid().max(1)
            # _, label = labels.max(1)
            all_predicted.append(pred.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            # all_predicted.append(outputs.sigmoid().sum(1).detach().round().cpu().numpy())
            # all_labels.append(labels.sum(1).detach().cpu().numpy())

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(22, 20))
        types = self.models
        types_len = np.arange(len(types))
        axes.set_yticks(types_len)
        axes.set_yticklabels(types, size=19)
        axes.set_xticks(types_len)
        axes.set_xticklabels(types, rotation=30, size=19)

        # Plot non-normalized confusion matrix
        plt.subplot(axes)
        all_labels = np.concatenate(all_labels).squeeze()
        all_predicted = np.concatenate(all_predicted).squeeze()
        plot_confusion_matrix(confusion_matrix(all_labels, all_predicted), normalize=True)

        neptune.log_image(f'predict ep:{epoch}', fig)

    def train_step(self, epoch):
        """
        Обучает модель в режиме train. Пропускается в режиме test
        """

        prev_data = torch.rand((self.batch_size, self.lstm_hidden_size * 2)).to(device='cuda')
        total_grad_norm = 0

        if self.rank != -1:
            self.train_loader.sampler.set_epoch(epoch)
        all_predicted, all_labels = [], []
        mloss = torch.zeros(1).to(device)  # mean losses
        nb = len(self.train_loader)
        self.model.train()

        #prev = torch.rand((64, 128), requires_grad=True).to(device='cuda')

        pbar = enumerate(self.train_loader)
        logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'total', 'batch'))
        if self.rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        self.optimizer.zero_grad()
        for i, (RCS, labels) in pbar:
            self.ni = i + nb * epoch  # number integrated batches (since train start)
            RCS, labels = RCS.to(device), labels.to(device).unsqueeze(0).transpose(0, 1)
            #print(RCS.shape)
            #print(labels)

            #Warmup
            if self.ni <= self.nw:
                xi = [0, self.nw]  # x interp
                self.accumulate = max(1, np.interp(self.ni, xi, [1, self.nbs / self.total_batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(self.ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(self.ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            with torch.set_grad_enabled(True):
                # Forward
                with amp.autocast(enabled=True):
                    pred, prev_data_1 = self.model(RCS, prev_data)  # forward
                    prev_data = prev_data_1.detach()
                    #pred = self.model(RCS)
                    loss = self.criterion(pred, labels)  # loss scaled by batch_size
                    #loss = self.criterion(pred, labels)

                # Backward
                self.scaler.scale(loss).backward()

                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2

                total_grad_norm = total_grad_norm ** (1./2)
                    #p.register_hook(lambda grad: torch.clamp(grad, -500, 500))

                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1000)

                # old accumulate
                # if (i + 1) % self.accumulate or self.accumulate == 1:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                #     self.model.zero_grad()

                # Optimize
                if self.ni % self.accumulate == 0:
                    self.scaler.step(self.optimizer)  # optimizer.step
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # self.model.zero_grad()  # test line of code
                    if self.ema:
                        self.ema.update(self.model)

            # Print
            if self.rank in [-1, 0]:
                mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (
                    torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 2) % (
                    '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, labels.shape[0])
                pbar.set_description(s)

            pred = pred.sigmoid().round()
            #print(pred)
            # _, label = labels.max(1)
            all_predicted.append(pred.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            # all_predicted.append(outputs.sigmoid().sum(1).detach().round().cpu().numpy())
            # all_labels.append(labels.sum(1).detach().cpu().numpy())

        all_predicted = np.vstack(all_predicted)
        #print(all_predicted)
        all_labels = np.vstack(all_labels)
        #print(all_labels)
        #f_1 = f1_score(self.nc, all_predicted, all_labels)
        f_1 = accuracy_score(all_labels, all_predicted)  #for 1 class dataset
        #print(acc)
        return mloss, f_1, total_grad_norm

    def validation_step(self, epoch):
        """
        Не изменяя веса, тестирует модель. Выполняется в обоих режимах
        """

        prev_data = torch.rand((self.batch_size, self.lstm_hidden_size * 2)).to(device='cuda')
        total_grad_norm = 0

        if self.rank != -1:
            self.val_loader.sampler.set_epoch(epoch)
        all_predicted, all_labels = [], []
        mloss = torch.zeros(1).to(device)  # mean losses

        self.model.eval()
        nb = len(self.val_loader)
        pbar = enumerate(self.val_loader)
        logger.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'total', 'batch'))
        if self.rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        for i, (RCS, labels) in pbar:
            RCS, labels = RCS.to(device), labels.to(device).unsqueeze(0).transpose(0, 1)
            with torch.set_grad_enabled(False):
                outputs, prev_data_val = self.model(RCS, prev_data)
                prev_data = prev_data_val.detach()
                #outputs = self.model(RCS)
                loss = self.criterion(outputs, labels)

                for p in self.model.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2

                #loss = self.criterion(outputs, labels)

            # Print
            if self.rank in [-1, 0]:
                mloss = (mloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = '%.3gG' % (
                    torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 2) % (
                    '%g/%g' % (epoch, self.epochs - 1), mem, *mloss, labels.shape[0])
                pbar.set_description(s)

            pred = outputs.sigmoid().round()
            # _, label = labels.max(1)
            all_predicted.append(pred.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            # all_predicted.append(outputs.sigmoid().sum(1).detach().round().cpu().numpy())
            # all_labels.append(labels.sum(1).detach().cpu().numpy())

        all_predicted = np.vstack(all_predicted)
        all_labels = np.vstack(all_labels)
        #f_1 = f1_score(self.nc, all_predicted, all_labels)
        f_1 = accuracy_score(all_labels, all_predicted)  #for 1 class dataset
        #print(acc)
        qwk = cohen_kappa_score(all_predicted, all_labels, weights='quadratic')

        return mloss, [f_1, qwk], total_grad_norm

    def load_data(self):
        """
        Загружает данные в обучающую и валидационную выборки
        """
        learning_data_set = DatasetCreator(pics_root_dir="../../signal_labels_new/patches_5000_grand/learning/",
                                           labels_root_dir="../../signal_labels_new/patches_5000_grand/learning/",
                                           obj_percentage=10, amp_data=True, complex_data=False)

        validation_data_set = DatasetCreator(pics_root_dir="../../signal_labels_new/patches_5000_grand/validation/",
                                             labels_root_dir="../../signal_labels_new/patches_5000_grand/validation/",
                                             obj_percentage=50, amp_data=True, complex_data=False)

        # create class balance validation
        learning_labels = np.fromfile("../../signal_labels_new/patches_5000_grand/learning/labels.bin", dtype=np.float)
        valid_labels = np.fromfile("../../signal_labels_new/patches_5000_grand/validation/labels.bin", dtype=np.float)
        train_idx = np.arange(len(learning_labels))
        valid_idx = np.arange(len(valid_labels))

        #df = data_set.portrait_data
        #for cl in df.Id_Class.unique():
        #    cl_df = df[df.Id_Class == cl]
        #    tmp_idx = cl_df.sample(int(0.3 * cl_df.shape[0]), replace=False).index
        #    valid_idx.append(tmp_idx)

        #valid_idx = np.concatenate(valid_idx).squeeze()
        #train_idx = df.index.difference(valid_idx)
        train_indices, val_indices = train_idx, valid_idx

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = torch.utils.data.DataLoader(learning_data_set, sampler=train_sampler, batch_size=opt.batch_size,
                                                        shuffle=False, num_workers=opt.workers, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(validation_data_set, sampler=valid_sampler, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=opt.workers, drop_last=True)

    def cyclical_lr(self, stepsize, epochs, min_lr=3e-4, max_lr=3e-3):

        scaler = lambda x: 1.

        lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * 2 ** (-it // (2 * stepsize))

        def relative(it, stepsize):
            cycle = np.floor(1 + it / (2 * stepsize))
            x = abs(it / stepsize - 2 * cycle + 1)
            return max(0, (1 - x)) * scaler(cycle)

        return lr_lambda

    def run(self):
        """
        Задает основные параметры модели и запускает процесс обучения и тренировки
        """
        #writer = SummaryWriter('runs/test_run')

        logger.info(f'Hyperparameters {hyp}')
        log_dir = Path(opt.log_dir)  # logging directory
        wdir = log_dir / 'weights'  # weights directory
        os.makedirs(wdir, exist_ok=True)
        last = wdir / 'last.pt'
        best = wdir / 'best.pt'
        results_file = str(log_dir / 'results.txt')
        self.epochs, self.batch_size, self.total_batch_size, weights, self.rank = \
            opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

        # Save run settings
        with open(log_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(log_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        # Configure
        self.cuda = device.type != 'cpu'
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        train_path = data_dict['train']
        test_path = data_dict['val']
        self.nc, names = (1, ['item']) if opt.single_cls else (
        int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(names), self.nc, opt.data)  # check
        print(len(names))

        # Model
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint

            self.model = ModelLSTM(input_size=1, output_size=self.nc,
                                   blocks_size=self.encoder_size, dropout=self.encoder_dropout,
                                   hidden_dim=self.lstm_hidden_size, lstm_layers=self.lstm_layers,
                                   bidirectional=self.is_bilstm).to(device)  # create

            #state_dict = ckpt['model'].float().state_dict()  # to FP32
            #state_dict = ckpt['state_dict_model'].float().state_dict()  # to FP32
            state_dict = ckpt['state_dict_model']
            self.model.load_state_dict(state_dict, strict=False)  # load
            logger.info(
                'Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), weights))  # report
        else:
            self.model = ModelLSTM(input_size=1, output_size=self.nc,
                                   blocks_size=self.encoder_size, dropout=self.encoder_dropout,
                                   hidden_dim=self.lstm_hidden_size, lstm_layers=self.lstm_layers,
                                   bidirectional=self.is_bilstm).to(device)  # create

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / opt.batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= opt.batch_size * self.accumulate / self.nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm1d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        #self.optimizer = optim.RMSprop(pg0, lr=hyp['lr0'])
        self.optimizer = optim.Adam(pg0, lr=hyp['lr0'], weight_decay=0.001)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # criterion
        # self.criterion = FocalCELoss(alpha=hyp["alpha"],
        #                              gamma=hyp["gamma"],
        #                              smooth=hyp["smooth"]).to(device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4))
        #self.criterion = nn.CrossEntropyLoss()

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        #self.lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
        self.lf = self.cyclical_lr(stepsize=10, epochs=self.epochs, min_lr=hyp['lr0'], max_lr=hyp['lrf'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        plot_lr_scheduler(self.optimizer, self.scheduler, self.epochs)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                             T_max=self.num_epochs,
        #                                                             eta_min=self.min_lr,
        #                                                             last_epoch=-1)

        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                               T_max=self.num_epochs,
        #                                                               eta_min=self.min_lr,
        #                                                               last_epoch=-1)
        #
        # self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=self.warmup_factor,
        #                                         total_epoch=self.warmup_epoch,
        #                                         after_scheduler=scheduler_cosine)

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if opt.sync_bn and self.cuda and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(device)
            logger.info('Using SyncBatchNorm()')

        # Exponential moving average
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None

        # TrainLoader
        self.load_data()
        nb = len(self.train_loader)

        # start training process
        # DDP mode
        if self.cuda and self.rank != -1:
            self.model = DDP(self.model, device_ids=[opt.local_rank], output_device=opt.local_rank)

        # Start training
        start_epoch, best_fitness = 0, 0.0
        #self.nw = max(round(hyp['warmup_epochs'] * nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        self.nw = 0
        self.scheduler.last_epoch = start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        logger.info('Using %g dataloader workers\nLogging results to %s\n'
                    'Starting training for %g epochs...' % (
                        self.train_loader.num_workers, log_dir, self.epochs))

        # epoch --------------------------------------------------------------------------------------------------------


        for epoch in range(self.epochs):
            final_epoch = epoch == (self.epochs - 1)

            train_loss, train_f1, total_grad_norm = self.train_step(epoch)  # batch iteration

            # Scheduler
            lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
            self.scheduler.step()

            # DDP process 0 or single-GPU
            if self.rank in [-1, 0]:
                # mAP
                # if self.ema:
                #     self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])

                val_loss, val_acc, total_grad_norm_val = self.validation_step(epoch)
                val_f1, val_qwk = val_acc

                results = [train_loss, train_f1,
                           val_loss, val_f1, val_qwk,
                           total_grad_norm, total_grad_norm_val]

                # Log
                tags = ['train/loss', 'train/F1',  # train history
                        'val/loss', 'val/F1', 'val/qwk', 'grad_norm_train',  'grad_norm_val',  # val history
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for name, weight in self.model.named_parameters():
                    if weight.requires_grad:
                        tb_writer.add_histogram(f'{name}.grad', weight.grad, epoch)

                for x, tag in zip(results + lr, tags):
                    #if tb_writer:
                     #   tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                    if nep_writer:
                        neptune.log_metric(tag, self.ni, x)  # neptune

                if (epoch + 1) % 50 == 0 or final_epoch:
                    self.neptune_plot_log(epoch)

                # Update best F1
                if val_f1 > best_fitness:
                    best_fitness = val_f1

                # Save model
                save = (not opt.nosave) or (final_epoch and not opt.evolve)
                if save:
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'state_dict_model': self.model.state_dict(),
                            'state_dict_optimizer': None if final_epoch else self.optimizer.state_dict(),
                            'nep_id': nep_id if nep_writer else None}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == val_f1:
                        torch.save(ckpt, best)
                    del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------

        # end training
        dist.destroy_process_group() if self.rank not in [-1, 0] else None
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='input/configs/military_all.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='input/configs/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=6, help='maximum number of dataloader workers')
    parser.add_argument('--use_neptune_log', type=bool, default=True, help='Use neptune.ai as a logger')
    parser.add_argument('--neptune_params', type=str, default='input/configs/neptune_settings.json',
                        help='neptune params file path')

    opt = parser.parse_args('--name net_32_64_128_256_512_1024_convs_diff_ep_100_amp --single-cls'.split())

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.weights, opt.resume = ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)  # check files

        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        seed_everything(hyp['seed'])

    # Train
    logger.info(opt)
    tb_writer, nep_writer = None, None  # init loggers
    tb_writer = SummaryWriter('runs/test_run')
    opt.log_dir = log_dir
    if opt.global_rank in [-1, 0]:
        # neptune
        try:
            import neptune

            # set neptune logger
            if opt.use_neptune_log:
                PARAMS = dict(hyp, **vars(opt))
                nep_writer = True

                with open(opt.neptune_params) as json_file:
                    np_params = json.load(json_file)

                api_token = np_params["API_token"] if np_params["API_token"] != 0 else "ANONYMOUS"
                full_proj_name = f'{np_params["user"]}/{np_params["proj_name"]}' if np_params["user"] != 0 else None

                assert api_token != "ANONYMOUS" or full_proj_name != None, "set corrected neptune's user and project name"

                np_params["tags"] = np_params["tags"]
                neptune.init(full_proj_name, api_token=api_token)
                experiment = neptune.create_experiment(name=np_params["experiment_name"],
                                                       params=PARAMS,
                                                       upload_source_files=np_params["source"],
                                                       tags=np_params["tags"]
                                                       )
                nep_id = experiment.id  # uniq ID token from neptune
                opt.log_dir = log_dir + '_' + nep_id

        except (ImportError, AssertionError):
            opt.log_imgs = 0
            logger.info("Install neptune for experiment logging via 'pip install neptune' (recommended)")

    train_model = TrainProcess()
    train_model.run()

    if opt.use_neptune_log:
        try:
            neptune.stop()
        except (ImportError, AssertionError):
            opt.log_imgs = 0
            logger.info("Install neptune for experiment logging via 'pip install neptune' (recommended)")
