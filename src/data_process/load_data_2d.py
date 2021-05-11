import os
import numpy as np
import random
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from src.models.lstm_conv2d import ModelLSTM_conv2d
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class DatasetCreator(Dataset):
    '''
    Создает датасет для его передачи в нейронную сеть
    '''

    def __init__(self, pics_root_dir, labels_root_dir, obj_percentage=20,
                 amp_data=True, complex_data=False, patch_size=1, transform=None):
        self.pics_root_dir = pics_root_dir
        self.labels_root_dir = labels_root_dir
        self.amp_data = amp_data
        self.complex_data = complex_data
        self.obj_percentage = obj_percentage/100
        self.spectre = False
        self.num_cls = 1
        self.patch_size = patch_size
        self.counter = 0
        self.prev_num = 0

    def __len__(self):
        return len(os.listdir(self.pics_root_dir))

    def __getitem__(self, item):
        '''
        Возвращает пару формата (torch tensor, int), содержащую данные RCS и соотвествующей метки класса ЛА
        '''
        #pic_num = item // 1406
        #row_idx = item % 512

        labels = np.fromfile(self.labels_root_dir + 'labels.bin', dtype=np.float)
        eps = random.random()

        if eps < self.obj_percentage:
            item = int(random.choice(np.arange(list(labels).index(0) - 1 - self.patch_size)))

        else:
            item = int(random.choice(np.arange(list(labels).index(0), len(labels) - self.patch_size)))

        label = float(labels[item])

        if self.amp_data:
            amp_signal = np.fromfile(self.pics_root_dir + 'amp/' + str(item) + ".bin", dtype=np.float)
            final_signal = torch.tensor(amp_signal).type(torch.FloatTensor)
            final_signal = torch.unsqueeze(final_signal, 0)
            for m in range(self.patch_size - 1):
                amp_signal_1 = np.fromfile(self.pics_root_dir + 'amp/' + str(item + m + 1) + ".bin", dtype=np.float)
                final_signal_1 = torch.tensor(amp_signal_1).type(torch.FloatTensor)
                final_signal_1 = torch.unsqueeze(final_signal_1, 0)

                final_signal = torch.cat((final_signal, final_signal_1), dim=0)
            final_signal = torch.unsqueeze(final_signal, 0)

        if self.complex_data:
            re_signal = np.fromfile(self.pics_root_dir + 're/' + str(item) + ".bin", dtype=np.float)
            im_signal = np.fromfile(self.pics_root_dir + 'im/' + str(item) + ".bin", dtype=np.float)

            final_re_signal = torch.tensor(re_signal).type(torch.FloatTensor)
            final_im_signal = torch.tensor(im_signal).type(torch.FloatTensor)
            final_re_signal = torch.unsqueeze(final_re_signal, 0)
            final_im_signal = torch.unsqueeze(final_im_signal, 0)
            final_row = torch.cat((final_re_signal, final_im_signal), 0)

        if self.spectre:
            spec = np.abs(np.fft.fft(re_signal + 1j * im_signal))
            spec /= np.amax(spec)

            final_spec = torch.tensor(spec).type(torch.FloatTensor)
            final_spec = torch.unsqueeze(final_spec, 0)

        return (final_signal, label)


if __name__ == '__main__':
    db = DatasetCreator(pics_root_dir="../../signal_labels_new/patches_5000_grand/learning/",
                        labels_root_dir="../../signal_labels_new/patches_5000_grand/learning/",
                        obj_percentage=30,
                        amp_data=True, complex_data=False, patch_size=5)  # load database

    learning_labels = np.fromfile("../../signal_labels_new/patches_5000_grand/learning/labels.bin", dtype=np.float)
    train_idx = np.arange(len(learning_labels))

    train_indices = train_idx
    train_sampler = SubsetRandomSampler(train_indices)
    # DataLoader test
    loader = torch.utils.data.DataLoader(db, sampler=train_sampler, batch_size=1,
                                               shuffle=False, num_workers=4, drop_last=True)
    #loader = DataLoader(db, batch_size=64, num_workers=4, shuffle=True)
    # for i in tqdm(loader, total=len(loader)):
    #     pass
    model = ModelLSTM_conv2d(input_size=1, output_size=1, blocks_size=[16, 32, 64, 128, 256, 512])
    x, target = next(iter(loader))

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, bias=False, padding=0)
    print(conv(x).shape)

    print(x.shape, target)
    print(torch.sum(target)/0.64)
    #print(x[0][0][:20])
    #print(target.type())
    #print(x.type())
    out = model(x)
    print('score = ', accuracy_score(np.asarray(target), out.sigmoid().round().detach().numpy()))
    print(out.sigmoid().round())
    #print(target.shape, out[:, 0].shape)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9))(out[0][0], target[0])
    #loss = nn.CrossEntropyLoss()(out, target)
    #print(loss)
    #print(x.type())
    #print('image batch shape -', x.shape)
    #print('mask batch shape -', target.shape)
    #print(target)
    #print(target.max(0))
    #print(db.num_cls)
