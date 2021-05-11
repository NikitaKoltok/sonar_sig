import torch
import cv2
import numpy as np
import time
import torch.nn as nn
import os
import matplotlib.pyplot as plt

from src.models.dist_lstm import ModelLSTM


class TrainModel(object):
    """
    Обучение модели в режиме mode = 'train'
    или тестирование модели в режиме 'test'
    с подгрузкой весов из существующего файла формата .pth
    """
    def __init__(self):
        # классы ЛА
        self.models = ['object']

        self.num_classes = len(self.models)

        self.device = torch.device('cuda')  # устройство для обучения
        self.is_pretrained = True  # флаг: True - модель предобучена, False - в противном случае

        # параметры модели
        self.encoder_size = [16, 32, 64, 128, 256, 512]
        self.encoder_dropout = 0.3
        self.lstm_layers = 2
        self.lstm_hidden_size = 64
        self.lstm_dropout = 0.1
        self.is_bilstm = True

    @property
    def model(self):
        model = ModelLSTM(input_size=1, output_size=self.num_classes,
                          blocks_size=self.encoder_size, dropout=self.encoder_dropout,
                          hidden_dim=self.lstm_hidden_size, lstm_layers=self.lstm_layers,
                          bidirectional=self.is_bilstm)

        #model = model.to(self.device)

        if self.is_pretrained:
            checkpoint = torch.load('runs/exp155_net_16_32_64_128_256_512_2_series_10_per_obj_SIG-307/weights/last.pt')
            #print(checkpoint)
            model.load_state_dict(checkpoint['state_dict_model'])


        return model.eval()


if __name__ == '__main__':
    train_model = TrainModel().model
    prev_data = torch.rand((1, 128))

    # res = np.zeros((804, 30))
    #
    # file = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_5000_grand/test/new_test/04.09-02.starboard.bin', dtype=np.float)
    # img = cv2.imread('/home/koltokng/LSTM/signal_labels_new/patches_5000_grand/test/new_test/04.09-02.starboard.png', cv2.IMREAD_GRAYSCALE)
    # #re_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/presentation/29/29_re.bin', dtype=np.float)
    # #im_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/presentation/29/29_im.bin', dtype=np.float)
    # file = np.reshape(file, img.shape)
    # #re_signal = np.reshape(re_signal, img.shape)
    # #im_signal = np.reshape(im_signal, img.shape)
    # print(file.shape)
    # start = time.time()
    # for i in range(804):
    #     row = file[i]
    #     #re_row = re_signal[i]
    #     #im_row = im_signal[i]
    #     #patch = np.zeros(5000)
    #
    #     patch = row[0:5000]
    #     #re_patch = re_row[200:5000]
    #     #im_patch = im_row[200:5000]
    #     #print(len(patch))
    #     #patch_num = int((len(row)/300) * 1.25)
    #     #start_point = 0
    #     #res_row = np.zeros(300 + 225*(patch_num-1))
    #     # for m in range(patch_num):
    #     #     final_row = torch.tensor(row[start_point:start_point + 300]).type(torch.FloatTensor)
    #     #     final_row = torch.unsqueeze(final_row, 0)
    #     #     final_row = torch.unsqueeze(final_row, 0)
    #     #     pred = train_model(final_row)
    #     #     res_row[m * 75:m * 75 + 300] = np.int(pred.sigmoid().round()[0][0])
    #     #     start_point += 75
    #     #final_re_signal = torch.tensor(re_patch).type(torch.FloatTensor)
    #     #final_im_signal = torch.tensor(im_patch).type(torch.FloatTensor)
    #     final_row = torch.tensor(patch).type(torch.FloatTensor)
    #     final_row = torch.unsqueeze(final_row, 0)
    #     #final_re_signal = torch.unsqueeze(final_re_signal, 0)
    #     #final_im_signal = torch.unsqueeze(final_im_signal, 0)
    #
    #     #final_row = torch.cat((final_re_signal, final_im_signal), 0)
    #     final_row = torch.unsqueeze(final_row, 0)
    #     #pred, data = train_model(final_row, prev_data)
    #     pred = train_model(final_row)
    #     res[i] = np.int(pred.sigmoid().round()[0][0])
    #     #prev_data = data.detach()

    # stop = time.time()
    # res = np.asarray(res)
    # print(stop - start)
    #cv2.imshow('asd', res)
    #cv2.imwrite('runs/exp149_net_16_16_32_32_64_64_convs_diff_ep_50_amp_SIG-225/res.png', res * 255)
    #cv2.waitKey(0)
    #res.tofile('/home/koltokng/LSTM/signal_labels_new/presentation/29/res.bin')
    # plt.plot(np.arange(804), res)
    # plt.show()

    sum = 0
    for name in os.listdir('/media/koltokng/Новый том/hyscan/big_size_dataset/sonar_dataset/dataset_obj/test/objects/amp'):
        res = []
        #work_pic = cv2.imread('../../pics_order/157.png', cv2.IMREAD_GRAYSCALE)
        #row = work_pic[14].astype(np.float) / 255.
        amp_signal = np.fromfile('/media/koltokng/Новый том/hyscan/big_size_dataset/sonar_dataset/dataset_obj/test/objects/amp/' + name)
        #re_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_5000/test_patches/re/' + name)
        #im_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_5000/test_patches/im/' + name)
        #spec = np.fft.fft(re_signal + 1j * im_signal)
        #spec /= np.amax(spec)

        final_row = torch.tensor(amp_signal).type(torch.FloatTensor)
        #final_re_signal = torch.tensor(re_signal).type(torch.FloatTensor)
        #final_im_signal = torch.tensor(im_signal).type(torch.FloatTensor)
        #final_spec = torch.tensor(spec).type(torch.FloatTensor)

        #final_re_signal = torch.unsqueeze(final_re_signal, 0)
        #final_im_signal = torch.unsqueeze(final_im_signal, 0)
        #final_spec = torch.unsqueeze(final_spec, 0)
        final_row = torch.unsqueeze(final_row, 0)

        #final_sig = torch.cat((final_re_signal, final_im_signal), 0)
        final_row = torch.unsqueeze(final_row, 0)
        #print(final_sig.shape, name)

        pred, data = train_model(final_row, prev_data)
        #pred = train_model(final_row)
        prev_data = data.detach()
        sum += pred.sigmoid()[0][0].round()
        #soft = nn.Softmax(dim=1)
        #print(name, pred.sigmoid()[0][0].round(), '\n')
        #rint(pred)
    print('True objects: ', sum/100)
    print('False background: ', (100 - sum)/100)

    sum = 0
    for name in os.listdir('/media/koltokng/Новый том/hyscan/big_size_dataset/sonar_dataset/dataset_obj/test/background/amp'):
        res = []
        #work_pic = cv2.imread('../../pics_order/157.png', cv2.IMREAD_GRAYSCALE)
        #row = work_pic[14].astype(np.float) / 255.
        amp_signal = np.fromfile('/media/koltokng/Новый том/hyscan/big_size_dataset/sonar_dataset/dataset_obj/test/background/amp/' + name)
        #re_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_5000/test_patches/re/' + name)
        #im_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_5000/test_patches/im/' + name)
        #spec = np.fft.fft(re_signal + 1j * im_signal)
        #spec /= np.amax(spec)

        final_row = torch.tensor(amp_signal).type(torch.FloatTensor)
        #final_re_signal = torch.tensor(re_signal).type(torch.FloatTensor)
        #final_im_signal = torch.tensor(im_signal).type(torch.FloatTensor)
        #final_spec = torch.tensor(spec).type(torch.FloatTensor)

        #final_re_signal = torch.unsqueeze(final_re_signal, 0)
        #final_im_signal = torch.unsqueeze(final_im_signal, 0)
        #final_spec = torch.unsqueeze(final_spec, 0)
        final_row = torch.unsqueeze(final_row, 0)

        #final_sig = torch.cat((final_re_signal, final_im_signal), 0)
        final_row = torch.unsqueeze(final_row, 0)
        #print(final_sig.shape, name)

        pred, data = train_model(final_row, prev_data)
        #pred = train_model(final_row)
        prev_data = data.detach()
        #print(pred.sigmoid().round())
        sum += pred.sigmoid()[0][0].round()
        #soft = nn.Softmax(dim=1)
        #print(name, pred.sigmoid()[0][0].round(), '\n')
        #rint(pred)
    print('False background: ', sum/100)
    print('True background: ', (100 - sum)/100)

    '''
    amp_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_1000_2/test/amp/627.bin')
    re_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_1000_2/test/re/627.bin')
    im_signal = np.fromfile('/home/koltokng/LSTM/signal_labels_new/patches_1000_2/test/im/627.bin')
    spec = np.fft.fft(re_signal + 1j * im_signal)

    final_row = torch.tensor(amp_signal).type(torch.FloatTensor)
    final_re_signal = torch.tensor(re_signal).type(torch.FloatTensor)
    final_im_signal = torch.tensor(im_signal).type(torch.FloatTensor)
    final_spec = torch.tensor(spec).type(torch.FloatTensor)

    final_re_signal = torch.unsqueeze(final_re_signal, 0)
    final_im_signal = torch.unsqueeze(final_im_signal, 0)
    final_spec = torch.unsqueeze(final_spec, 0)
    final_row = torch.unsqueeze(final_row, 0)

    final_sig = torch.cat((final_row, final_re_signal, final_im_signal, final_spec), 0)
    final_sig = torch.unsqueeze(final_sig, 0).to("cuda")
    print(final_sig.shape)

    pred = train_model.model(final_sig)
    # soft = nn.Softmax(dim=1)
    print(pred.sigmoid().round())
    '''