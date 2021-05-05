import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary


class ModelLSTM(nn.Module):
    def __init__(self, input_size=1, blocks_size=[30, 180, 360, 540], hidden_dim=64, lstm_layers=2, dropout=0.2,
                 bidirectional=False,
                 output_size=2, activation='mish'):
        super().__init__()
        self.bn = nn.BatchNorm1d(blocks_size[-1])
        self.lstm = nn.LSTM(blocks_size[-1], hidden_dim, lstm_layers, bidirectional=bidirectional, dropout=dropout)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 4, output_size)
        else:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        #self.fc = nn.Linear(blocks_size[-1] * 62, output_size)
        self.IncConvEncoder = Encoder(input_size, blocks_size, dropout, activation)
        self.flag = False
        # if bidirectional:
        #     self.prev_data = torch.rand(64, hidden_dim * 2).to(device='cuda')
        # else:
        #     self.prev_data = torch.rand(64, hidden_dim).to(device='cuda')


    def forward(self, RCS, prev_data):
        batch_size = RCS.size(0)
        x1 = self.IncConvEncoder(RCS)
        #print(x.shape)
        x1 = self.bn(x1)
        x1 = x1.permute(2, 0, 1)
        #print(x.shape)
        x1, hidden = self.lstm(x1)
        #print(x.shape)

        x1 = x1.permute(1, 0, 2)
        # x = torch.mean(x, dim=1).view((batch_size, -1))
        x1 = x1[:, -1, :].view((batch_size, -1))
        #print(x.shape)
        #x2 = x.clone()
        if self.flag:
            x = torch.cat([x1, prev_data], dim=1)
        else:
            x = torch.cat([x1, x1.detach()], dim=1)
            self.flag = True
        #print(x.shape)

        out = self.fc(x)

        #out = self.fc(x.view(-1, x.shape[1]*x.shape[2]))
        #print(out[:, 0].shape)
        return out, x1


class Encoder(nn.Module):
    def __init__(self, input_size=1, blocks_sizes=[30, 180, 360, 540], dropout=0.2, activation='relu', *args, **kwargs):
        super().__init__()
        kwargs['activation'] = activation
        self.blocks_sizes = blocks_sizes.insert(0, input_size)
        self.blocks = nn.ModuleList(
            [nn.Sequential(IncConv(blocks_sizes[0], blocks_sizes[1], cn_blocks=[201, 101, 15], *args, **kwargs),
                           GeM(),
                           Mish(),
                           nn.BatchNorm1d(blocks_sizes[1]),
                           nn.Dropout(dropout),
                           ),
             nn.Sequential(
                 IncConv(blocks_sizes[1], blocks_sizes[2], cn_blocks=[101, 51, 9], *args, **kwargs),
                 GeM(),
                 Mish(),
                 nn.BatchNorm1d(blocks_sizes[2]),
                 nn.Dropout(dropout),
                 ),
             nn.Sequential(
                 IncConv(blocks_sizes[2], blocks_sizes[3], cn_blocks=[51, 27, 7], *args, **kwargs),
                 GeM(),
                 Mish(),
                 nn.BatchNorm1d(blocks_sizes[3]),
                 nn.Dropout(dropout),
                 ),
             nn.Sequential(
                 IncConv(blocks_sizes[3], blocks_sizes[4], cn_blocks=[27, 13, 5], *args, **kwargs),
                 GeM(),
                 Mish(),
                 nn.BatchNorm1d(blocks_sizes[4]),
                 nn.Dropout(dropout),
                 ),
             nn.Sequential(
                 IncConv(blocks_sizes[4], blocks_sizes[5], cn_blocks=[13, 7, 3], *args, **kwargs),
                 GeM(),
                 Mish(),
                 nn.BatchNorm1d(blocks_sizes[5]),
                 nn.Dropout(dropout),
                 ),
             nn.Sequential(
                 IncConv(blocks_sizes[5], blocks_sizes[6], cn_blocks=[7, 3, 1], *args, **kwargs),
                 GeM(),
                 Mish(),
                 nn.BatchNorm1d(blocks_sizes[6]),
                 nn.Dropout(dropout),
             )
             ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            #print(x.shape)
        return x


class IncConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, cn_blocks=[201, 101, 15], *args, **kwargs):
        super(IncConv, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, out_channels // 3, kernel_size=cn_blocks[0], padding=int((cn_blocks[0]-1)/2), *args, **kwargs)

        self.branch5x5_1 = conv_block(in_channels, out_channels // 3, kernel_size=cn_blocks[1], padding=int((cn_blocks[1] - 1)/2),
                                      dilation=1, *args, **kwargs)
        self.branch5x5_2 = conv_block(out_channels // 3, out_channels // 3, kernel_size=cn_blocks[1], padding=int((cn_blocks[1] - 1)/2),
                                      dilation=1, *args, **kwargs)

        out_channels_mod = out_channels % 3
        self.branch3x3dbl_1 = conv_block(in_channels, out_channels // 3 + out_channels_mod,
                                         kernel_size=cn_blocks[2], padding=int((cn_blocks[2] - 1)/2), dilation=1, *args, **kwargs)
        self.branch3x3dbl_2 = conv_block(out_channels // 3 + out_channels_mod,
                                         out_channels // 3 + out_channels_mod,
                                         kernel_size=cn_blocks[2], padding=int((cn_blocks[2] - 1)/2), dilation=1, *args, **kwargs)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)
        #print("1", branch1x1.shape)

        branch5x5 = self.branch5x5_1(x)
        #print("2", branch5x5.shape)
        branch5x5 = self.branch5x5_2(branch5x5)
        #print("3", branch5x5.shape)
        branch3x3dbl = self.branch3x3dbl_1(x)
        #print("4", branch3x3dbl.shape)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        #print("5", branch3x3dbl.shape)

        outputs = [branch1x1, branch5x5, branch3x3dbl]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)

        return outputs


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky_relu', **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)
        self.activate = activation_func(activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['mish', Mish()],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), 2).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


if __name__ == "__main__":
    """ Тестирование сети """
    writer = SummaryWriter('runs/test')
    prev_data = torch.rand(64, 64 * 2)

    model = ModelLSTM(input_size=1, output_size=1, blocks_size=[16, 16, 16, 16, 16, 16], hidden_dim=64, bidirectional=True)
    print(model.flag)
    model.train()

    x = torch.rand((64, 1, 5000))
    #print(x)
    out = model(x, prev_data)
    print(model.flag)
    out = model(x, prev_data)
    print(model.flag)
    # for f in model.parameters():
    #     hist_name = 'hist' + str(list(f.grad.data.size()))
    #     writer.add_histogram(hist_name, f)

    soft = nn.Softmax(0)
    #print(soft(out))
    print(out.shape)
    #print(model)
    #print(summary(model, x))

    """ Сохранить параметры модели """
    # model_log.to_csv(r'Output.txt', sep=' ', mode='a')
    # pd.set_option('display.max_columns', None)
    # with open("Output.txt", "w") as text_file:
    #     text_file.write(str(model_log))
