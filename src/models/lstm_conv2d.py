import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelLSTM(nn.Module):
    def __init__(self, input_size=1, output_size=14, blocks_sizes=[30, 180, 360, 540], dropout=0.2, activation='relu',
                 hidden_dim=128, lstm_layers=2, bidirectional=True, *args, **kwargs):
        super().__init__()
        kwargs['activation'] = activation
        self.blocks_sizes = blocks_sizes.insert(0, input_size)
        self.blocks = nn.ModuleList(
            [nn.Sequential(IncConv(blocks_sizes[block], blocks_sizes[block + 1], *args, **kwargs),
                           nn.BatchNorm1d(blocks_sizes[block + 1]))
             for block in range(len(blocks_sizes) - 1)]
        )
        self.lstm_blocks_sizes = blocks_sizes.copy()
        self.lstm_blocks_sizes.append(blocks_sizes[-1] * 2)
        del self.lstm_blocks_sizes[0]
        self.lstm_blocks = nn.ModuleList([nn.Sequential(nn.LSTM(self.lstm_blocks_sizes[block], self.lstm_blocks_sizes
            [block + 1] // 2, lstm_layers, dropout=dropout)) for block in range(len(self.lstm_blocks_sizes) - 1)])
        self.fc = nn.Linear(self.lstm_blocks_sizes[-1] // 2, output_size)
        self.log_softmax = nn.Softmax(dim=1)  # nn.LogSoftmax(dim=0)

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.unsqueeze(x, 1)
        for idx_block in range(len(self.blocks)):
            x = self.blocks[idx_block](x)
            x = x.permute(2, 0, 1)
            x, _ = self.lstm_blocks[idx_block](x)
            x = x.permute(1, 2, 0)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=1).view((batch_size, -1))  # x[:, -1, :].view((...
        x = self.fc(x)
        return self.log_softmax(x)


class ModelLSTM_conv2d(nn.Module):
    def __init__(self, input_size=1, blocks_size=[30, 180, 360, 540], hidden_dim=64, lstm_layers=2, dropout=0.2,
                 bidirectional=True,
                 output_size=2, activation='leaky_relu'):
        super().__init__()
        self.lstm = nn.LSTM(blocks_size[-1], hidden_dim, lstm_layers, bidirectional=bidirectional, dropout=dropout)
        self.IncConvEncoder = Encoder(input_size, blocks_size, dropout, activation)
        # if bidirectional:
        #     self.fc = nn.Linear(hidden_dim // 4 * 2, output_size)
        # else:
        # self.Encoder2D = BasicConv2d(1, 5)
        # self.fc_ = nn.Linear(hidden_dim * 2, output_size)
        self.lstm2 = nn.LSTM(288, hidden_dim, lstm_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.Conv2d = InceptionA(hidden_dim, hidden_dim // 2)

    def forward(self, xs):
        bs, dim, length = xs[0].size()
        x = torch.cat(xs, dim=0)

        x = self.IncConvEncoder(x)
        x = x.permute(2, 0, 1)
        x, hidden = self.lstm(x)
        x = x.permute(1, 0, 2)
        # all_x.append(x)

        # x_ = torch.mean(x, dim=1)
        # x_ = x_.reshape(dim, bs, -1)
        # x_ = x_.reshape(bs * dim, -1)
        # out_1 = self.fc_(x_)
        # all_out_1.append(out_1)

        # x = torch.stack(all_x, dim=3)
        x = x.reshape(len(xs), bs, -1, length)
        x = x.permute(1, 2, 3, 0)
        x = self.Conv2d(x)
        x = torch.mean(x, dim=3)
        x = x.permute(2, 0, 1)
        x, hidden = self.lstm2(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((bs, -1))
        out = self.fc2(x)
        # out_1 = torch.cat(all_out_1, 0)
        return out  # out_1


class Model2d(nn.Module):
    def __init__(self,  hidden_dim=64,  bidirectional=True, output_size=2, lstm_layers=2, dropout=0.2):
        super().__init__()
        if bidirectional:
            self.fc = nn.Linear(hidden_dim // 4 * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim // 4, output_size)
        self.Conv2d = InceptionA(hidden_dim, hidden_dim // 2)
        self.lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 4, lstm_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.Conv2d(x)
        x = x.permute(2, 0, 1)
        x, hidden = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.fc(x)
        return out


class ModelLSTM2(nn.Module):
    def __init__(self, input_size=1, blocks_sizes=[30, 180, 360, 540], hidden_dim=16, lstm_layers=2, dropout=0.2,
                 bidirectional=False,
                 output_size=2, activation='leaky_relu'):
        super().__init__()
        self.lstm = nn.LSTM(blocks_sizes[-1], hidden_dim, lstm_layers, bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
        self.IncConvEncoder = Encoder(input_size, blocks_sizes, dropout, activation)

    def forward(self, x, mode='train'):
        if mode == 'test':
            x = torch.cat(x, dim=0)
        else:
            x = torch.unsqueeze(x, 1)
        batch_size = x.size()[0]
        seq_len = x.size()[2]
        x = self.IncConvEncoder(x)
        x = x.view((seq_len, batch_size, -1))
        x, hidden = self.lstm(x)
        x = x.view((batch_size, seq_len, -1))
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.fc(x)
        if mode == 'test':
            out = out.reshape(5, batch_size // 5, -1)
            out = torch.mean(out, dim=0)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size=1, blocks_sizes=[30, 180, 360, 540], dropout=0.2, activation='relu', *args, **kwargs):
        super().__init__()
        kwargs['activation'] = activation
        self.blocks_sizes = blocks_sizes.insert(0, input_size)
        self.blocks = nn.ModuleList(
            [nn.Sequential(IncConv(blocks_sizes[block], blocks_sizes[block + 1], *args, **kwargs),
                           nn.Dropout(dropout))
             for block in range(len(blocks_sizes) - 1)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class IncConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, *args, **kwargs):
        super(IncConv, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, out_channels // 3, kernel_size=1, *args, **kwargs)

        self.branch5x5_1 = conv_block(in_channels, out_channels // 3, kernel_size=5, padding=2, *args, **kwargs)
        self.branch5x5_2 = conv_block(out_channels // 3, out_channels // 3, kernel_size=5, padding=2, *args, **kwargs)

        out_channels_mod = out_channels % 3
        self.branch3x3dbl_1 = conv_block(in_channels, out_channels // 3 + out_channels_mod,
                                         kernel_size=3, padding=1, *args, **kwargs)
        self.branch3x3dbl_2 = conv_block(out_channels // 3 + out_channels_mod,
                                         out_channels // 3 + out_channels_mod,
                                         kernel_size=3, padding=1, *args, **kwargs)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch5x5, branch3x3dbl]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


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
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == "__main__":
    """ Тестирование сети """
    model = ModelLSTM(input_size=1, output_size=15)
    # print(model)

    # model_log = summary(model, torch.zeros((10, 1, 100)))

    """ Сохранить параметры модели """
    # model_log.to_csv(r'Output.txt', sep=' ', mode='a')
    # pd.set_option('display.max_columns', None)
    # with open("Output.txt", "w") as text_file:
    #     text_file.write(str(model_log))