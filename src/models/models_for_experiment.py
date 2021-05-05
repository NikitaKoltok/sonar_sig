import torch
import torch.nn as nn
import torch.nn.functional as F


class IncResA(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super().__init__()
        self.inc_res = nn.Sequential(InceptionA(input_size, 64), ResidualBlock(324, hidden_dim))
        self.inc_res_Linear = nn.Linear(100, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.inc_res(x)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.inc_res_Linear(x)
        return out


class IncResAux(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super().__init__()
        self.inc_res_aux = nn.Sequential(InceptionAux(input_size, 64), ResidualBlock(324, hidden_dim))
        self.inc_res_Linear_aux = nn.Linear(1, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.inc_res_aux(x)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.inc_res_Linear_aux(x)
        return out


class Inception(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.inc = InceptionA(input_size, 64)
        self.inc_Linear = nn.Linear(100, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.inc(x)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.inc_Linear(x)
        return out


class Residual(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super().__init__()
        self.res = ResidualBlock(input_size, hidden_dim)
        self.res_Linear = nn.Linear(100, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.res(x)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.res_Linear(x)
        return out


class LayerLSTM(nn.Module):
    def __init__(self, output_size, hidden_dim=64, bidirectional=False, n_layers=2):
        super().__init__()
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.is_bidirectional = bidirectional
        self.includeLSTM = nn.LSTM(288, self.hidden_dim, self.n_layers, bidirectional=bidirectional)
        self.includeLinear = nn.Linear(self.hidden_dim * 2 if bidirectional else self.hidden_dim, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * 2 if self.is_bidirectional else self.n_layers, batch_size, self.hidden_dim)
        return hidden.to('cuda:0'), hidden.to('cuda:0')

    def forward(self, x):
        batch_size = x.size()[0]
        hidden = self.init_hidden(batch_size)
        x = x.permute(2, 0, 1)
        x, hidden = self.includeLSTM(x, hidden)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.includeLinear(x)
        return out


class LayerRNN(nn.Module):
    def __init__(self, output_size, hidden_dim=64, bidirectional=False, n_layers=2):
        super().__init__()
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.is_bidirectional = bidirectional
        self.rnn = nn.RNN(1, self.hidden_dim, self.n_layers, bidirectional=bidirectional)
        self.includeLinear = nn.Linear(self.hidden_dim * 2 if bidirectional else self.hidden_dim, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * 2 if self.is_bidirectional else self.n_layers, batch_size, self.hidden_dim)
        return hidden.to('cuda:0'), hidden.to('cuda:0')

    def forward(self, x):
        batch_size = x.size()[0]
        hidden = self.init_hidden(batch_size)
        hidden = hidden[0].to('cuda:0')
        x = x.permute(2, 0, 1)
        x, hidden = self.rnn(x, hidden)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.includeLinear(x)
        return out


class IncResLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64, bidirectional=False, n_layers=2):
        super().__init__()
        self.hidden_dim, self.n_layers = hidden_dim, n_layers
        self.is_bidirectional = bidirectional
        self.inc_res = nn.Sequential(InceptionA(input_size, 64), ResidualBlock(324, hidden_dim))
        self.includeLinear = nn.Linear(self.hidden_dim * 2 if bidirectional else self.hidden_dim, output_size)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * 2 if self.is_bidirectional else self.n_layers, batch_size, self.hidden_dim)
        return hidden.to('cuda:0'), hidden.to('cuda:0')

    def forward(self, x):
        batch_size = x.size()[0]
        hidden = self.init_hidden(batch_size)
        x = self.inc_res(x)
        x = x.permute(2, 0, 1)
        x, hidden = self.includeLSTM(x, hidden)
        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.includeLinear(x)
        return out


class Conv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Sequential(BasicConv1d(input_size, 64, kernel_size=3), nn.LeakyReLU())
        self.drop = nn.Dropout(0.2)
        self.conv2 = nn.Sequential(nn.Linear(2, 100), nn.LeakyReLU(), nn.Linear(100, 48), nn.Softmax())
        self.lin = nn.Linear(48, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        out = self.conv1(x)
        out = self.drop(out)
        out = F.adaptive_max_pool1d(out, 2)
        out = self.conv2(out)
        out = torch.mean(out, dim=1).view((batch_size, -1))
        out = self.lin(out)
        return out


class IncConvLSTMEncoder(nn.Module):
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
        self.log_softmax = nn.Softmax(dim=1)

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


class IncConvEncoderLSTM(nn.Module):
    def __init__(self, input_size=1, blocks_size=[30, 180, 360, 540], hidden_dim=64, lstm_layers=2, dropout=0.2,
                 bidirectional=True,
                 output_size=2, activation='mish'):
        super().__init__()
        self.lstm = nn.LSTM(blocks_size[-1], hidden_dim, lstm_layers, bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)
        self.IncConvEncoder = Encoder(input_size, blocks_size, dropout, activation)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.IncConvEncoder(x)

        x = x.permute(2, 0, 1)
        x, hidden = self.lstm(x)

        x = x.permute(1, 0, 2)
        x = torch.mean(x, dim=1).view((batch_size, -1))
        # x = x[:, -1, :].view((batch_size, -1))

        out = self.fc(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_size=1, blocks_sizes=[30, 180, 360, 540], dropout=0.2, activation='relu', *args, **kwargs):
        super().__init__()
        kwargs['activation'] = activation
        self.blocks_sizes = blocks_sizes.insert(0, input_size)
        self.blocks = nn.ModuleList(
            [nn.Sequential(IncConv(blocks_sizes[block], blocks_sizes[block + 1], *args, **kwargs),
                           GeM(),
                           Mish(),
                           nn.BatchNorm1d(blocks_sizes[block + 1]),
                           nn.Dropout(dropout),
                           )
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

        self.branch5x5_1 = conv_block(in_channels, out_channels // 3, kernel_size=5, padding=2,
                                      dilation=1, *args, **kwargs)
        self.branch5x5_2 = conv_block(out_channels // 3, out_channels // 3, kernel_size=5, padding=2,
                                      dilation=1, *args, **kwargs)

        out_channels_mod = out_channels % 3
        self.branch3x3dbl_1 = conv_block(in_channels, out_channels // 3 + out_channels_mod,
                                         kernel_size=3, padding=2, dilation=2, *args, **kwargs)
        self.branch3x3dbl_2 = conv_block(out_channels // 3 + out_channels_mod,
                                         out_channels // 3 + out_channels_mod,
                                         kernel_size=3, padding=2, dilation=2, *args, **kwargs)

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
        outputs = torch.cat(outputs, 1)

        return outputs


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool1d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


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
