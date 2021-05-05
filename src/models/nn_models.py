import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


class ModelLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        input_size – Количество характеристик, ожидаемых на входном объекте
        output_size - Количество характеристик на выходном слое
        hidden_dim – Количество характеристик в скрытом слое
        n_layers – Количество рекуррентных слоев
        '''
        super(ModelLSTM, self).__init__()
        self.hidden_dim = 64  # размер скрытого слоя в LSTM слоя
        self.n_layers = 2  # количество слоев LSTM слоя

        self.encoder_1 = IncConv(1, 10)
        # self.encoder_1_2 = IncConv(30, 10)
        # self.encoder_1_3 = IncConv(30, 10)
        self.drop_1 = nn.Dropout(0.2)

        self.encoder_2 = IncConv(30, 20 * 3)
        # self.encoder_2_2 = IncConv(180, 20 * 3)
        # self.encoder_2_3 = IncConv(180, 20 * 3)
        self.drop_2 = nn.Dropout(0.2)

        self.encoder_3 = IncConv(180, 40 * 3)
        # self.encoder_3_2 = IncConv(360, 40 * 3)
        # self.encoder_3_3 = IncConv(360, 40 * 3)
        self.drop_3 = nn.Dropout(0.2)

        self.encoder_4 = IncConv(360, 60 * 3)
        # self.encoder_4_2 = IncConv(540, 60 * 3)
        # self.encoder_4_3 = IncConv(540, 60 * 3)
        self.drop_4 = nn.Dropout(0.2)

        # self.encoder_5 = IncConv(540, 80 * 3)
        # self.encoder_5_2 = IncConv(720, 80 * 3)
        # self.encoder_5_3 = IncConv(720, 80 * 3)
        # self.drop_5 = nn.Dropout(0.2)

        self.rnn = nn.LSTM(540, self.hidden_dim, self.n_layers, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(self.hidden_dim * 2, output_size)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.encoder_1(x)
        residual = x
        # x = self.encoder_1_2(x)
        # x = self.drop_1(x)
        # x = self.encoder_1_3(x) + residual

        x = self.encoder_2(x)
        residual = x
        # x = self.encoder_2_2(x)
        # x = self.drop_2(x)
        # x = self.encoder_2_3(x) + residual

        x = self.encoder_3(x)
        residual = x
        # x = self.encoder_3_2(x)
        # x = self.drop_3(x)
        # x = self.encoder_3_3(x) + residual

        x = self.encoder_4(x)
        # residual = x
        # x = self.encoder_4_2(x)
        # x = self.drop_4(x)
        # x = self.encoder_4_3(x) + residual

        # x = self.encoder_5(x)

        x = x.view((100, batch_size, -1))
        x, hidden = self.rnn(x)
        x = x.view((batch_size, 100, -1))
        x = torch.mean(x, dim=1).view((batch_size, -1))
        out = self.fc(x)
        out = F.softmax(out, 1)
        return out, hidden


class IncConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=None, *args, **kwargs):
        super(IncConv, self).__init__()
        if conv_block is None:
            conv_block = BasicConv1d
        self.branch1x1 = conv_block(in_channels, out_channels, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch5x5_2 = conv_block(out_channels, out_channels, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3x3dbl_2 = conv_block(out_channels, out_channels, kernel_size=3, padding=1)

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
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return nn.LeakyReLU(negative_slope=0.02, inplace=True)(x)


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='leaky_relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        # print(x.size(), residual.size())
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetBasicBlock(ResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling = expansion, downsampling

        self.shortcut = nn.Sequential(
            nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm1d(self.expanded_channels)) if self.should_apply_shortcut else None

        self.blocks = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, bias=False,
                      stride=self.downsampling, padding=2),
            activation_func(self.activation),
            nn.Dropout(0.4),
            nn.Conv1d(self.out_channels, self.expanded_channels, kernel_size=3, bias=False),
            # nn.BatchNorm1d(self.expanded_channels)
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2, 2, 2, 2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=1, padding=3, bias=False),
            # nn.BatchNorm1d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        # print(x.size())
        x = self.gate(x)
        # print(x.size())
        for block in self.blocks:
            x = block(x)
        return x


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()

        self.input_dim = 1
        self.output_dim = 15

        self.rnn_layers = 2
        self.hidden_dim = 128

        self.blocks_sizes = [64, 128, 256]
        self.deepths = [2, 2, 2]
        self.encoder = ResNetEncoder(in_channels=self.input_dim,
                                     blocks_sizes=self.blocks_sizes, deepths=self.deepths)
        self.rnn = nn.LSTM(input_size=self.blocks_sizes[-1], hidden_size=self.hidden_dim,
                           num_layers=self.rnn_layers, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        # x.size() = [batch_size, input_dim, len_dim]
        x = self.encoder(x)
        # for lstm x.size() = [len_dim, batch_size, input_dim]
        x = x.permute(2, 0, 1)
        x, hidden = self.rnn(x, None)
        # x.size() = [batch_size, input_dim, len_dim]
        x = x.permute(1, 2, 0)
        x = torch.mean(x, dim=2).view((batch_size, -1))
        out = self.fc(x)
        out = F.softmax(out, 1)
        return out, hidden


if __name__ == "__main__":
    from torchsummaryX import summary
    # res_model = ResNetBasicBlock(1, 15)
    # res_model = ResNetLayer(1, 64, block=ResNetBasicBlock, n=3)
    # res_model = ResNetEncoder(in_channels=1, blocks_sizes=[64, 128, 256], deepths=[2, 2, 2])
    # print(res_model)

    # model = EncoderRNN()
    model = ModelLSTM(1, 15)
    # print(model)

    model_log = summary(model, torch.zeros((10, 1, 100)))
    # import pandas as pd
    # model_log.to_csv(r'Output.txt', sep=' ', mode='a')
    # pd.set_option('display.max_columns', None)
    # with open("Output.txt", "w") as text_file:
    #     text_file.write(str(model_log))


    input_test = torch.zeros((10, 1, 100))
    output = model(input_test)

    print(output[0].size())