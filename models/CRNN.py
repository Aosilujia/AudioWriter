import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, cirH, channel_input, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        channel_sizes=[64, 128, 256, 256, 512, 512, 512]
        kernel_sizes=[3, 3, 3, 3, 3, 3, 2]
        strides=[1, 1, 1, 1, 1, 1, 1]
        paddings=[1, 1, 1, 1, 1, 1, 0]


        cnn=nn.Sequential

        def convRelu(i, batchNormalization=False):
            nIn = channel_input if i == 0 else channel_sizes[i - 1]
            nOut = channel_sizes[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_sizes[i], strides[i], paddings[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

            convRelu(0)

            convRelu(1)

            convRelu(2)

            convRelu(3)

            convRelu(4)

            convRelu(5)

            convRelu(6)
