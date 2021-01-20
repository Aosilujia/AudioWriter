import torch.nn as nn
import torch.nn.functional as functional

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):

    def __init__(self, cirH, channel_input, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()

        channel_sizes=[64, 128, 256, 256, 512, 512, 512]
        kernel_sizes=[3, 3, 3, 3, 3, 3, 2]
        strides=[1, 1, 1, 1, 1, 1, 1]
        paddings=[1, 1, 1, 1, 1, 1, 0]


        cnn=nn.Sequential()

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

        convRelu(0,True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1,True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2,True)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2)))  # 256x4x16
        convRelu(3)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2)))  # 256x4x16
        convRelu(4, True)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2)))  # 256x4x16
        convRelu(5)
        cnn.add_module('pooling{0}'.format(5),
                       nn.MaxPool2d((2, 2), (1, 2), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, w, h = conv.size()
        assert h == 1, "the height of conv must be 1,now is {},width is {}".format(h,w)
        conv = conv.squeeze(3) #remove the height
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        output = functional.log_softmax(output, dim=2)

        return output


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero
