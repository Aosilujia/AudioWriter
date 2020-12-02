import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, nclass, channel_input=1,leakyRelu=False):
        super(CNN, self).__init__()

        channel_sizes=[32, 64, 128, 256, 256, 512, 512]
        kernel_sizes=[3, 3, 3, 3, 3, 3, 3]
        strides=[1, 1, 1, 1, 1, 1, 1]
        paddings=[1, 1, 1, 1, 1, 1, 0]
        cnn=nn.Sequential()

        def convRelu(i,batchNormalization=False,relu=True,leakyRelu=False):
            nIn = channel_input if i == 0 else channel_sizes[i - 1]
            nOut = channel_sizes[i]
            cnn.add_module('conv{0}'.format(i), \
                           nn.Conv2d(nIn, nOut, kernel_sizes[i], strides[i], paddings[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),\
                               nn.LeakyReLU(0.2, inplace=True))
            if relu:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        def doubleConvRelu(i,batchNormalization=False,relu=True,leakyRelu=False):
            nIn = channel_input if i == 0 else channel_sizes[i - 1]
            nOut = channel_sizes[i]

            for n in (i*2,i*2+1):
                cnn.add_module('conv{0}'.format(n), \
                               nn.Conv2d(nIn, nOut, kernel_sizes[i], strides[i], paddings[i]))
                if batchNormalization:
                    cnn.add_module('batchnorm{0}'.format(n), nn.BatchNorm2d(nOut))
                if leakyRelu:
                    cnn.add_module('relu{0}'.format(n),\
                                   nn.LeakyReLU(0.2, inplace=True))
                if relu:
                    cnn.add_module('relu{0}'.format(n), nn.ReLU(True))

                nIn=nOut



        doubleConvRelu(0,True) #  *60
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # *30
        doubleConvRelu(1,True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 62*8
        doubleConvRelu(2,True) #15*2
        cnn.add_module('pooling{0}'.format(2),
                        nn.MaxPool2d((2, 2)))  # *3
        doubleConvRelu(3, True) #*1
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2)))  # 512x2x16
        doubleConvRelu(4, True) #*1
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2)))  # 512x2x16
        self.cnn=cnn

        full_connect=nn.Sequential()
        full_connect.add_module('fc1',nn.Linear(7*256,nclass))
        self.full_connect=full_connect

    def forward(self, input):
        # conv features
        convoutput = self.cnn(input)
        output = convoutput.reshape(convoutput.size(0),-1)

        output = self.full_connect(output)
        return output

"""pytorch official tutorial cnn """
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    print(CNN(14))
