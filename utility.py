import os
import queue
import random
from operator import itemgetter
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import matplotlib.pyplot as plt

def csvfilelist(directory_name):
    """
    得到一个csv文件路径列表
    :param directory_name
    """
    if not os.path.isdir(directory_name):
        print(directory_name+" is not a directory")
        return []
    filelist=[]
    dirqueue = queue.Queue()
    dirqueue.put(directory_name)
    while not dirqueue.empty():
        node = dirqueue.get()
        for filename in os.listdir(node):
            nextpath = os.path.join(node, filename)
            if os.path.isdir(nextpath):
                dirqueue.put(nextpath)
            elif nextpath.endswith('.csv'):
                filelist.append(nextpath)
    return filelist

def padding(raw_samples, max_length=-1, padding_value=0,padding_position=0) -> np.ndarray:
    """
    originated by yyg.对齐所有数据到最大长度或给定长度，输入为[N,C,W,H]
    :param padding_value:
    :param raw_samples: list of sample(numpy array)
    :param max_length: if max_length == -1,then we use max_len of raw_sample,else we use max_length passed in
    :param padding_position: 0 pad on right, 1 pad on left, 2 pad randomly
    :return:
    """
    shapes = list(map(np.shape, raw_samples))
    lengths = list(map(itemgetter(1),shapes))
    if max_length < 0:
        max_length = max(lengths)
    new_shape = [len(shapes),shapes[0][0],max_length,shapes[0][2]]
    padding_data = np.zeros(new_shape) + padding_value
    for idx, seq in enumerate(raw_samples):
        if padding_position==0:
            padding_data[idx,:,:lengths[idx]] = seq
        elif padding_position==1:
            padding_data[idx,:,-lengths[idx]:] = seq
        elif padding_position==2:
            padd_pos=0
            if max_length!=lengths[idx]:
                padd_pos=random.randint(0,max_length-lengths[idx])
            padding_data[idx,:,padd_pos:padd_pos+lengths[idx]] = seq
    return padding_data

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
            plt.text(i, j, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()

def plot_error_matrix(cm, classes, normalize=False, title='Error matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''
    cmpaintx=np.zeros(cm.shape[0])
    cmpainty=np.zeros(cm.shape[1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j]!=0 and i!=j:
                cmpaintx[i]=1
                cmpainty[j]=1
    xclasses=[]
    yclasses=[]
    for i,ispaint in enumerate(cmpaintx):
        if ispaint==1:
            xclasses.append(classes[i])
    for i,ispaint in enumerate(cmpainty):
        if ispaint==1:
            yclasses.append(classes[i])

    cm_new=np.zeros((len(xclasses),len(yclasses)))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j]!=0 and i!=j:
                xaxis=xclasses.index(classes[i])
                yaxis=yclasses.index(classes[j])
                cm_new[xaxis,yaxis]=cm[i,j]

    if normalize:
        cm_new = cm_new.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized error matrix")
    else:
        print('Error matrix, without normalization')
    plt.imshow(cm_new, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(xclasses))
    plt.xticks(tick_marks, xclasses, rotation=90)
    plt.yticks(tick_marks, yclasses)


    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm_new.max() / 2.

    for i in range(cm_new.shape[0]):
        for j in range(cm_new.shape[1]):
            num = '{:.2f}'.format(cm_new[i, j]) if normalize else int(cm_new[i, j])
            plt.text(i, j, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()


class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        alphabet+= '=' + '-' # '=' is special label,'-' for `-1` index
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text, label_list):
        """Support batch or single str.
        Args:
            text (int or list of int): texts to convert.
            label_list: label str responding to each int.
        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:
            item = label_list[item.item()]
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                # result.append(index)
                r.append(index)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return (torch.LongTensor(text), torch.LongTensor(length))


    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img
