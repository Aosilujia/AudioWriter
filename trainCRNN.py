import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
import utility as utils
import dataset
from difflib import SequenceMatcher

import models.CRNN as net
import params

os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--trainroot', required=False, help='path to train dataset')
parser.add_argument('-val', '--valroot', required=False, help='path to val dataset')
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")

# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
#all_dataset = dataset.diskDataset("../GSM_generation/training_data/Word")
all_dataset = dataset.Dataset("augcir_moving2.npz")
label_list=all_dataset.label_list
test_dataset = ""
test_dataset = dataset.Dataset("jxynew_dataset.npz",initlabels=label_list)
if (test_dataset!=""):
    label_list=test_dataset.label_list
used_tag=params.tag_choice

batch_size=params.batchSize

def data_loader(all_dataset):
    assert all_dataset
    train_length=int(len(all_dataset)*0.8)
    train_dataset,val_dataset=dataset.int_split(all_dataset,5,0.2)
    #train_dataset,val_dataset=random_split(all_dataset,[train_length,len(all_dataset)-train_length])

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, \
            shuffle=True)
    # val
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

train_loader, val_loader = data_loader(all_dataset)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=True)
nclass = len(label_list)
# -----------------------------------------------
"""
In this block
    Net init
    Weight init
    Load pretrained model
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init(nclass,cirH,channel_input):
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(cirH, channel_input, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    return crnn

datashape=all_dataset.datashape
cirH=datashape[3]
channel_input=datashape[1]
crnn = net_init(nclass,cirH,channel_input)
print(crnn)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# Convert between str and label.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        because train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, channel_input, cirH, cirH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=params.device_ids)

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

if params.multi_gpu:
    crnn = torch.nn.DataParallel(crnn, device_ids=params.device_ids)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version.
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    n_count = 0
    char_correct = 0
    char_count=0
    loss_avg = utils.averager() # The global loss_avg is used by train

    max_iter = len(val_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts, cpu_sources= data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts[used_tag],label_list)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts[used_tag]:
            cpu_texts_decode.append(label_list[i])
        for pred, target in zip(sim_preds, cpu_texts_decode):
            n_count+=1
            char_count+=len(target)
            if pred == target:
                n_correct += 1
                char_correct+=len(target)
            else:
                matching_blocks = SequenceMatcher(None, pred, target).get_matching_blocks()
                sim_len=0
                for block in matching_blocks:
                    sim_len+=block.size
                char_correct+=sim_len
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / n_count
    sim_accuracy = char_correct / char_count
    print('Val loss: %f, accuracy: %f, character_accuracy: %f' % (loss_avg.val(), accuracy,sim_accuracy))
    return accuracy, sim_accuracy

def test(net, criterion):
    if (test_dataset==""):
        return
    print('Start test')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    test_iter = iter(test_loader)

    i = 0
    n_correct = 0
    n_count = 0
    char_correct = 0
    char_count=0
    loss_avg = utils.averager() # The global loss_avg is used by train

    max_iter = len(test_loader)
    for i in range(max_iter):
        data = test_iter.next()
        i += 1
        cpu_images, cpu_texts, cpu_sources= data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts[used_tag],label_list)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts[used_tag]:
            cpu_texts_decode.append(label_list[i])
        for pred, target in zip(sim_preds, cpu_texts_decode):
            n_count+=1
            char_count+=len(target)
            if pred == target:
                n_correct += 1
                char_correct+=len(target)
            else:
                matching_blocks = SequenceMatcher(None, pred, target).get_matching_blocks()
                sim_len=0
                for block in matching_blocks:
                    sim_len+=block.size
                char_correct+=sim_len
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / n_count
    sim_accuracy = char_correct / char_count
    print('Test loss: %f, accuracy: %f, character_accuracy: %f' % (loss_avg.val(), accuracy,sim_accuracy))
    return accuracy, sim_accuracy


def train(net, criterion, optimizer, train_iter):
    for p in net.parameters():
        p.requires_grad = True
    net.train()

    data = train_iter.next()
    cpu_images, cpu_texts ,cpu_source= data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts[used_tag],label_list)
    utils.loadData(text, t)
    utils.loadData(length, l)

    optimizer.zero_grad()
    preds = net(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    best_accuracy=0
    best_sim_accuracy=0
    best_epoch=0
    best_epoch_sim=0
    for epoch in range(params.nepoch):
        train_iter = iter(train_loader)
        i = 0
        data_num=len(train_loader)
        if params.val_mode:
            i=0
        while i < data_num:
            cost = train(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            displayInterval=int(data_num/params.displayPerEpoch)
            if displayInterval==0: displayInterval=1
            valInterval=int(data_num/params.valPerEpoch)
            if valInterval==0: valInterval=1
            if i % displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if params.pretrained == '':
                if epoch>=10 and epoch%params.valEpochInterval==0 and i % valInterval == 0:
                    accuracy,sim_accuracy=val(crnn, criterion)
                    test(crnn, criterion)
                    if accuracy>best_accuracy:
                        best_accuracy=accuracy
                        best_epoch=epoch
                    if sim_accuracy>best_sim_accuracy:
                        best_sim_accuracy=sim_accuracy
                        best_epoch_sim=epoch
            else :
                if epoch%params.valEpochInterval==0 and i % valInterval == 0:
                    accuracy,sim_accuracy=val(crnn, criterion)
                    if accuracy>best_accuracy:
                        best_accuracy=accuracy
                        best_epoch=epoch
                    if sim_accuracy>best_sim_accuracy:
                        best_sim_accuracy=sim_accuracy
                        best_epoch_sim=epoch
            # do checkpointing
        if epoch % params.saveInterval == 0 or epoch==params.nepoch-1:
            torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_.pth'.format(params.expr_dir, epoch))
    print("best accuracy:{},at epoch:{}".format(best_accuracy,best_epoch))
    print("best sim accuracy:{},at epoch:{}".format(best_sim_accuracy,best_epoch_sim))
    np.save('crnn_accuracy_2',[best_epoch,best_accuracy,best_epoch_sim,best_sim_accuracy])
