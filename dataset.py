import os
import queue
import string
import torch
import random
import itertools
import numpy as np
import multiprocessing
from torch.utils.data import Dataset,sampler,Subset
import torchvision.transforms as transforms
from sklearn import preprocessing
from utility import csvfilelist,padding
from typing import List


def normalize01(a):
    b =a-np.min(a)
    b/=np.max(b)
    return b

def preprocess_cir(filepath):
    cir_data=np.genfromtxt(filepath, dtype=complex, delimiter=',')
    """数据预处理"""
    real_sample =cir_data.real
    imag_sample =cir_data.imag
    amp_sample = np.abs(cir_data)
    ang_sample = np.angle(cir_data)

    diff_sample=np.concatenate((np.asarray([np.zeros(121)]),np.diff(cir_data,axis=0)),axis=0)
    amp_diff_sample=np.abs(diff_sample)

    diff_ang_sample = np.diff(ang_sample,n=1,axis=0)
    diff_amp_sample = np.diff(amp_sample,n=1,axis=0)
    # padding_diff_ang_sample, padding_diff_amp_sample = 0,0

    sample = np.stack((real_sample, imag_sample,diff_sample.real), axis=0)
    #sample = np.asarray([amp_diff_sample])
    """标签预处理"""
    user_tag=os.path.basename(os.path.split(filepath)[0])
    content_tag=os.path.split(filepath)[1]
    tag_pure=""
    tag_full=content_tag[0:content_tag.find('_')] #带转笔标签的tag
    """去除文件名末尾的_?.csv"""
    if (content_tag[0]=='='):
        tag_pure=content_tag[1:content_tag.find('_')]
    else:
        tag_pure=content_tag[0:content_tag.find('_')]
    return sample,tag_pure,tag_full


class diskDataset(Dataset):
    """读取一个文件夹内所有的csv文件"""
    def __init__(self,directory_name,transform=None,initlabels=[],max_length=-1):
        if not os.path.isdir(directory_name):
            print(path+" is not a directory")
            return
        self.transform = transform
        """设置标签"""
        labels=initlabels
        samples=[]
        tag_data=[]
        sourcefile=[]
        max=0.0
        """遍历文件读数据"""
        filepaths=csvfilelist(directory_name)
        #pool = multiprocessing.Pool(processes=5)
        #it = pool.imap_unordered(preprocess_cir, filepaths)
        #for i in it:
        for path in filepaths:
            sourcefile.append(path)
            i=preprocess_cir(path)
            sample=i[0]
            tag=i[1]
            samples.append(sample)
            tag_data.append(tag)
            if tag not in labels:
                labels.append(tag)
        padded_samples = padding(samples,max_length)
        self.data_shape=padded_samples.shape
        self.all_data=torch.from_numpy(padded_samples).float()
        self.source_files=sourcefile
        """索引标签到编号"""
        le=preprocessing.LabelEncoder()
        le.fit(labels)
        self.labels=labels
        self.all_tags=torch.tensor(le.transform(tag_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,idx):
        """transform预处理"""
        datum,tag=self.all_data[idx],self.all_tags[idx]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum,tag

    @property
    def label_list(self):
        #所有种类标签
        return self.labels

    """@property
    def tags(self):
        #所有标签
        return self.all_tags"""

    @property
    def channel(self):
        return self.data_shape[1]

    @property
    def datashape(self):
        return self.data_shape

class Dataset(Dataset):
    """读取保存好的npz文件"""
    def __init__(self,datafile_name,transform=None,initlabels=[],max_length=-1):
        if not datafile_name.endswith('.npz'):
            print(datafile_name+" is not a npz file")
            return
        self.transform = transform
        """设置初始标签，用来同步多数据集的标签编号"""
        labels=initlabels

        """读数据文件"""
        origin_data=np.load(datafile_name)
        samples=origin_data['samples']
        tag_data=origin_data['tags']
        tag_full=origin_data['tag_full']
        sourcefile=origin_data['sources']
        originlabels=origin_data['labels']

        """处理并合并标签"""
        for newlabel in originlabels:
            if newlabel not in labels:
                labels.append(newlabel)

        padded_samples = padding(samples,max_length) #统一数据长度
        self.data_shape=padded_samples.shape
        self.all_data=torch.from_numpy(padded_samples).float() #转为torch tensor
        self.source_files=sourcefile
        """索引标签到编号"""
        le=preprocessing.LabelEncoder()
        le.fit(labels)
        self.labels=labels
        self.all_tags=torch.tensor(le.transform(tag_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,idx):
        """transform预处理"""
        datum,tag=self.all_data[idx],self.all_tags[idx]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum,tag

    @property
    def label_list(self):
        #所有种类标签
        return self.labels

    @property
    def tags(self):
        #所有标签
        return self.all_tags

    @property
    def channel(self):
        return self.data_shape[1]

    @property
    def datashape(self):
        return self.data_shape

def packCIRData(directory_name,outputfile="cirdata.npz",initlabels=[],environment=""):
    """ 读所有cir csv文件打包到一个npz里面"""
    samples=[]
    tag_data=[] #标签
    tag_full_data=[]
    sourcefile=[]
    labels=initlabels #标签种类
    max=0.0
    """遍历文件读数据"""
    filepaths=csvfilelist(directory_name)
    #pool = multiprocessing.Pool(processes=5)
    #it = pool.imap_unordered(preprocess_cir, filepaths)
    #for i in it:
    for path in filepaths:
        sourcefile.append(path)
        i=preprocess_cir(path)
        sample=i[0]
        tag=i[1]
        tag_full=i[2]
        samples.append(sample)
        tag_data.append(tag)
        tag_full_data.append(tag_full)
        if tag not in labels:
            labels.append(tag)
    padded_samples = padding(samples,-1)
    np.savez(outputfile,samples=padded_samples,tags=tag_data,tag_full=tag_full,labels=labels,sources=sourcefile)

def int_split(dataset: Dataset, length: int) -> List[Subset]:
    """
    unfinished:从每个标签对应的数据中分割固定int个
    Arguments:
        dataset (Dataset): Dataset to be split
        length (int): length of elements to be split
    """
    # Cannot verify that dataset is Sized
    if length >= len(dataset):  # type: ignore
        raise ValueError("Input length larger than the input dataset!")

    labels=dataset.label_list
    tags=dataset.tags
    train_indices=[]
    val_indices=[]
    tag_indices=[]
    for i in range(len(labels)):
        tag_indices.append([])
    for indice,tag in enumerate(tags):
        tag_indices[tag].append(indice)
    for i in range(len(labels)):
        tag_indice=tag_indices[i]
        val_indice=random.sample(tag_indice,length)
        val_indices.append(val_indice)
        for value in val_indice:
            tag_indice.remove(value)
        train_indices.append(tag_indice)
    indices=[list(itertools.chain.from_iterable(train_indices)),list(itertools.chain.from_iterable(val_indices))]
    return [Subset(dataset, indices)]



if __name__ == '__main__':
    #dataset=Dataset("../GSM_generation/training_data/Alge")
    #packCIRData("../GSM_generation/training_data/Word_jxydorm","testcir.npz")
    dataset=Dataset("testcir.npz")
    int_split(dataset,2)
