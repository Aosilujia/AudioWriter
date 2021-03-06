import pandas as pd
import os
import queue
import string
import torch
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

    diff_sample=np.diff(cir_data,axis=0)

    diff_ang_sample = np.diff(ang_sample,n=1,axis=0)
    diff_amp_sample = np.diff(amp_sample,n=1,axis=0)
    # padding_diff_ang_sample, padding_diff_amp_sample = 0,0

    sample = np.stack((real_sample, imag_sample), axis=0)
    #sample = np.asarray([diff_sample.real])
    """标签预处理"""
    user_tag=os.path.basename(os.path.split(filepath)[0])
    content_tag=os.path.split(filepath)[1]
    tag_pure=""
    """去除文件名末尾的_?.csv共六位"""
    if (content_tag[0]=='='):
        tag_pure=content_tag[1:-6]
    else:
        tag_pure=content_tag[0:-6]
    return sample,tag_pure


class Dataset(Dataset):
    def __init__(self,directory_name,transform=None,initlabels=[]):
        if not os.path.isdir(directory_name):
            print(path+" is not a directory")
            return
        self.transform = transform
        """设置标签"""
        strlabels=string.ascii_letters
        labels=initlabels
        samples=[]
        tag_data=[]
        max=0.0
        """遍历文件读数据"""
        filepaths=csvfilelist(directory_name)
        #pool = multiprocessing.Pool(processes=5)
        #it = pool.imap_unordered(preprocess_cir, filepaths)
        #for i in it:
        for path in filepaths:
            i=preprocess_cir(path)
            sample=i[0]
            tag=i[1]
            samples.append(sample)
            tag_data.append(tag)
            if tag not in labels:
                labels.append(tag)
        padded_samples = padding(samples)
        print(padded_samples.shape)
        self.all_data=torch.from_numpy(padded_samples).float()
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
        return self.labels

    @property
    def tags(self):
        return self.all_tags

    @property
    def channel(self):
        return self.all_data.size()[1]

def int_split(dataset: Dataset, length: int) -> List[Subset]:
    """
    从每个标签对应的数据中分割固定int个
    Arguments:
        dataset (Dataset): Dataset to be split
        length (int): length of elements to be split
    """
    # Cannot verify that dataset is Sized
    if length >= len(dataset):  # type: ignore
        raise ValueError("Input length larger than the input dataset!")

    labels=dataset.label_list

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]



if __name__ == '__main__':
    #dataset=Dataset("../GSM_generation/training_data/alge")
    dataset=Dataset("../GSM_generation/training_data/word")
    print(dataset.channel)
    print(dataset.label_list)
