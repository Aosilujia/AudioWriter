import pandas as pd
import os
import queue
import string
import torch
import numpy as np
from torch.utils.data import Dataset,sampler
from sklearn import preprocessing

class AlgeDataset(Dataset):
    def __init__(self,directory_name):
        if not os.path.isdir(directory_name):
            print(path+" is not a directory")
            return

        """设置标签"""
        strlabels=string.ascii_letters
        labels=[]
        labels[:0]=strlabels
        self.labels=labels

        dcir_data=None
        tag_data=[]

        dirqueue = queue.Queue()
        dirqueue.put(directory_name)
        while not dirqueue.empty():
            node = dirqueue.get()
            for filename in os.listdir(node):
                nextpath = os.path.join(node, filename)
                if os.path.isdir(nextpath):
                    dirqueue.put(nextpath)
                elif nextpath.endswith('.csv'):
                    user_tag=os.path.basename(node)
                    content_tag=os.path.basename(os.path.split(node)[0])
                    cir_data = np.genfromtxt(nextpath, dtype=complex, delimiter=',')
                    """预处理数据"""
                    cir_diff = np.diff(np.abs(cir_data), axis=0)
                    """预处理标签"""
                    tag_pure=""
                    if (content_tag[0]=='='):
                        tag_pure=content_tag[1]
                    else:
                        tag_pure=content_tag[0]
                    """合并数据"""
                    if (tag_pure=='A' or tag_pure=='B'):
                        if (dcir_data is None):
                            dcir_data=cir_diff.reshape((1,1,)+cir_diff.shape)
                        else:
                            cir_diff_t=cir_diff.reshape((1,1,)+cir_diff.shape)
                            dcir_data=np.concatenate((dcir_data,cir_diff_t))
                        tag_data.append(tag_pure)

        self.all_data=torch.from_numpy(dcir_data).float()

        le=preprocessing.LabelEncoder()
        le.fit(labels)
        self.all_tags=torch.tensor(le.transform(tag_data))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,idx):
        return self.all_data[idx],self.all_tags[idx]

    @property
    def label_list(self):
        return self.labels


class randomSequentialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples




if __name__ == '__main__':
    AlgeDataset("../GSM_generation/training_data")
