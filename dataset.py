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
from utility import csvfilelist,padding
import utility
from typing import List
import lmdb as lmdb
import pickle

def normalize01(a):
    b =a-np.min(a)
    b/=np.max(b)
    return b

def preprocess_cir(filepath):
    cir_data=np.genfromtxt(filepath, dtype=complex, delimiter=',')
    """数据预处理"""
    #上下限切割
    #np.clip(cir_data.real, -0.03, 0.03, out=cir_data.real)
    #np.clip(cir_data.imag, 0, 4, out=cir_data.imag)

    """各个维度的数据"""
    real_sample =cir_data.real
    imag_sample =cir_data.imag
    amp_sample = np.abs(cir_data)
    ang_sample = np.angle(cir_data)

    try:
        diff_sample=np.concatenate((np.asarray([np.zeros(121)]),np.diff(cir_data,axis=0)),axis=0)
    except(ValueError):
        print(filepath)
        print(cir_data)
        diff_sample=np.concatenate((np.asarray([np.zeros(121)]),np.diff(cir_data,axis=0)),axis=0)
    amp_diff_sample=np.abs(diff_sample)

    diff_ang_sample = np.diff(ang_sample,n=1,axis=0)
    diff_amp_sample = np.diff(amp_sample,n=1,axis=0)
    # padding_diff_ang_sample, padding_diff_amp_sample = 0,0

    sample = np.stack((real_sample, imag_sample,amp_diff_sample), axis=0)
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
        print("reading dataset from disk:"+directory_name)
        self.transform = transform
        """设置标签"""
        labels=initlabels[:]
        labels_full=initlabels[:]
        samples=[]
        tag_data=[]
        tag_full_data=[]
        sourcefile=[]
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
            if tag_full not in labels_full:
                labels_full.append(tag_full)
        padded_samples = padding(samples,max_length)
        self.data_shape=padded_samples.shape
        self.all_data=torch.from_numpy(padded_samples).float()
        self.source_files=sourcefile
        """索引标签到编号"""
        self.labels=labels
        self.all_tags=torch.tensor(label_mapping(tag_data,labels))

        self.labels_full=labels_full
        self.all_full_tags=torch.tensor(label_mapping(tag_full_data,labels_full))
        print("{} dataset datashape is:".format(directory_name))
        print(self.data_shape)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,idx):
        """transform预处理"""
        datum,tag,tag_full=self.all_data[idx],self.all_tags[idx],self.all_full_tags[idx]
        sourcefile=self.source_files[idx]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum,[tag,tag_full],sourcefile

    @property
    def label_list(self):
        #所有种类标签
        return self.labels

    @property
    def label_full_list(self):
        #所有种类标签
        return self.labels_full

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

    @property
    def sourcefiles(self):
        return self.source_files


class Dataset(Dataset):
    """读取保存好的npz文件"""
    def __init__(self,datafile_name,transform=None,initlabels=[],max_length=-1,padded=True):
        if not datafile_name.endswith('.npz'):
            print(datafile_name+" is not a npz file")
            return
        print("reading dataset from packed file:"+datafile_name)
        self.transform = transform
        self.padded=padded
        """设置初始标签，用来同步多数据集的标签编号"""
        labels=initlabels[:]
        labels_full=initlabels[:]

        """读数据文件"""
        origin_data=np.load(datafile_name)
        samples=origin_data['samples']
        tag_data=origin_data['tags']
        tag_full_data=origin_data['tag_full']
        tag_ground=origin_data['tag_ground']
        sourcefile=origin_data['sources']
        originlabels=origin_data['labels']
        if not padded:
            sample_lengths=origin_data['sample_lengths']
            self.sample_lengths=sample_lengths

        """处理并合并标签"""
        for newlabel in originlabels:
            if newlabel not in labels:
                labels.append(newlabel)
                labels_full.append(newlabel)
                labels_full.append('='+newlabel)

        if (max_length>=samples.shape[2]):
            padded_samples = padding(samples,max_length)#统一数据长度
        else:
            padded_samples = samples
        self.data_shape=padded_samples.shape
        self.all_data=torch.from_numpy(padded_samples).float() #转为torch tensor
        self.source_files=sourcefile
        self.ground_tags=tag_ground
        """索引标签到编号"""
        self.labels=labels
        self.all_tags=torch.tensor(label_mapping(tag_data,labels))

        self.labels_full=labels_full
        self.all_full_tags=torch.tensor(label_mapping(tag_full_data,labels_full))

        print("{} dataset datashape is:".format(datafile_name))
        print(self.data_shape)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self,idx):
        """transform预处理"""
        datum,tag,tag_full=self.all_data[idx],self.all_tags[idx],self.all_full_tags[idx]
        tag_ground=self.ground_tags[idx]
        sourcefile=self.source_files[idx]
        if not self.padded:
            datum=datum[:,:self.sample_lengths[idx]]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum,[tag,tag_full,tag_ground],sourcefile

    @property
    def label_list(self):
        #所有种类标签
        return self.labels

    @property
    def label_full_list(self):
        #所有种类标签
        return self.labels_full

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

    @property
    def groundtags(self):
        return self.ground_tags

    @property
    def sourcefiles(self):
        return self.source_files

"""从lmdb数据库读取数据"""
class LmdbDataset(Dataset):
    """初始化函数"""
    def __init__(self,lmdbs,transform=None,initlabels=[],max_length=-1,padded=True):
        initdb=""
        self.currentenv=None
        if not isinstance(lmdbs,(str,list)):
            print("first parameter, presenting the database directories, must be string or list of strings")
            return
        elif isinstance(lmdbs,str):
            #单个lmdb数据库的判断及部分初始化
            if not os.path.exists(lmdbs):
                print(lmdbs+" is not an existing file")
                return
            initdb=lmdbs
        else:
            #lmdb数据库列表的判断及部分初始化
            for db in lmdbs:
                if not os.path.exists(db):
                    print(db+" is not an existing file")
                    return
            initdb=lmdbs[0]

        self.lmdbs=lmdbs
        self.currentdb = ""
        self.transform = transform
        self.padded=padded
        self.labels=initlabels[:]
        self.labels_full=initlabels[:]
        self.max_width=0
        self.length=0
        self.db_sizes=[]
        self.index_bounds=[]
        self.nchannels=3 ## WARNING: hard coding
        self.nheight=121 ## WARNING: hard coding

        print("initiating lmdb:",initdb)

        """初始化各数据集参数"""
        if isinstance(lmdbs,str):
            """对单个库，读取标签和长度"""
            self.switchenv(initdb)
            self.getdbparams()
        elif isinstance(lmdbs,list):
            """对列表,循环记录所有标签和每个db数据量"""
            for db in lmdbs:
                self.switchenv(db)
                self.getdbparams()

    """工具函数"""
    def getdbparams(self):
        with self.currentenv.begin() as txn:
            labels=pickle.loads(txn.get('labels'.encode('ascii')))
            #处理并合并标签
            for newlabel in labels:
                if newlabel not in self.labels:
                    self.labels.append(newlabel)
                    self.labels_full.append(newlabel)
                    self.labels_full.append('='+newlabel) ## WARNING: not elegant
            currentdbsize=int(str(txn.get("sample_number".encode("ascii")),encoding='utf8'))
            currentdatawidth=int(str(txn.get("max_length".encode("ascii")),encoding='utf8'))
            if currentdatawidth>self.max_width:
                self.max_length=currentdatawidth
            self.length+=currentdbsize
            self.db_sizes.append(currentdbsize)
            self.index_bounds.append(self.length)

    def switchenv(self,dbname):
        #关闭上一个数据库，解除内存占用
        if self.currentdb==dbname:
            return
        if self.currentenv is not None:
            self.currentenv.close()
        #开启下一个数据库
        map_size =1024*1024*1024*50 # 50G=50*1024**3
        env=lmdb.open(dbname,map_size=map_size, readonly=True)
        self.currentenv=env
        self.currentdb=dbname

    def __len__(self):
        return self.max_length

    def __getitem__(self,idx):
        """找到数据idx对应的数据集"""
        dbidx=0
        if len(self.index_bounds)==1:
            #单db
            dbidx=idx
        else:
            for i in range(len(self.index_bounds)):
                #多db找idx
                if idx>=self.index_bounds[i]:
                    continue
                else:
                    if i==0:
                        dbidx=idx
                    else:
                        dbidx=idx-self.index_bounds[i-1]
                    self.switchenv(self.lmdbs[i])
                    break

        """读数据"""
        with self.currentenv.begin() as txn:
            datum=pickle.loads(txn.get('sample_{}'.format(dbidx).encode("ascii")))
            tag=pickle.loads(txn.get('tag_{}'.format(dbidx).encode("ascii")))
            tag_full=pickle.loads(txn.get('tag_full_{}'.format(dbidx).encode("ascii")))
            tag_ground=pickle.loads(txn.get('tag_ground_{}'.format(dbidx).encode("ascii")))
            sourcefile=pickle.loads(txn.get('source_{}'.format(dbidx).encode("ascii")))
            sample_length=pickle.loads(txn.get('sample_length_{}'.format(dbidx).encode("ascii")))

            if not self.padded:
                datum=datum[:,:sample_length]
            if self.transform is not None:
                """transform预处理"""
                datum = self.transform(datum)
            return datum,[tag,tag_full,tag_ground],sourcefile
        return 0,0,0

    @property
    def label_list(self):
        #所有种类标签
        return self.labels

    @property
    def label_full_list(self):
        #所有种类标签
        return self.labels_full

    @property
    def channel(self):
        return self.nchannels

    @property
    def datashape(self):
        return (self.length,self.nchannels,self.max_width,self.nheight)

    @property
    def idxbounds(self):
        return self.index_bounds


def packCIRData(directory_name,outputfile="cirdata.npz",lmdbname="",initlabels=[],for_aug=False,randomshift_n=1):
    """ 读所有cir csv文件打包到一个npz里面"""
    print("packing data from:"+directory_name+" into:"+outputfile)
    samples=[]
    tag_data=[] #标签
    tag_ground=[]
    tag_full_data=[]
    sourcefiles=[]
    data_lengths=[]
    labels=initlabels #标签种类
    max=0.0
    """遍历文件读数据"""
    filepaths=csvfilelist(directory_name)
    #pool = multiprocessing.Pool(processes=5)
    #it = pool.imap_unordered(preprocess_cir, filepaths)
    #for i in it:
    for path in filepaths:
        sourcefiles.append(path)
        i=preprocess_cir(path)
        sample=i[0]
        data_lengths.append(sample.shape[1])
        tag=i[1]
        tag_full=i[2]
        samples.append(sample)
        tag_data.append(tag)
        tag_full_data.append(tag_full)
        tag_ground.append(True)
        if tag not in labels:
            labels.append(tag)
    padded_samples = padding(samples,-1)
    all_tag_data=tag_data
    all_tag_full_data=tag_full_data
    all_tag_ground=tag_ground
    all_sourcefiles=sourcefiles
    all_data_lengths=data_lengths
    """数据增强：随机填充空白(左右移动)"""
    if (randomshift_n!=0):
        for i in range(randomshift_n):
            print("random padding {}".format(i))
            random_padded_samples= padding(samples,-1,padding_position=2)
            if for_aug and i==0:
                padded_samples=random_padded_samples
            else:
                padded_samples=np.concatenate((padded_samples,random_padded_samples),axis=0)
            all_tag_data=all_tag_data+tag_data
            all_tag_full_data=all_tag_full_data+tag_full_data
            all_tag_ground=all_tag_ground+[False]*len(tag_ground)
            all_sourcefiles=all_sourcefiles+sourcefiles
            all_data_lengths=all_data_lengths+data_lengths
    all_samples=padded_samples

    if (lmdbname==""):
        np.savez(outputfile,samples=all_samples,tags=all_tag_data,tag_full=all_tag_full_data,labels=labels,sources=all_sourcefiles,tag_ground=all_tag_ground,sample_lengths=all_data_lengths)
    else:
        map_size =1024*1024*1024*50
        dbname='../lmdb/'+lmdbname
        env=lmdb.open(dbname,map_size=map_size)
        print("start writing to lmdb "+dbname)
        with env.begin(write=True) as txn:
            #写所有数据
            for i in range(len(all_samples)):
                txn.put("sample_{}".format(i).encode("ascii"),pickle.dumps(all_samples[i]))
                txn.put("tag_{}".format(i).encode("ascii"),pickle.dumps(all_tag_data[i]))
                txn.put("tag_full_{}".format(i).encode("ascii"),pickle.dumps(all_tag_full_data[i]))
                txn.put("source_{}".format(i).encode("ascii"),pickle.dumps(all_sourcefiles[i]))
                txn.put("tag_ground_{}".format(i).encode("ascii"),pickle.dumps(all_tag_ground[i]))
                txn.put("sample_length_{}".format(i).encode("ascii"),pickle.dumps(all_data_lengths[i]))
            #写meta数据
            txn.put("sample_number".encode("ascii"),'{}'.format(len(all_samples)).encode("ascii"))
            txn.put("labels".encode("ascii"),pickle.dumps(labels))
            txn.put("max_length".encode("ascii"),'{}'.format(all_samples.shape[2]).encode("ascii"))
        env.close()

def int_split(dataset: Dataset, length: int, partial=1) -> List[Subset]:
    """
    从每个标签对应的数据中分割固定int个,或者按照partial 0.x 分割部分
    Arguments:
        dataset (Dataset): Dataset to be split
        length (int): length of elements to be split
    """
    # Cannot verify that dataset is Sized
    if length >= len(dataset):  # type: ignore
        raise ValueError("Input length larger than the input dataset!")

    labels=dataset.label_list
    tags=dataset.tags
    tag_ground=[]
    if hasattr(dataset,"groundtags"):
        tag_ground=dataset.groundtags
    else:
        tag_ground=[True]*len(tags)
    sources=dataset.sourcefiles
    train_indices=[]
    val_indices=[]
    """因为有增强的数据，所以要注意是否是groundtruth"""
    tag_indices=[]
    ground_indices=[]
    #初始化tag下标二维数组
    for i in range(len(labels)):
        tag_indices.append([])
        ground_indices.append([])
    #将数据集中tag加入tag数组
    for indice,tag in enumerate(tags):
        tag_indices[tag].append(indice)
        if tag_ground[indice]:
            ground_indices[tag].append(indice)
    #遍历每个label对应的所有标签
    for i in range(len(labels)):
        tag_indice=tag_indices[i]
        ground_indice=ground_indices[i]
        """必须从真实data里选验证集"""
        val_length=length
        if partial<1:
            val_length=int(partial*len(ground_indice))
        #从ground truth抽取作为验证集的下标
        val_indice=random.sample(ground_indice,val_length)
        val_indices.append(val_indice)
        for value in val_indice: #抽中的当验证集的数据的下标
            tag_indice.remove(value)
            for index in tag_indice: #同标签的所有数据的下标
                if sources[index]==sources[value]: #同一个文件增强出的数据
                    tag_indice.remove(index)
        train_indices.append(tag_indice)
    return [Subset(dataset,list(itertools.chain.from_iterable(train_indices))), Subset(dataset,list(itertools.chain.from_iterable(val_indices)))]

def data_augmentation(sample):
    """
    数据增强，输入一个sample，做处理后输出一个list(sample)
    """
    return 0

def label_mapping(tags,label_list)-> List[int]:
    """
    重写的其它类型映射到label，返回数字数组
    """
    label_dict={}
    for i in range(len(label_list)):
        label_dict[label_list[i]]=i
    tags=list(map(label_dict.get,tags))
    return tags

def lmdbtest():
    map_size =1024**3
    env=lmdb.open('../lmdb/jxyaug1',map_size=map_size)
    a=np.asarray([[1,2,3,4],[5,6,7,8]])
    #with env.begin(write=True) as txn:
    #    value=a
    #    txn.put("test1".encode("ascii"),pickle.dumps(value))
        #txn.commit()
    txn=env.begin()
    file=txn.get('sample_6000'.encode("ascii"))
    #print(file)
    print(pickle.loads(file))
    return


if __name__ == '__main__':
    #dataset=diskDataset("../GSM_generation/training_data/Word_jxydorm")
    #packCIRData("../GSM_generation/training_data/Word",lmdbname="jxyword",randomshift_n=0,for_aug=False)
    #packCIRData("../GSM_generation/training_data/Word","jxy_dcirset.npz",randomshift_n=0)
    #packCIRData("../GSM_generation/training_data/Word_jjz",lmdbname="jjzword",randomshift_n=0)
    #packCIRData("../GSM_generation/training_data/Word_jxydorm","jxydorm_dataset.npz",randomshift_n=0)
    #dataset=Dataset("jxy_dataset.npz")
    #dataset2=Dataset("jxynew_dataset.npz",padded=False)
    #dataset3=LmdbDataset(["../lmdb/jxydorm","../lmdb/jjzword"])
    dataset3=LmdbDataset("../lmdb/jxyaug1")
    print(len(dataset3))
    print(dataset3[1000])
    #print(dataset2[40][0].shape)
    #set1,set2=int_split(dataset,2,partial=0.2)
    #print(len(set1))
    #lmdbtest()
