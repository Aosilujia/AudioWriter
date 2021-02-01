import os
import queue
import string
import torch
import random
import itertools
import numpy as np
import multiprocessing
from torch.utils.data import Dataset,Sampler,Subset
import torchvision.transforms as transforms
from utility import csvfilelist,padding
import utility
from typing import List
import lmdb as lmdb
import pickle


"""---------------------------------------------------------------------
针对单个原始数据的预处理
"""
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


"""------------------------------------------------------------------------------
dataset的各个实现
"""
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

        """原始str标签，不带或带="""
        self.tag_data=[]
        self.tag_full_data=[]
        self.source_files=[]
        self.ground_tags=[]
        """映射到数字int的标签"""
        self.all_tags=[]
        self.all_full_tags=[]


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

        """索引标签到编号"""
        self.all_tags=torch.tensor(label_mapping(self.tag_data,self.labels))
        self.all_full_tags=torch.tensor(label_mapping(self.tag_full_data,self.labels_full))

        print(self.datashape)

    """工具函数"""
    def getdbparams(self):
        with self.currentenv.begin() as txn:
            labels=pickle.loads(txn.get('labels'.encode('ascii')))
            """处理并合并标签"""
            for newlabel in labels:
                if newlabel not in self.labels:
                    self.labels.append(newlabel)
                    self.labels_full.append(newlabel)
                    self.labels_full.append('='+newlabel) ## WARNING: not elegant
            currentdbsize=int(str(txn.get("sample_number".encode("ascii")),encoding='utf8'))
            currentdatawidth=int(str(txn.get("max_length".encode("ascii")),encoding='utf8'))
            if currentdatawidth>self.max_width:
                self.max_width=currentdatawidth
            self.length+=currentdbsize
            self.db_sizes.append(currentdbsize)
            self.index_bounds.append(self.length)
            """遍历记录tag，以便根据tag分割数据集"""
            for i in range(currentdbsize):
                tag=pickle.loads(txn.get('tag_{}'.format(i).encode("ascii")))
                tag_full=pickle.loads(txn.get('tag_full_{}'.format(i).encode("ascii")))
                tag_ground=pickle.loads(txn.get('tag_ground_{}'.format(i).encode("ascii")))
                sourcefile=pickle.loads(txn.get('source_{}'.format(i).encode("ascii")))
                self.tag_data.append(tag)
                self.tag_full_data.append(tag_full)
                self.ground_tags.append(tag_ground)
                self.source_files.append(sourcefile)

    def switchenv(self,dbname):
        #关闭上一个数据库，解除内存占用
        if self.currentdb==dbname:
            return
        if self.currentenv is not None:
            self.currentenv.close()
        #开启下一个数据库
        env=lmdb.open(dbname, readonly=True) #读的时候readonly一定是True，不然会重新分配空间
        self.currentenv=env
        self.currentdb=dbname

    def __len__(self):
        return self.length

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
            tag=self.all_tags[idx]
            tag_full=self.all_full_tags[idx]
            tag_ground=self.ground_tags[idx]
            sourcefile=self.source_files[idx]
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
    def tags(self):
        #所有标签
        return self.all_tags

    @property
    def groundtags(self):
        return self.ground_tags

    @property
    def sourcefiles(self):
        return self.source_files

    @property
    def dblist(self):
        return self.lmdbs

    @property
    def idxbounds(self):
        return self.index_bounds

    @property
    def dbsizes(self):
        return self.db_sizes

def packCIRData(directory_name,outputfile="cirdata.npz",lmdbname="",initlabels=[],for_aug=False,randomshift_n=0,GB=0):
    """ 读所有cir csv文件打包到一个npz里面"""
    print("packing data from:"+directory_name+" into:"+outputfile)
    samples=[]
    tag_data=[] #标签
    tag_ground=[]
    tag_full_data=[]
    sourcefiles=[]
    data_lengths=[]
    labels=initlabels #标签种类
    dataset_size=0
    #初始化数据库
    if lmdbname!="":
        #基准1G
        map_size =1024*1024*1024
        if GB==0:
            #默认5G
            map_size*=5
        else:
            map_size=int(map_size*GB)
        #根据增强次数分配多倍空间
        map_size*=(randomshift_n+int(not for_aug))
        dbname='../lmdb/'+lmdbname
        env=lmdb.open(dbname,map_size=map_size)

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
    """写数据库可以分段写，防止内存的数组过大"""
    if lmdbname!="" and not for_aug:
        print("start writing data to lmdb "+dbname)
        with env.begin(write=True) as txn:
            #先写原始sample数据
            for i in range(len(samples)):
                txn.put("sample_{}".format(i).encode("ascii"),pickle.dumps(padded_samples[i]))
        dataset_size+=len(samples)
    """for_aug表示不存原始数据只存增强数据"""
    if not for_aug:
        all_tag_data=tag_data
        all_tag_full_data=tag_full_data
        all_tag_ground=tag_ground
        all_sourcefiles=sourcefiles
        all_data_lengths=data_lengths
    else:
        all_tag_data=[]
        all_tag_full_data=[]
        all_tag_ground=[]
        all_sourcefiles=[]
        all_data_lengths=[]
    """数据增强：随机填充空白(左右移动)"""
    if (randomshift_n!=0):
        for i in range(randomshift_n):
            print("random padding {}".format(i))
            random_padded_samples= padding(samples,-1,padding_position=2)
            if lmdbname!="":
                print("start writing data to lmdb "+dbname)
                with env.begin(write=True) as txn:
                    for i in range(len(random_padded_samples)):
                        txn.put("sample_{}".format(dataset_size+i).encode("ascii"),pickle.dumps(random_padded_samples[i]))
                dataset_size+=len(random_padded_samples)
                padded_samples=random_padded_samples
            else:
                """一般方法把所有数据拼在一起，放在内存"""
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
        with env.begin(write=True) as txn:
            #写除sample外所有数据
            for i in range(dataset_size):
                txn.put("tag_{}".format(i).encode("ascii"),pickle.dumps(all_tag_data[i]))
                txn.put("tag_full_{}".format(i).encode("ascii"),pickle.dumps(all_tag_full_data[i]))
                txn.put("source_{}".format(i).encode("ascii"),pickle.dumps(all_sourcefiles[i]))
                txn.put("tag_ground_{}".format(i).encode("ascii"),pickle.dumps(all_tag_ground[i]))
                txn.put("sample_length_{}".format(i).encode("ascii"),pickle.dumps(all_data_lengths[i]))
            #写meta数据
            txn.put("sample_number".encode("ascii"),'{}'.format(dataset_size).encode("ascii"))
            txn.put("labels".encode("ascii"),pickle.dumps(labels))
            txn.put("max_length".encode("ascii"),'{}'.format(all_samples.shape[2]).encode("ascii"))
        env.close()

def data_augmentation(sample):
    """
    数据增强，输入一个sample，做处理后输出一个list(sample)
    """
    return 0

"""-------------------------------------------------------------------------------
对dataset和dataloader使用的各种工具函数，分割，sampler以及batchsampler
"""
def int_split(dataset: Dataset, length: int, partial=1):
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
    return [Subset(dataset,list(itertools.chain.from_iterable(train_indices))), Subset(dataset,list(itertools.chain.from_iterable(val_indices))),
            list(itertools.chain.from_iterable(train_indices)),list(itertools.chain.from_iterable(val_indices))]


class DBRandomSampler(Sampler):
    r"""多lmdb风格的随机sampler，根据不同db分别进行shuffle
    Arguments:
        data_source (Dataset): lmdbdataset必须有数据库的名称表dblist 和每个对应的数据索引输idxbounds
        replacement (bool): 现在没有用的参数。samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): 返回总数据集长度。number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """
    def __init__(self, data_source,idxbounds=[],indices=[], replacement=False, num_samples=None):
        self.data_source = data_source
        # 这个参数控制的应该为是否重复采样
        self.replacement = replacement
        self._num_samples = num_samples
        self.idxbounds = idxbounds
        self.indices = indices
        self.indexes = []
        self.trueidxbounds = []
        self.dborder=[]
        """把所有下标根据idxbound分到数据库对应的数组中"""
        for i in range(len(idxbounds)):
            self.indexes.append([])
        for idx in self.indices:
            dbidx,size=self.finddb(idx)
            self.indexes[dbidx].append(idx)
        for i in range(len(idxbounds)):
            if self.trueidxbounds==[]:
                self.trueidxbounds.append(len(self.indexes[i]))
            else:
                self.trueidxbounds.append(self.trueidxbounds[i-1]+len(self.indexes[i]))

    def finddb(self,idx):
        for i in range(len(self.idxbounds)):
            if (idx<self.idxbounds[i]):
                if i==0:
                    size=self.idxbounds[i]
                else:
                    size=self.idxbounds[i]-self.idxbounds[i-1]
                return i,size

    # 省略类型检查
    @property
    def num_samples(self):
        # dataset size might change at runtime
        # 初始化时不传入num_samples的时候使用数据源的长度
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
    # 返回数据集长度
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        """原始版本是打乱所有数据。db版本在各个数据集内打乱数据，然后再打乱各个数据集的顺序。
            因为每个数据集的cir宽度不同，所以为了输入到网络需要分别处理(靠batchsampler)。
        """
        idxbounds= self.idxbounds
        result_indexs=[]
        """随机数据库顺序"""
        dborder=torch.randperm(len(idxbounds))
        self.dborder=dborder
        for dbidx in dborder:
            """子数据集随机下标"""
            indexlist=self.indexes[dbidx]
            np.random.shuffle(indexlist)
            randomidxs=indexlist
            result_indexs+=randomidxs
        return iter(result_indexs)

    @property
    def idx_bounds(self):
        return self.trueidxbounds

    @property
    def db_order(self):
        return self.dborder

"""lmdb风格批采样，必须根据数据库把不同长度的数据装到不同batch"""
class DBBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices."""
    def __init__(self ,sampler, batch_size, drop_last=False):
        # ...省略类型检查
        # 定义使用何种采样器Sampler
        self.sampler = sampler
        self.batch_size = batch_size
        # 是否在采样个数小于batch_size时剔除本次采样
        self.drop_last = drop_last

        #数据库相关
        if not (hasattr(sampler,"db_order") and hasattr(sampler,"idx_bounds")):
            print("illegal sampler for batch sampler,please use a lmdb sampler")
            return
        self.idxbounds=sampler.idx_bounds
        self.currentdb=-1 #表示目前batch所存db
        self.length=0 #batch数量记录器
        #iter全局变量
        self.currentdbsize=0
        self.currentiter=0 # current db counter

        idxbounds=self.idxbounds
        for dbidx in range(len(idxbounds)):
            #遍历计算每个db会装几个batch
            if dbidx==0:
                dbsize=idxbounds[0]
            else:
                dbsize=idxbounds[dbidx]-idxbounds[dbidx-1]
            if self.drop_last:
                self.length+=dbsize // self.batch_size
            else:
                self.length+=(dbsize + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch = []
        dbcounter = 0
        counter = 0
        for idx in self.sampler:
            counter+=1
            dborder=self.sampler.db_order
            if self.currentdb==-1:
                """切换到对应的数据库"""
                self.currentdb=dborder[dbcounter]
                if self.currentdb==0:
                    self.currentdbsize=self.idxbounds[self.currentdb]
                else:
                    self.currentdbsize=self.idxbounds[self.currentdb]-self.idxbounds[self.currentdb-1]
                """数据增强集或者太小的db不会包括到验证集，导致当前dbsize为0，跳到下一个"""
                while self.currentdbsize==0 and dbcounter<len(dborder):
                    dbcounter+=1
                    self.currentdb=dborder[dbcounter]
                    if self.currentdb==0:
                        self.currentdbsize=self.idxbounds[self.currentdb]
                    else:
                        self.currentdbsize=self.idxbounds[self.currentdb]-self.idxbounds[self.currentdb-1]
                self.currentiter=0
            batch.append(idx)
            # 如果采样个数和batch_size相等则本次采样完成
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            self.currentiter+=1
            if self.currentiter>=self.currentdbsize:
                """当前数据库的数据已全部装载到batch中，准备切换下一个数据库"""
                self.currentdb=-1
                dbcounter+=1
                # 在不需要剔除不足batch_size的采样个数时返回当前batch
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch=[]

    def __len__(self):
        # 在不进行剔除时，数据的长度就是采样器索引的长度
        return self.length

"""------------------------------------------------------------------
工具小函数
"""
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
    #map_size =1024**3
    env=lmdb.open('../lmdb/jjzword2',readonly=True)
    print(env.stat())
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


def finddb(idx,idxbounds):
    for i in range(len(idxbounds)):
        if (idx<idxbounds[i]):
            if i==0:
                size=idxbounds[i]
            else:
                size=idxbounds[i]-idxbounds[i-1]
            return i,size



"""测试以及快速调用区域"""
if __name__ == '__main__':
    #dataset=diskDataset("../GSM_generation/training_data/Word_jxydorm")
    #packCIRData("../GSM_generation/training_data/Word",lmdbname="jxyaug3",randomshift_n=3,for_aug=True,GB=6.5)
    #packCIRData("../GSM_generation/training_data/Word","jxy_dcirset.npz",randomshift_n=0)
    #packCIRData("../GSM_generation/training_data/Word_jxynew",lmdbname="jxynew",randomshift_n=0,for_aug=False,GB=0.75)
    #packCIRData("../GSM_generation/training_data/Word_zq",lmdbname="zqword",randomshift_n=0,for_aug=False,GB=2.2)
    #packCIRData("../GSM_generation/training_data/Word_zq",lmdbname="zqaug1",randomshift_n=1,for_aug=True,GB=2.2)
    #packCIRData("../GSM_generation/training_data/Word_zq",lmdbname="zqaug2",randomshift_n=2,for_aug=True,GB=2.2)
    #packCIRData("../GSM_generation/training_data/Word_zq",lmdbname="zqaug3",randomshift_n=3,for_aug=True,GB=2.2)
    #dataset=Dataset("jxy_dataset.npz")
    #dataset2=Dataset("jxynew_dataset.npz",padded=False)
    dblist=[]
    for user in ["jxy","zq","jjz"]:
        dblist.append("../lmdb/"+user+"word")
        dblist.append("../lmdb/"+user+"aug1")

    print(dblist)
    #dataset3=LmdbDataset(["../lmdb/jjzword","../lmdb/jxynew","../lmdb/jxydorm"])
    dataset3=LmdbDataset(dblist)
    #print(dataset3.label_list)
    #print(dataset3.tags)
    #print(dataset3.ground_tags)
    #print(dataset3[40][0][0][000:400])
    idxbounds=dataset3.idxbounds
    print(idxbounds)
    set1,set2,train_indices,val_indices=int_split(dataset3,2,partial=0.2)
    #sampler=DBRandomSampler(set1,idxbounds,train_indices)
    #batchsampler=DBBatchSampler(sampler,7)
    valsampler=DBBatchSampler(DBRandomSampler(set2,idxbounds,val_indices),7)
    """for x in batchsampler:
        dbidx,size0=finddb(x[0],idxbounds)
        for i in x:
            dbidx_t,size=finddb(i,idxbounds)
            if (dbidx_t!=dbidx):
                print(x)"""
    print("---------------------")
    for x in valsampler:
        dbidx,size0=finddb(x[0],idxbounds)
        for i in x:
            dbidx_t,size=finddb(i,idxbounds)
            if (dbidx_t!=dbidx):
                print(x)
    #print(len(set1))
    #lmdbtest()
    a=1
