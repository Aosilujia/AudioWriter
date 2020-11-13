import os
import queue
from operator import itemgetter
import numpy as np


def csvfilelist(directory_name):
    """
    得到一个csv文件路径列表
    :param directory_name
    """
    if not os.path.isdir(directory_name):
        print(path+" is not a directory")
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

def padding(raw_samples, max_length=-1, padding_value=0) -> np.ndarray:
    """
    originated by yyg.对齐所有数据到最大长度或给定长度，输入为[N,C,W,H]
    :param padding_value:
    :param raw_samples: list of sample(numpy array)
    :param max_length: if max_length == -1,then we use max_len of raw_sample,else we use max_length passed in
    :return:
    """
    shapes = list(map(np.shape, raw_samples))
    lengths = list(map(itemgetter(1),shapes))
    if max_length < 0:
        max_length = max(lengths)
    new_shape = [len(shapes),shapes[0][0],max_length,shapes[0][2]]
    padding_data = np.zeros(new_shape) + padding_value
    for idx, seq in enumerate(raw_samples):
        padding_data[idx,:,:lengths[idx]] = seq
    return padding_data
