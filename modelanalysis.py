import torch
import utility
import numpy as np

a=torch.Tensor([ [[1,2],[3,4]] ,
            [[5,6],[7,8]] ])
print(a.view((4,2,1)))


conf_matrix=np.load('cfs\confusion_matrix4.npy',allow_pickle = True)
labels=np.load('labels.npy')
#results=np.load('crnn_accuracy.npy')
#print(results)

classes=labels #np.zeros(40)
#utility.plot_error_matrix(conf_matrix.T, classes, normalize=False)
#utility.plot_confusion_matrix(conf_matrix.T, classes, normalize=False)

cm=conf_matrix.T
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i!=j and cm[i,j]!=0:
            print(classes[i],classes[j],cm[i,j])
testnum=np.argwhere(labels=='because')
print(np.sum(conf_matrix,axis=0))
print(cm[testnum,testnum])
