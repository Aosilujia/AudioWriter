import utility
import numpy as np

conf_matrix=np.load('confusion_matrix.npy',allow_pickle = True)
labels=np.load('labels.npy')
classes=labels #np.zeros(40)
utility.plot_confusion_matrix(conf_matrix.T, classes, normalize=False)
