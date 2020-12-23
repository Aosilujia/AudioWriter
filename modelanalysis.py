import utility
import numpy as np

conf_matrix=np.load('confusion_matrix0.npy',allow_pickle = True)
labels=np.load('labels1.npy')
classes=labels #np.zeros(40)
utility.plot_confusion_matrix(conf_matrix, classes, normalize=False)
