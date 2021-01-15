import string

# about data and net
datafilepath_crnn="../GSM_generation/training_data/Word"
alphabet =string.ascii_lowercase
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiment
random_sample = True # whether to sample the dataset with random sampler
nh = 256 # size of the lstm hidden state
pretrained = '' # path to pretrained model (to continue training)
expr_dir = 'expr' # where to store samples and models
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

# training process
displayPerEpoch = 4 # interval to be print the train loss
valPerEpoch = 2 # interval to val the model loss and accuray
valInterval = 100
valEpochInterval = 5
displayInterval = 100
saveInterval = 1000000 # interval to save model
n_val_disp = 10 # number of samples to display when val the model

# finetune
nepoch = 1200 # number of epochs to train for
batchSize = 20 # input batch size
lr = 0.0001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = False # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)
