import torch
import torch.nn as nn
import dataset
import numpy as np
from models.CNN import CNN
from torch.utils.data import random_split,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utility


device = torch.device('cuda')
if not torch.cuda.is_available():
    device = torch.device('cpu')

num_epochs = 75
batch_size = 40
learning_rate = 0.0001
val_interval = 5


# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""
all_dataset = dataset.diskDataset("../GSM_generation/training_data/Word")
dorm_dataset=dataset.diskDataset("../GSM_generation/training_data/Word_jxydorm",max_length=all_dataset.datashape[2],initlabels=all_dataset.label_list)

def data_loader(all_dataset):
    assert all_dataset
    train_length=int(len(all_dataset)*0.85)
    #train_dataset,val_dataset=random_split(all_dataset,[train_length,len(all_dataset)-train_length])
    train_dataset,val_dataset=dataset.int_split(all_dataset,5,0.2)
    test_length=int(len(train_dataset)*0.15)
    test_dataset,no_dataset=random_split(train_dataset,[test_length,len(train_dataset)-test_length])

    train_loader = DataLoader(train_dataset,batch_size=batch_size, \
            shuffle=True)
    # val
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

train_loader, val_loader= data_loader(all_dataset)
dorm_loader= DataLoader(dorm_dataset,batch_size=batch_size, shuffle=True)
# -----------------------------------------------
"""
    data augmentation tranformation
"""

num_classes = len(all_dataset.label_list)
#Model initialization
model = CNN(num_classes,channel_input=all_dataset.channel).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def val(cm_save=False):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():

        """     #跨数据集验证
        correct = 0
        total = 0
        for images, labels, sources in dorm_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #print(outputs.data)
            t, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print (predicted,labels)
            correct += (predicted == labels).sum().item()
        print('Accuracy on the {}  test images: {} %'.format(total , 100 * correct / total))
        """
        #验证集验证
        conf_matrix = torch.zeros(num_classes,num_classes) #初始化混淆矩阵

        correct = 0
        correct_twoclass=0
        total = 0
        error_file_list=[]
        for i, (images, labels,sources) in enumerate(val_loader):
            """for image in images:
                plt.imshow(image.permute(1,2,0))
                print(labels)
                plt.show()"""
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #print(outputs.data)
            t, predicted = torch.max(outputs.data, 1)
            t, predicted_twoclass = torch.topk(outputs.data, 2 , 1)
            total += labels.size(0)
            print (predicted,labels)
            for p, t in zip(predicted, labels):
                conf_matrix[p, t] += 1 #更新混淆矩阵
            correct += (predicted == labels).sum().item()
            for predict,label,source in zip(predicted,labels,sources):
                if (predict!=label):
                    error_file_list.append(source)
            for predict2,truelabel in zip(predicted_twoclass,labels):
                if (truelabel in predicted_twoclass):
                    correct_twoclass+=1
        print('Accuracy on the {} valid images: {} %'.format(total , 100 * correct / total))
        print('Accuracy within two results on the {} valid images: {} %'.format(total , 100 * correct_twoclass / total))
        if cm_save==True:
            np.save('confusion_matrix',conf_matrix.numpy())
            np.save('labels',np.asarray(all_dataset.label_list))
            np.save('wrong_files',error_file_list)
    model.train() #切回训练模式
    return

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels, sources) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    if epoch%val_interval==0:
        val()
    if epoch==num_epochs-1:
        val(cm_save=True)


#torch.save(model.state_dict(), 'modelcnn.ckpt')
