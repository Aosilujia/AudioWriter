import torch
import torch.nn as nn
import dataset
from models.CNN import CNN
from torch.utils.data import random_split,DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utility


device = torch.device('cuda')
if not torch.cuda.is_available():
    device = torch.device('cpu')

num_epochs = 50
batch_size = 10
learning_rate = 0.0001


# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""

data_transform=transforms.Compose([transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5,0.5))])
all_dataset = dataset.Dataset("../GSM_generation/training_data/Word")
dorm_dataset=dataset.Dataset("../GSM_generation/training_data/Word_jxydorm",max_length=450,initlabels=all_dataset.label_list)

def data_loader(all_dataset):
    assert all_dataset
    train_length=int(len(all_dataset)*0.85)
    train_dataset,val_dataset=random_split(all_dataset,[train_length,len(all_dataset)-train_length])

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

num_classes = len(all_dataset.label_list)


model = CNN(num_classes,channel_input=all_dataset.channel).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():

    #跨数据集验证
    correct = 0
    total = 0
    for images, labels in dorm_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #print(outputs.data)
        t, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print (predicted,labels)
        correct += (predicted == labels).sum().item()
    print('Accuracy on the {}  test images: {} %'.format(total , 100 * correct / total))

    #验证集验证
    conf_matrix = torch.zeros(num_classes,num_classes) #初始化混淆矩阵

    correct = 0
    correct_twoclass=0
    total = 0
    for images, labels in val_loader:
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
        for predict2,truelabel in zip(predicted_twoclass,labels):
            if (truelabel in predicted_twoclass):
                correct_twoclass+=1
    print('Accuracy on the {} valid images: {} %'.format(total , 100 * correct / total))
    print('Accuracy within two results on the {} valid images: {} %'.format(total , 100 * correct_twoclass / total))
    numpy.save('çonfusion_matrix',conf_matrix.numpy)
#torch.save(model.state_dict(), 'modelcnn.ckpt')
