import torch
import torch.nn as nn
import dataset
from models.CNN import CNN
from torch.utils.data import random_split,DataLoader

device = torch.device('cpu')

num_epochs = 5
num_classes = 2
batch_size = 10
learning_rate = 0.0001


# -----------------------------------------------
"""
In this block
    Get train and val data_loader
"""

def data_loader():
    all_dataset = dataset.AlgeDataset("../GSM_generation/training_data")
    assert all_dataset
    train_length=int(len(all_dataset)*0.5)
    train_dataset,val_dataset=random_split(all_dataset,[train_length,len(all_dataset)-train_length])

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, \
            shuffle=True)
    # val
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

train_loader, val_loader = data_loader()
# -----------------------------------------------

model = CNN(num_classes).to(device)

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
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        print (predicted,labels)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(len(val_loader),100 * correct / total))

torch.save(model.state_dict(), 'modelcnn.ckpt')
