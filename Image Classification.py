#Moazam Khalid
#161051763

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import matplotlib.pyplot as plt
import numpy as np;
import matplotlib.pyplot as plt


train_set = torchvision.datasets.FashionMNIST(root = ".", train = True,
                                             download = True, transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train = False,
                                             download = True, transform = transforms.ToTensor())

training_loader = torch.utils.data.DataLoader(train_set, batch_size =32,
                                             shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32,
                                         shuffle = False)
torch.manual_seed(0)





class Nnet(nn.Module):
    
    #Weight initialisation with Xavier normal
    def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
                  
    def __init__(self):
        super(Nnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 0),#1 in, 32 out. Due to getting mismatch errors, changing the padding seemed to work 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 0),#32 from previous later goes in, 64 comes out
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #part c
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.ELU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc1 = nn.Linear(4*4*64, 256)
        self.fc2 = nn.Linear(256, 10)
        #part d
        #self.drop_out = nn.Dropout(0.3)
        
        self.initialize_weights()
                 
    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        
        return y
   






    model = Nnet()
    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning)
    total_step = len(training_loader)
    loss_list = []
    acc_list = []
    epochs = 50
    learning = 0.1 
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_loader): #running the forward pass here
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            #back propagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #SGD optimizer

            #Tracking the accuracy here
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch {}:{}, Loss: {:.3f}, Accuracy: {:.2f}%'.format(epoch + 1, epochs, i + 1, loss.item(),(correct / total) * 100))










#To Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy : {} %'.format((correct / total) * 100))
      
#To plot a graph     

#p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='Neural Netwok results')
#p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
#p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
#p.line(np.arange(len(loss_list)), loss_list)
#p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
#show(p)


# In[54]:


#Just for testing purposes
#Not relavent for question 
#import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

#def matplotlib_imshow(img, one_channel=False):
    #if one_channel:
      #  img = img.mean(dim=0)
   # img = img / 2 + 0.5     # unnormalize
   # npimg = img.numpy()
  #  if one_channel:
  #      plt.imshow(npimg, cmap="Greys")
 #   else:
 #       plt.imshow(np.transpose(npimg, (1, 2, 0)))
#writer = SummaryWriter('runs/Question2')
#
#dataiter = iter(training_loader)
#images, labels = dataiter.next()

# create grid of images
#img_grid = torchvision.utils.make_grid(images)

# show images
#matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
#writer.add_image('four_fashion_mnist_images', img_grid)


# In[7]:



#labels_map = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              #7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};
#fig = plt.figure(figsize=(8,8));
#columns = 4;
#rows = 5;
#for i in range(1, columns*rows +1):
  #  img_xy = np.random.randint(len(train_set));
  #  img = train_set[img_xy][0][0,:,:]
  #  fig.add_subplot(rows, columns, i)
  ##  plt.title(labels_map[train_set[img_xy][1]])
  #  plt.axis('off')
  #  plt.imshow(img, cmap='gray')
#plt.show()


# In[233]:


#images


# In[199]:


#plt.imshow(images[0].view(28, 28))
#plt.show()


# In[200]:


#print(labels[0])


# In[ ]:




