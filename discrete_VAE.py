
import cv2
import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import pywt
import csv
from torch.utils.data import Dataset, DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True
SEED = 1
LOG_INTERVAL = 10
no_of_sample = 10

EPOCHS = 1200  # number of epochs
BATCH_SIZE = 2000  # batch size
ZDIMS_vett = np.arange(start=15,stop=16,step=5,dtype=int)  # vector of possible dimensions of the latent space
weight_KL = 0.000001  # weight of the Kullback-Leibler divergence in the loss
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# -------------------------------------------------------------------------------------------------------------------
N = 1  # number of Vanishing moments of the Daubechies wavelet (Only to take the training set)
n = 6  # number of downsampling done to obtain the scaling coefficients
name = 'db'+str(N)
path = '/home/Desktop/Dataset/n='+str(n)+'/N='+str(N)+'/'

N_training_set = 10000  # number of elements of the training set
N_test_set = 2000  # number of elements of the test set

len_ = np.load(path+'len_t.npy')  
min_X = np.load(path+'min_X.npy')
max_X = np.load(path+'max_X.npy')

X_train_norm = np.load(path+'XTrain_norm.npy')   
X_test_norm = np.load(path+'XTest_norm.npy')   
y_train = np.zeros((N_training_set,1,1))
y_test = np.zeros((N_test_set,1,1))
X_train_norm = np.transpose(X_train_norm[:,0,0,:])
X_test_norm = np.transpose(X_test_norm[:,0,0,:])

# training set
class trainData(Dataset):     
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)
X_trainz = torch.FloatTensor(X_train_norm).view(-1,1,len_)
train_data = trainData(X_trainz, torch.FloatTensor(y_train)) 

# test set   
class testData(Dataset):      
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)   
X_testz = torch.FloatTensor(X_test_norm).view(-1,1,len_) 
test_data = testData(X_testz, torch.FloatTensor(y_test))
trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
# --------------------------------------------------------------------------------------------------------------------

for ZDIMS in ZDIMS_vett:
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            # ENCODER
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=4, padding=1, stride=2)    
            self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=4, padding=1, stride=2)
            self.conv3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=4, padding=1, stride=2)    
            self.fc1 = nn.Linear(4096, ZDIMS)
            self.fc2 = nn.Linear(4096, ZDIMS)

            self.sigmoid = nn.Sigmoid()
            self.leaky_relu = nn.LeakyReLU()
   
            # DECODER
            self.fc3 = nn.Linear(ZDIMS,4096)
            self.convt1 = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=4, padding=1, stride=2)
            self.convt2 = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=4, padding=1, stride=2)
            self.convt3 = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=4, padding=1, stride=2)

            self.N = torch.distributions.Normal(0,1)
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        def encode(self, x: Variable) -> (Variable, Variable):
            x = x.view(-1, 1, len_)
            x = 0.3*self.leaky_relu(self.conv1(x))
            x = 0.3*self.leaky_relu(self.conv2(x))
            x = 0.3*self.leaky_relu(self.conv3(x))
            x = x.view(-1,4096)
            mu_z = self.fc1(x)       
            logvar_z = self.fc2(x)    
            return mu_z, logvar_z
    
        def decode(self, z: Variable) -> Variable:
            x = 0.3*self.leaky_relu(self.fc3(z))
            x = x.view(-1,8,512)
            x = 0.3*self.leaky_relu(self.convt1(x))
            x = 0.3*self.leaky_relu(self.convt2(x))
            x = self.sigmoid(self.convt3(x))
            return x

        def reparametrize(self, mu: Variable, logvar: Variable) -> Variable:
            sigma = torch.exp(logvar)
            z = mu + sigma*self.N.sample(mu.shape)
            return z

        def forward(self, x: Variable) -> (Variable, Variable, Variable):
            mu, logvar = self.encode(x.view(-1,len_))
            sigma = torch.exp(logvar)
            z = mu + sigma*self.N.sample(mu.shape)
            return self.decode(z), mu, logvar

        def loss_function(self, recon_x, x, mu, logvar) -> (Variable,Variable,Variable):
            MSE = F.mse_loss(recon_x.view(-1,len_), x.view(-1,len_))
            sigma = torch.exp(logvar)
            KLD = (sigma.pow(2)+mu.pow(2)-torch.log(sigma)-0.5).sum()
            KLD /=  len(recon_x)
            return MSE+weight_KL*KLD,MSE,KLD      

    model = VAE()
    if CUDA:
        model.cuda()

    lr=0.01
    optimizer = optim.Adam(model.parameters(),lr=lr)

    def train(epoch):
        model.train()
        train_loss = 0
        train_loss_MSE = 0
        train_loss_KL = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = Variable(data)
            if CUDA:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)[0]
            loss_MSE = model.loss_function(recon_batch, data, mu, logvar)[1]
            loss_KL = model.loss_function(recon_batch, data, mu, logvar)[2]
            loss.backward()
            train_loss += loss.cpu().data.numpy()
            train_loss_MSE += loss_MSE.cpu().data.numpy()
            train_loss_KL += loss_KL.cpu().data.numpy()
            optimizer.step()
        train_loss /= len(trainloader.dataset)
        train_loss_MSE /= len(trainloader.dataset)
        train_loss_KL /= len(trainloader.dataset)
        print('====> Epoch: {} Train loss: {:.10f}'.format(epoch, train_loss))
        print('====> Epoch: {} Train MSE: {:.10f}'.format(epoch, train_loss_MSE))
        print('====> Epoch: {} Train KLD: {:.10f}'.format(epoch, weight_KL*train_loss_KL))
        y_a.append(train_loss)
        y_a_MSE.append(train_loss_MSE)
        y_a_KL.append(train_loss_KL)

    def test(epoch):
        model.eval()
        test_loss = 0
        test_loss_MSE = 0
        test_loss_KL = 0
        for i, (data, _) in enumerate(testloader):
            if CUDA:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar = model(data)
            test_loss += model.loss_function(recon_batch, data, mu, logvar)[0].item()
            test_loss_MSE += model.loss_function(recon_batch, data, mu, logvar)[1].item()
            test_loss_KL += model.loss_function(recon_batch, data, mu, logvar)[2].item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 1, len_)[:n]])
        test_loss /= len(testloader.dataset)
        test_loss_MSE /= len(testloader.dataset)
        test_loss_KL /= len(testloader.dataset)
        print('====> Epoch: {} Test loss: {:.10f}'.format(epoch, test_loss))
        print('====> Epoch: {} Test MSE: {:.10f}'.format(epoch, test_loss_MSE))
        print('====> Epoch: {} Test KLD: {:.10f}'.format(epoch, weight_KL*test_loss_KL))
        y_t.append(test_loss)
        y_t_MSE.append(test_loss_MSE)
        y_t_KL.append(test_loss_KL)


    path = '/home/Desktop/discrete_VAE/'
    title = 'leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'_wavelet='+name+'_lr='+str(lr)
    os.mkdir(path+title)
    path_title = path+title+'/'

    y_a = []
    y_t = []
    y_a_MSE = []
    y_t_MSE = []
    y_a_KL = []
    y_t_KL = []
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
    
    path_s = 'state_dict_model_leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'.pt'
    torch.save(model.state_dict(),path_title+path_s)

    # Plot total loss function training from epoch 30
    x = list(range(30,EPOCHS+1))
    x_a = np.array(x)
    y_a_a = np.array(y_a)
    y_t_a = np.array(y_t)
    y_a_a = y_a_a[29:len(y_a_a)]
    y_t_a = y_t_a[29:len(y_t_a)]
    plt.plot(x_a,y_a_a, label= 'Train loss')
    plt.plot(x_a,y_t_a, label='Test loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    path_l = 'Loss_Adam_lr='+str(lr)+'_leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'.jpg'
    plt.savefig(path_title+path_l)
    plt.close()

    # Plot MSE loss function training from epoch 30
    y_a_a = np.array(y_a_MSE)
    y_t_a = np.array(y_t_MSE)
    y_a_a = y_a_a[29:len(y_a_a)]
    y_t_a = y_t_a[29:len(y_t_a)]
    plt.plot(x_a,y_a_a, label= 'Train MSE')
    plt.plot(x_a,y_t_a, label='Test MSE')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    path_l = 'MSE_Adam_lr='+str(lr)+'_leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'.jpg'
    plt.savefig(path_title+path_l)
    plt.close()

    # Plot KL loss function training from epoch 30
    y_a_a = np.array(y_a_KL)
    y_t_a = np.array(y_t_KL)
    y_a_a = y_a_a[29:len(y_a_a)]
    y_t_a = y_t_a[29:len(y_t_a)]
    plt.plot(x_a,y_a_a, label= 'Train KLD')
    plt.plot(x_a,y_t_a, label='Test KLD')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    path_l = 'KLD_Adam_lr='+str(lr)+'_leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'.jpg'
    plt.savefig(path_title+path_l)
    plt.close()











