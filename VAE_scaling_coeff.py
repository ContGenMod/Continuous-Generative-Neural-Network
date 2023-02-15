
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
N = 1  # number of Vanishing moments of the Daubechies wavelet
n = 6  # number of downsampling done to obtain the scaling coefficients
name = 'db'+str(N)
path = '/home/Desktop/Dataset/n='+str(n)+'/N='+str(N)+'/'

N_training_set = 10000  # number of elements of the training set
N_test_set = 2000  # number of elements of the test set

len_s_c = np.load(path+'len_s_c.npy')
X_train_s_c_norm = np.load(path+'XTrain_s_c_norm.npy')   
X_test_s_c_norm = np.load(path+'XTest_s_c_norm.npy')   
y_train = np.zeros((N_training_set,1,1))
y_test = np.zeros((N_test_set,1,1))
X_train_s_c_norm = np.transpose(X_train_s_c_norm[:,0,0,:])
X_test_s_c_norm = np.transpose(X_test_s_c_norm[:,0,0,:])

# training set
class trainData(Dataset):     
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)
X_trainz = torch.FloatTensor(X_train_s_c_norm).view(-1,1,len_s_c)
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
X_testz = torch.FloatTensor(X_test_s_c_norm).view(-1,1,len_s_c) 
test_data = testData(X_testz, torch.FloatTensor(y_test))
trainloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
# --------------------------------------------------------------------------------------------------------------------

# Import eta
path = '/data/matlab/eta_values/csv/'
reader = csv.reader(open(path+'eta_'+name+'.csv', 'r'), delimiter=',')
eta = list(reader)
eta = np.array(eta).astype("float")
eta = np.reshape(eta,[1,1,eta.shape[1]])
eta = torch.from_numpy(eta)
eta = eta.cuda()
eta = eta.type(torch.cuda.FloatTensor)
len_eta = eta.shape[2]

reader = csv.reader(open(path+'eta_bar_'+name+'.csv', 'r'), delimiter=',')
eta_bar = list(reader)
eta_bar = np.array(eta_bar).astype("float")
eta_bar = np.reshape(eta_bar,[1,1,eta_bar.shape[1]])
eta_bar = torch.from_numpy(eta_bar)
eta_bar = eta_bar.cuda()
eta_bar = eta_bar.type(torch.cuda.FloatTensor)
len_eta_bar = eta_bar.shape[2]
# --------------------------------------------------------------------------------------------------------------------

for ZDIMS in ZDIMS_vett:
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()
            # ENCODER
            self.conv_filtri_list = nn.ModuleList([])
            for p in range(42):
                self.conv_filtri_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, padding=len_eta_bar-1, stride=1))
                self.conv_filtri_list[p].bias = torch.nn.Parameter(torch.zeros((1)).type(torch.cuda.FloatTensor))
                self.conv_filtri_list[p].bias.requires_grad = False
            if N == 1 or N == 2:
                self.fc1 = nn.Linear(64, ZDIMS)
                self.fc2 = nn.Linear(64, ZDIMS) 
            elif N == 6:
                self.fc1 = nn.Linear(72, ZDIMS)
                self.fc2 = nn.Linear(72, ZDIMS) 
            elif N == 10:
                self.fc1 = nn.Linear(80, ZDIMS)
                self.fc2 = nn.Linear(80, ZDIMS)                          
            self.sigmoid = nn.Sigmoid()
            self.leaky_relu = nn.LeakyReLU()
            # DECODER
            if N == 1 or N == 2:
                self.fc3 = nn.Linear(ZDIMS, 64)
            elif N == 6:
                self.fc3 = nn.Linear(ZDIMS, 72)
            elif N == 10:
                self.fc3 = nn.Linear(ZDIMS, 80)
            self.conv_filtri_t_list = nn.ModuleList([])
            for p in range(42):
                self.conv_filtri_t_list.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, padding=len_eta-1, stride=1))
                self.conv_filtri_t_list[p].bias = torch.nn.Parameter(torch.zeros((1)).type(torch.cuda.FloatTensor))
                self.conv_filtri_t_list[p].bias.requires_grad = False

            self.N = torch.distributions.Normal(0,1)
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

        def encode(self, x: Variable) -> (Variable, Variable):
            x = x.view(-1, 1, len_s_c)
            # convolution and projection 
            len_bar = self.conv_filtri_list[0](eta_bar).shape[2]
            weights_conv = torch.empty(size=(2,1,len_bar)).type(torch.cuda.FloatTensor)
            for i in range(2):
                for j in range(1):
                    weights_conv[i,j,:] = self.conv_filtri_list[i](eta_bar)
            if name == 'db1':
                x = F.conv1d(x,weights_conv,bias=torch.zeros((2)).cuda(),padding=4, stride=2)
            elif name == 'db2':
                x = F.conv1d(x,weights_conv,bias=torch.zeros((2)).cuda(),padding=4, stride=2)
            elif name == 'db6':
                x = F.conv1d(x,weights_conv,bias=torch.zeros((2)).cuda(),padding=12, stride=2)
            elif name == 'db10':
                x = F.conv1d(x,weights_conv,bias=torch.zeros((2)).cuda(),padding=15, stride=2)
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            # convolution and projection               
            len_bar = self.conv_filtri_list[0](eta_bar).shape[2]
            weights_conv_1 = torch.empty(size=(4,2,len_bar)).type(torch.cuda.FloatTensor)
            for i in range(4):
                for j in range(2):
                    weights_conv_1[i,j,:] = self.conv_filtri_list[j+2*i+2](eta_bar)
            if name == 'db1':
                x = F.conv1d(x,weights_conv_1,bias=torch.zeros((4)).cuda(),padding=4, stride=2)
            elif name == 'db2':
                x = F.conv1d(x,weights_conv_1,bias=torch.zeros((4)).cuda(),padding=4, stride=2)
            elif name == 'db6':
                x = F.conv1d(x,weights_conv_1,bias=torch.zeros((4)).cuda(),padding=11, stride=2)
            elif name == 'db10':
                x = F.conv1d(x,weights_conv_1,bias=torch.zeros((4)).cuda(),padding=14, stride=2)
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            # convolution and projection
            len_bar = self.conv_filtri_list[0](eta_bar).shape[2]
            weights_conv_2 = torch.empty(size=(8,4,len_bar)).type(torch.cuda.FloatTensor)
            for i in range(8):
                for j in range(4):
                    weights_conv_2[i,j,:] = self.conv_filtri_list[j+4*i+10](eta_bar)
            if name == 'db1':
                x = F.conv1d(x,weights_conv_2,bias=torch.zeros((8)).cuda(),padding=4, stride=2)
            elif name == 'db2':
                x = F.conv1d(x,weights_conv_2,bias=torch.zeros((8)).cuda(),padding=4, stride=2)
            elif name == 'db6':
                x = F.conv1d(x,weights_conv_2,bias=torch.zeros((8)).cuda(),padding=12, stride=2)
            elif name == 'db10':
                x = F.conv1d(x,weights_conv_2,bias=torch.zeros((8)).cuda(),padding=15, stride=2)
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            x = x.view(-1,x.shape[1]*x.shape[2])
            mu_z = self.fc1(x)       
            logvar_z = self.fc2(x)    
            return mu_z, logvar_z    
        def decode(self, z: Variable) -> Variable:
            x = self.fc3(z)
            x = x.view(-1,8,int(len_s_c/8))
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            # deconvolution and projection
            len_ = self.conv_filtri_t_list[0](eta).shape[2]
            weights_conv_t = torch.empty(size=(8,4,len_)).type(torch.cuda.FloatTensor)
            for i in range(8):
                for j in range(4):
                    weights_conv_t[i,j,:] = self.conv_filtri_t_list[j+4*i+10](eta)
            if name == 'db1':
                x = F.conv_transpose1d(x,weights_conv_t,bias=torch.zeros((4)).cuda(),padding=7, stride=2)
            elif name == 'db2':
                x = F.conv_transpose1d(x,weights_conv_t,bias=torch.zeros((4)).cuda(),padding=12, output_padding=1, stride=2)
            elif name == 'db6':
                x = F.conv_transpose1d(x,weights_conv_t,bias=torch.zeros((4)).cuda(),padding=22, stride=2)
            elif name == 'db10':
                x = F.conv_transpose1d(x,weights_conv_t,bias=torch.zeros((4)).cuda(),padding=34, stride=2)
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            # deconvolution and projection
            len_ = self.conv_filtri_t_list[0](eta).shape[2]
            weights_conv_t_1 = torch.empty(size=(4,2,len_)).type(torch.cuda.FloatTensor)
            for i in range(4):
                for j in range(2):
                    weights_conv_t_1[i,j,:] = self.conv_filtri_t_list[j+2*i+2](eta)
            if name == 'db1':
                x = F.conv_transpose1d(x,weights_conv_t_1,bias=torch.zeros((2)).cuda(),padding=7, stride=2)
            elif name == 'db2':
                x = F.conv_transpose1d(x,weights_conv_t_1,bias=torch.zeros((2)).cuda(),padding=11, stride=2)
            elif name == 'db6':
                x = F.conv_transpose1d(x,weights_conv_t_1,bias=torch.zeros((2)).cuda(),padding=22, output_padding=1, stride=2)
            elif name == 'db10':
                x = F.conv_transpose1d(x,weights_conv_t_1,bias=torch.zeros((2)).cuda(),padding=34, output_padding=1, stride=2)
            # nonlinearity
            x = 0.3*self.leaky_relu(x)
            # deconvolution and projection 
            len_ = self.conv_filtri_t_list[0](eta).shape[2]
            weights_conv_t_2 = torch.empty(size=(2,1,len_)).type(torch.cuda.FloatTensor)
            for i in range(2):
                for j in range(1):
                    weights_conv_t_2[i,j,:] = self.conv_filtri_t_list[i](eta)
            if name == 'db1':
                x = F.conv_transpose1d(x,weights_conv_t_2,bias=torch.zeros((1)).cuda(),padding=7, stride=2)
            elif name == 'db2':
                x = F.conv_transpose1d(x,weights_conv_t_2,bias=torch.zeros((1)).cuda(),padding=12, output_padding=1, stride=2)
            elif name == 'db6':
                x = F.conv_transpose1d(x,weights_conv_t_2,bias=torch.zeros((1)).cuda(),padding=22, stride=2)
            elif name == 'db10':
                x = F.conv_transpose1d(x,weights_conv_t_2,bias=torch.zeros((1)).cuda(),padding=34, stride=2)
            # nonlinearity
            x = self.sigmoid(x)
            return x

        def reparametrize(self, mu: Variable, logvar: Variable) -> Variable:
            sigma = torch.exp(logvar)
            z = mu + sigma*self.N.sample(mu.shape)
            return z

        def forward(self, x: Variable) -> (Variable, Variable, Variable):
            mu, logvar = self.encode(x.view(-1,len_s_c))
            sigma = torch.exp(logvar)
            z = mu + sigma*self.N.sample(mu.shape)
            return self.decode(z), mu, logvar

        def loss_function(self, recon_x, x, mu, logvar) -> (Variable,Variable,Variable):
            MSE = F.mse_loss(recon_x.view(-1,len_s_c), x.view(-1,len_s_c))
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
                                    recon_batch.view(BATCH_SIZE, 1, len_s_c)[:n]])
        test_loss /= len(testloader.dataset)
        test_loss_MSE /= len(testloader.dataset)
        test_loss_KL /= len(testloader.dataset)
        print('====> Epoch: {} Test loss: {:.10f}'.format(epoch, test_loss))
        print('====> Epoch: {} Test MSE: {:.10f}'.format(epoch, test_loss_MSE))
        print('====> Epoch: {} Test KLD: {:.10f}'.format(epoch, weight_KL*test_loss_KL))
        y_t.append(test_loss)
        y_t_MSE.append(test_loss_MSE)
        y_t_KL.append(test_loss_KL)


    path = '/home/Desktop/VAE_scaling_coeff/'
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











