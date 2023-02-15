
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

N_test_set = 2000  # number of elements of the test set

len_ = np.load(path+'len_t.npy')  
min_X = np.load(path+'min_X.npy')
max_X = np.load(path+'max_X.npy')
   
X_test_norm = np.load(path+'XTest_norm.npy')   
y_test = np.zeros((N_test_set,1,1))
X_test_norm = np.transpose(X_test_norm[:,0,0,:])

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
                 
    path = '/home/Desktop/discrete_VAE/leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'_wavelet='+name+'_lr='+str(lr)
    path_title = path+'/'

    device = torch.device('cuda')
    model = VAE()
    path_s = 'state_dict_model_leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'.pt'
    model.to(device)
    model.load_state_dict(torch.load(path_title+path_s, map_location = 'cuda:0'))
    model.eval()

    # Reconstruction power of VAE (compare D(E(x)) with x)
    i = 0
    n_im = 10
    while i < n_im:
        M = np.zeros((len_), dtype=float)
        for k in range(len_):
            M[k] = testloader.dataset[i][0][0][k]
        original_fun = (max_X-min_X)*M+min_X
        M_aux = np.array(M, dtype=float) 
        imgray_ten = torch.cuda.FloatTensor(M_aux).view(1,1,len_)
        imgray_ten = Variable(imgray_ten)
        mu,logvar = model.encode(imgray_ten)
        z = model.reparametrize(mu,logvar).cuda()
        ricostr = model.decode(z).cpu().detach().numpy()
        ricostr = np.array(ricostr, dtype = np.float32)
        ricostr.squeeze()
        ricostr.resize(len_)
        ricostr_fun = (max_X-min_X)*ricostr+min_X
        # plot functions
        plt.plot(t,original_fun,label='original function')
        plt.plot(t,ricostr_fun,label='reconstructed function')
        plt.legend(loc='upper right')
        plt.savefig(path_title+'function_e_recon_'+str(i+1)+'.jpg')
        plt.close()
        # save functions
        np.save(path_title+'before'+str(i+1)+'.npy',original_fun) 
        np.save(path_title+'after'+str(i+1)+'.npy',ricostr_fun)  
        i+=1

    # Generative power of VAE (generate D(z) with z standard gaussian vector in the latent space)
    i = 0
    n_im = 10
    while i < n_im:
        z = Variable(torch.randn(ZDIMS))
        if CUDA:
            z = z.cuda()
        ricostr = model.decode(z).cpu().detach().numpy()
        ricostr = np.array(ricostr, dtype = np.float32)
        ricostr.squeeze()
        ricostr.resize(len_)
        ricostr = (max_X-min_X)*ricostr+min_X
        plt.plot(ricostr)
        plt.savefig(path_title+'generate_function_'+str(i+1)+'.jpg')
        plt.close()
        i+=1

    # Generation of 6 signals
    fig, (ax0,ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,6,figsize=(14,1))
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax0.plot(ricostr)
    ax0.set_ylim([0.6*min_X,0.6*max_X])
    ax0.set_yticks([])
    ax0.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax1.plot(ricostr)
    ax1.set_ylim([0.6*min_X,0.6*max_X])
    ax1.set_yticks([])
    ax1.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax2.plot(ricostr)
    ax2.set_ylim([0.6*min_X,0.6*max_X])
    ax2.set_yticks([])
    ax2.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax3.plot(ricostr)
    ax3.set_ylim([0.6*min_X,0.6*max_X])
    ax3.set_yticks([])
    ax3.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax4.plot(ricostr)
    ax4.set_ylim([0.6*min_X,0.6*max_X])
    ax4.set_yticks([])
    ax4.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_)
    ricostr = (max_X-min_X)*ricostr+min_X
    ax5.plot(ricostr)
    ax5.set_ylim([0.6*min_X,0.6*max_X])
    ax5.set_yticks([])
    ax5.set_xticks([])
    plt.savefig(path_title+"Generation.pdf")
    plt.close()






