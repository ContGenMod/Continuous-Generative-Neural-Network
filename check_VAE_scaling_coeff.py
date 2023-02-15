

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
X_test_s_c_norm = np.load(path+'XTest_s_c_norm.npy')   
y_test = np.zeros((N_test_set,1,1))
X_test_s_c_norm = np.transpose(X_test_s_c_norm[:,0,0,:])

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

    path = '/home/Desktop/VAE_scaling_coeff/leaky_relu_ZDIMS='+str(ZDIMS)+'_weight_KL='+str(weight_KL)+'_wavelet='+name+'_lr='+str(lr)
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
        M = np.zeros((len_s_c), dtype=float)
        for k in range(len_s_c):
            M[k] = testloader.dataset[i][0][0][k]
        M_aux = np.array(M, dtype=float) 
        imgray_ten = torch.cuda.FloatTensor(M_aux).view(1,1,len_s_c)
        imgray_ten = Variable(imgray_ten)
        mu,logvar = model.encode(imgray_ten)
        z = model.reparametrize(mu,logvar).cuda()
        ricostr = model.decode(z).cpu().detach().numpy()
        ricostr = np.array(ricostr, dtype = np.float32)
        ricostr.squeeze()
        ricostr.resize(len_s_c)
        plt.plot(M,label='original scaling coefficients')
        plt.plot(ricostr,label='reconstructed scaling coefficients')
        plt.legend(loc='upper right')
        plt.savefig(path_title+'scaling_coeff_e_recon_'+str(i+1)+'.jpg')
        plt.close()     
        ricostr = (max_s_c-min_s_c)*ricostr+min_s_c  
        l = l_true
        c_r = np.zeros((c_true.shape))
        c_r[0:l[0]] = ricostr
        ricostr_fun = matlab_waverec(c_r,l_true,name)
        M = np.zeros((len_), dtype=float)
        for k in range(len_):
            M[k] = testloader_fun.dataset[i][0][0][k]
        # plot functions
        plt.plot(t,M,label='original function')
        plt.plot(t,ricostr_fun,label='reconstructed function')
        plt.legend(loc='upper right')
        plt.savefig(path_title+'function_e_recon_'+str(i+1)+'.jpg')
        plt.close()
        # save functions .npy 
        np.save(path_title+'before'+str(i+1)+'.npy',M) 
        np.save(path_title+'after'+str(i+1)+'.npy',ricostr_fun)    
        i+=1

    # Generative power of VAE (generate D(z) with z standard gaussian vector in the latent space)
    i = 0
    n_im = 10
    while i < n_im:
        z = Variable(torch.randn(1, ZDIMS))
        if CUDA:
            z = z.cuda()
        ricostr = model.decode(z).cpu().detach().numpy()
        ricostr = np.array(ricostr, dtype = np.float32)
        ricostr.squeeze()
        ricostr.resize(len_s_c)
        ricostr = (max_s_c-min_s_c)*ricostr+min_s_c 
        plt.plot(ricostr)
        plt.savefig(path_title+'generate_scaling_coeffs_'+str(i+1)+'.jpg')
        plt.close()
        i+=1

    # Generation of 6 signals
    fig, (ax0,ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,6,figsize=(14,1))
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax0.plot(ricostr)
    ax0.set_ylim([0.6*min_X,0.6*max_X])
    ax0.set_yticks([])
    ax0.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax1.plot(ricostr)
    ax1.set_ylim([0.6*min_X,0.6*max_X])
    ax1.set_yticks([])
    ax1.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax2.plot(ricostr)
    ax2.set_ylim([0.6*min_X,0.6*max_X])
    ax2.set_yticks([])
    ax2.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax3.plot(ricostr)
    ax3.set_ylim([0.6*min_X,0.6*max_X])
    ax3.set_yticks([])
    ax3.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax4.plot(ricostr)
    ax4.set_ylim([0.6*min_X,0.6*max_X])
    ax4.set_yticks([])
    ax4.set_xticks([])
    z = Variable(torch.randn(ZDIMS)).cuda()
    ricostr = model.decode(z).cpu().detach().numpy()
    ricostr = np.array(ricostr, dtype = np.float32)
    ricostr.squeeze()
    ricostr.resize(len_s_c)
    ricostr = (max_s_c-min_s_c)*ricostr+min_s_c
    l = l_true
    c_r = np.zeros((c_true.shape))
    c_r[0:l[0]] = ricostr
    ricostr = matlab_waverec(c_r,l_true,name)
    ax5.plot(ricostr)
    ax5.set_ylim([0.6*min_X,0.6*max_X])
    ax5.set_yticks([])
    ax5.set_xticks([])
    plt.savefig(path_title+"Generation.pdf")
    plt.close()




