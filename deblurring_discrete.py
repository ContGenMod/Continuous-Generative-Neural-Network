

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
X_test = np.load(path+'XTest.npy')   
X_test = np.transpose(X_test[:,0,0,:])
y_test = np.zeros((N_test_set,1,1))

# test set   
class testData(Dataset):      
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)   
X_testz = torch.FloatTensor(X_test).view(-1,1,len_) 
test_data = testData(X_testz, torch.FloatTensor(y_test))
testloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
# --------------------------------------------------------------------------------------------------------------------
# Functions

def mse(x,y):
    return np.sum((x - y)**2)/np.sum(x**2)

def landweber_deep_deblurring(z_prec,A,y,x):
    z_prec_ten = torch.cuda.FloatTensor(z_prec)
    x_prec_ten = model.decode(z_prec_ten[:,0]).cpu()
    x_prec = x_prec_ten.cpu().detach().numpy() 
    x_prec = np.array(x_prec,dtype = np.float32)
    x_prec.squeeze()
    x_prec.resize(len_,1)
    # denormalize x_prec
    x_prec = (max_X-min_X)*x_prec+min_X
    y = np.reshape(y,(len_,1)) 
    arg_2 = np.dot(np.transpose(A),np.dot(A,x_prec)-y)
    # derivative of denormalization arg_2
    arg_2 = (max_X-min_X)*arg_2    
    der_decoder_ten = torch.autograd.functional.jacobian(model.decode,z_prec_ten[:,0])
    der_decoder = der_decoder_ten.cpu().detach().numpy()
    der_decoder = np.array(der_decoder, dtype = np.float32)
    der_decoder.resize(len_,ZDIMS)
    arg_1 = np.transpose(der_decoder)   
    return np.dot(arg_1,arg_2)  
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

	path = "/home/Desktop/discrete_VAE/"
	title = "leaky_relu_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"_wavelet=db"+str(N)+"_lr="+str(lr)
	path_title = path+title+"/"
	device = torch.device("cuda")
	model = VAE()
	path_s = "state_dict_model_leaky_relu_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+".pt"
	model.to(device)
	model.load_state_dict(torch.load(path_title+path_s, map_location = "cuda:0"))
	model.eval()

    SEED=1
    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    # Original image from the test set
    i = 2
    M = np.zeros((len_), dtype=float)
    for k in range(len_):
        M[k] = testloader.dataset[i][0][0][k]    
    x = np.reshape(M,(len_,1)) 

    # gaussian denoising + gaussian deblurring
    eps = 0.2  # noise level
    e = eps*np.random.randn(len_)

    # gaussian filter
    len_fil = 1501
    mu = 0
    sigma = 2
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    value = np.linspace(-4,4,len_fil)
    fil = gaussian(value,mu,sigma)
    fil /= np.sum(fil)
    A = np.zeros((len_,len_))
    k = 0
    for i in range(len_):
        x_aux = np.zeros((len_))
        x_aux[i] = 1
        aux = signal.convolve(x_aux,fil,mode="same")
        A[i,:] = np.reshape(aux,(len_))

    y = np.dot(A,x)+e
    y = np.reshape(y,(len_,1)) 
    mse_original = mse(x,y)

    path = "/home/silvias/Desktop/Deconvolution_discrete/" 
    s_d_l = 0.00025  # Landweber step size
    K = 7500  # number of iteration of Landweber

    title = "eps="+str(cost)+"_s_d_l="+str(s_d_l)+"_K="+str(K)+"_weight_KL="+str(weight_KL)
    os.mkdir(path+title)
    path = path+title+"/"

    # Initialize DL
    z_in = np.random.randn(ZDIMS,1)
    z_in_ten = torch.cuda.FloatTensor(z_in)
    aux = model.decode(torch.cuda.FloatTensor(z_in_ten[:,0])).cpu()
    aux = aux.cpu().detach().numpy() 
    aux = np.array(aux,dtype = np.float32)
    aux.squeeze()
    aux.resize(len_,1)
    aux = (max_X-min_X)*aux+min_X
    im_0_d_l = np.array(aux,dtype = np.float32)
    im_0_d_l.squeeze()
    im_0_d_l.resize(len_,1)
    mse_initial = mse(x,im_0_d_l)  
#----------------------------------------------------------------------------------------------------------------------

    z_prec = z_in
    mse_d_l_array = []
    k = 0
    while k < K:
        derivative = np.reshape(landweber_deep_deblurring(z_prec,A,y,x),(ZDIMS,1))  
        z_new = z_prec - s_d_l * derivative
        z_prec = z_new
        x_final_d_l = model.decode(torch.cuda.FloatTensor(z_prec[:,0])).cpu()
        x_final_d_l = np.array(x_final_d_l.cpu().detach().numpy(),dtype = np.float32) 
        x_final_d_l.squeeze()
        x_final_d_l = (max_X-min_X)*x_final_d_l+min_X
        x_final_d_l.resize(len_,1)
        mse_d_l_array.append(mse(x,x_final_d_l))
        k += 1

    mse_d_l = mse(x,x_final_d_l)

    # Table
	table = [["", "MSE"]]
	table.append(["Original",mse_original])
	table.append(["Initial",mse_initial])
	table.append(["Deep Landweber",mse_d_l])
	T = tabulate(table, headers='firstrow', tablefmt='fancy_grid')
	f = open(path+'table_MSE='+str(mse_original)+'.txt', 'w')
	f.write(T)
	f.close()

    # Images
	fig, ax = plt.subplots()
	ax.plot(im_0_d_l)
	ax.plot(x)
	ax.set_yticks([])
	ax.set_xticks([])
	plt.savefig(path+"Initial_guess_MSE="+str(mse_original)+".pdf")
	plt.close()

	fig, ax = plt.subplots()
	ax.plot(x_final_d_l)
	ax.plot(x)
	ax.set_yticks([])
	ax.set_xticks([])
	plt.savefig(path+"Final_debl_MSE="+str(mse_original)+".pdf")
	plt.close()

	fig, ax = plt.subplots()
	ax.plot(y)
	ax.plot(x)
	ax.set_yticks([])
	ax.set_xticks([])
	plt.savefig(path+"Initial_debl_MSE="+str(mse_original)+".pdf")
	plt.close()

	plt.plot(mse_d_l_array)
	plt.title('MSE function along iterations')
	plt.savefig(path+"Mse_function_MSE="+str(mse_original)+".pdf")
	plt.close()

	# save functions
	np.save(path+'x.npy',x) 
	np.save(path+'y.npy',y)
	np.save(path+'x_final_d_l.npy',x_final_d_l) 
	np.save(path+'im_0_d_l.npy',im_0_d_l) 







