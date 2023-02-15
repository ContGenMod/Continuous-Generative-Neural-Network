
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
from scipy import signal
from tabulate import tabulate
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

N_test_set = 2000  # number of elements of the test set

len_s_c = np.load(path+'len_s_c.npy')
X_test = np.load(path+'XTest.npy')   
X_test = np.transpose(X_test[:,0,0,:])
y_test = np.zeros((N_test_set,1,1))
len_ = np.load(path+'len_t.npy')
min_s_c = np.load(path+'min_s_c.npy') 
max_s_c = np.load(path+'max_s_c.npy')
l_true = np.load(path+'l_true.npy')
c_true = np.load(path+'c_true.npy')

# test set   
class testData(Dataset):      
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]        
    def __len__ (self):
        return len(self.X_data)   
X_testz = torch.FloatTensor(X_test).view(-1,1,len_s_c) 
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

# Functions
def mse(x,y):
    return np.sum((x - y)**2)/np.sum(x**2)

def matlab_wavedec(x,wname,n):
  dec  = pywt.wavedec(x,wname,level=n,mode='symmetric')
  c = np.concatenate(dec)
  l = np.zeros([n+2],dtype='int')
  for i in range(n+1):
    l[i] = len(dec[i])
  l[n+1] = len(x)
  return([c,l])

def matlab_waverec(c,l,wname):
  dec = []
  start = 0
  for i in range(len(l)-1):
    dec.append(c[start:start+l[i]])
    start = start+l[i]
  rec  = pywt.waverec(dec,wname,mode='symmetric')
  return(rec)

def landweber_deep_deblurring(z_prec,A,y,x):
    z_prec_ten = torch.cuda.FloatTensor(z_prec)
    x_prec_ten = model.decode(z_prec_ten[:,0])
    x_prec = x_prec_ten.cpu().detach().numpy() 
    x_prec = np.array(x_prec,dtype = np.float32)
    x_prec.squeeze()
    x_prec.resize(len_s_c,1)
    # denormalize x_prec
    x_prec = (max_s_c-min_s_c)*x_prec+min_s_c
    y = np.reshape(y,(len_,1)) 
    # find function from scaling coeff
    c = np.zeros((c_true.shape))
    c[0:l_true[0]] = x_prec[:,0]
    x_prec = matlab_waverec(c,l_true,name)
    x_prec.resize(len_,1)
    arg_2 = np.dot(np.transpose(A),np.dot(A,x_prec)-y)
    # derivative of denormalization arg_2
    arg_2 = (max_s_c-min_s_c)*arg_2
    # find scaling coeff from function
    [c,l] = matlab_wavedec(arg_2[:,0],name,n)
    arg_2 = c[0:l_true[0]]
    arg_2.resize(len_s_c,1)
    der_decoder_ten = torch.autograd.functional.jacobian(model.decode,z_prec_ten[:,0])
    der_decoder = der_decoder_ten.cpu().detach().numpy()
    der_decoder = np.array(der_decoder, dtype = np.float32)
    der_decoder.resize(len_s_c,ZDIMS)
    arg_1 = np.transpose(der_decoder)
    der = np.dot(arg_1,arg_2)    
    return der 
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

    path = "/home/Desktop/VAE_scaling_coeff/"
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

    path = "/home/silvias/Desktop/Deconvolution/" 
    s_d_l = 0.00025  # Landweber step size
    K = 7500  # number of iteration of Landweber

    title = "eps="+str(cost)+"_s_d_l="+str(s_d_l)+"_K="+str(K)+"_db"+str(N)+"_weight_KL="+str(weight_KL)
    os.mkdir(path+title)
    path = path+title+"/"

    # Initialize DL
    z_in = np.random.randn(ZDIMS,1)
    z_in_ten = torch.cuda.FloatTensor(z_in)
    aux = model.decode(z_in_ten[:,0])
    aux = aux.cpu().detach().numpy() 
    aux = np.array(aux,dtype = np.float32)
    aux.squeeze()
    aux.resize(len_s_c,1)
    aux = (max_s_c-min_s_c)*aux+min_s_c
    c = np.zeros((c_true.shape))
    c[0:l_true[0]] = aux[:,0]
    aux = matlab_waverec(c,l_true,name)
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
        x_final_d_l.resize(len_s_c,1)
        x_final_d_l = (max_s_c-min_s_c)*x_final_d_l+min_s_c
        c = np.zeros((c_true.shape))
        c[0:l_true[0]] = x_final_d_l[:,0]
        x_final_d_l = matlab_waverec(c,l_true,name)
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




