
import numpy as np
import math
import os
import torch
import torch.utils.data
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim
from scipy import signal
from tabulate import tabulate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True   
BATCH_SIZE = 2048
# Latent space dimensions S
ZDIMS = 40
# Weight of the Kullback-Leibler Divergence
weight_KL = 1
# Choose an activation function
# act_fun = "myactfun"
# act_fun = "relu"
act_fun = "leaky_relu"

def myactfun(x):
    return torch.abs(x)*torch.atan(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, padding=3, stride=2)    
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.fc1 = nn.Linear(4*4*64, ZDIMS)
        self.fc2 = nn.Linear(4*4*64, ZDIMS)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        self.fc3 = nn.Linear(ZDIMS, 4*4*64)
        self.convt1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.convt2 = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=4, padding=1, stride=2)
        self.convt3 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=4, padding=3, stride=2)

    if act_fun == "myactfun":
        def encode(self, x: Variable) -> (Variable, Variable):
            x = x.view(-1, 1, 28, 28)
            x = myactfun(self.conv1(x))
            x = myactfun(self.conv2(x))
            x = myactfun(self.conv3(x))
            x = x.view(-1, 4*4*64)
            mu_z = self.fc1(x)       
            logvar_z = self.fc2(x)    
            return mu_z, logvar_z
    
        def decode(self, z: Variable) -> Variable:
            x = myactfun(self.fc3(z))
            x = x.view(-1,64,4,4)
            x = myactfun(self.convt1(x))
            x = myactfun(self.convt2(x))
            return self.sigmoid(self.convt3(x))

    if act_fun == "relu":
        def encode(self, x: Variable) -> (Variable, Variable):
            x = x.view(-1, 1, 28, 28)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = x.view(-1, 4*4*64)
            mu_z = self.fc1(x)       
            logvar_z = self.fc2(x)    
            return mu_z, logvar_z
    
        def decode(self, z: Variable) -> Variable:
            x = self.relu(self.fc3(z))
            x = x.view(-1,64,4,4)
            x = self.relu(self.convt1(x))
            x = self.relu(self.convt2(x))
            return self.sigmoid(self.convt3(x))

    if act_fun == "leaky_relu":
        def encode(self, x: Variable) -> (Variable, Variable):
            x = x.view(-1, 1, 28, 28)
            x = self.leaky_relu(self.conv1(x))
            x = self.leaky_relu(self.conv2(x))
            x = self.leaky_relu(self.conv3(x))
            x = x.view(-1, 4*4*64)
            mu_z = self.fc1(x)       
            logvar_z = self.fc2(x)    
            return mu_z, logvar_z
    
        def decode(self, z: Variable) -> Variable:
            x = self.leaky_relu(self.fc3(z))
            x = x.view(-1,64,4,4)
            x = self.leaky_relu(self.convt1(x))
            x = self.leaky_relu(self.convt2(x))
            return self.sigmoid(self.convt3(x))

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))
            return sample_z
        else:
            return mu

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one.view(-1, 784), x.view(-1, 784))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= BATCH_SIZE * 784
        return BCE+weight_KL*KLD

path = "/home/Desktop/"
title = act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)
path_title = path+title+"/"
device = torch.device("cuda")
model = VAE()
path_s = "state_dict_model_"+act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+".pt"
model.to(device)
model.load_state_dict(torch.load(path_title+path_s, map_location = "cuda:0"))
model.eval()

SEED=1
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,transform=transforms.ToTensor()),
   batch_size=1, shuffle=True, **kwargs)
i = 1   # to choose the image of 0
# i = 4   # to choose the image of 9
M = np.zeros((28,28,3), dtype=float)
for k in range(28):
    for h in range(28):
        M[k,h,:] = loader.dataset[i][0][0][k][h]
x = M[:,:,0]

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

x = np.reshape(x,(28*28,1))  
N = 28
weight = 0.005
weight = 0.1
weight = 0.5
weight = 0.7

fil = [[1.0, 4.0, 7.0, 4.0, 1.0], [4.0, 16.0, 26.0, 16.0, 4.0], [7.0, 26.0, 41.0, 26.0, 7.0], [4.0, 16.0, 26.0, 16.0, 4.0], [1.0, 4.0, 7.0, 4.0, 1.0]]
fil = (1/273)*np.reshape(fil,(5,5))
A = np.zeros((N*N,N*N))
k = 0
for i in range(N):
    for j in range(N):
        x_aux = np.zeros((N,N))
        x_aux[i,j] = 1
        aux = signal.convolve2d(x_aux,fil,mode="same")
        A[k,:] = np.reshape(aux,(1,N*N))
        k = k+1

path = "/home/Desktop/Datas/e="+str(weight)+"/"
y = np.reshape(np.load(path+"im_y_e="+str(weight)+".npy"),(28*28,1))

path = "/home/Desktop/

im_x = np.array(x,dtype = np.float32)
im_x.squeeze()
im_x.resize(28,28)
im_y = np.array(y,dtype = np.float32)
im_y.squeeze()
im_y.resize(28,28)
# Normalization of y values
im_y_norm = (im_y-im_y.min())/(im_y.max()-im_y.min())

im_x_255 = 255*im_x
im_y_255 = 255*im_y_norm
psnr_original = PSNR(im_x_255,im_y_255)
ssim_original = ssim(im_x_255,im_y_255,multichannel="False")

s_d_l = 0.01
s_l = 0.01
d_d_l = 0.0000001
d_l = 0.0000000001

title = "e="+str(weight)+"_s_d_l="+str(s_d_l)+"_d_d_l="+str(d_d_l)+"_s_l="+str(s_l)+"_d_l="+str(d_l)
os.mkdir(path+title)
path = path+title+"/"

# Initialize DL
N = 100
z_in = np.random.randn(ZDIMS)
z_in_ten = torch.cuda.FloatTensor(z_in)
aux = model.decode(torch.cuda.FloatTensor(z_in_ten)).cpu()
aux = aux.cpu().detach().numpy() 
aux = np.array(aux,dtype = np.float32)
aux.squeeze()
aux.resize(28,28)
Sim = ssim(aux,im_y,multichannel="False")

for i in range(N):
    z_aux = np.random.randn(ZDIMS)
    z_aux_ten = torch.cuda.FloatTensor(z_aux)
    aux = model.decode(torch.cuda.FloatTensor(z_aux_ten)).cpu()
    aux = aux.cpu().detach().numpy()  
    aux = np.array(aux,dtype = np.float32)
    aux.squeeze()
    aux.resize(28,28)
    sim = ssim(aux,im_y,multichannel="False")
    if sim > Sim:
        z_in = z_aux
        Sim = sim

z_aux_ten = torch.cuda.FloatTensor(z_in)
aux = model.decode(torch.cuda.FloatTensor(z_aux_ten)).cpu()
aux = aux.cpu().detach().numpy() 
aux = np.reshape(aux,(28*28,1)) 
im_0_d_l = np.array(aux,dtype = np.float32)
im_0_d_l.squeeze()
im_0_d_l.resize(28,28)

# Initialize L
x_prec = np.zeros((28*28,1), dtype=float)
   
#----------------------------------------------------------------------------------------------------------------------

def landweber_deblurring(x_prec,A,y):
    x_prec = np.reshape(x_prec,(28*28,1)) 
    y = np.reshape(y,(28*28,1)) 
    aux = np.dot(A,x_prec)-y   
    return(np.dot(np.transpose(A),aux))

def landweber_deep_deblurring(z_prec,A,y,x):
    z_prec_ten = torch.cuda.FloatTensor(z_prec)
    x_prec_ten = model.decode(z_prec_ten).cpu()
    x_prec = x_prec_ten.cpu().detach().numpy() 
    x_prec = np.reshape(x_prec,(28*28,1)) 
    y = np.reshape(y,(28*28,1)) 
    arg_2 = np.dot(A,x_prec)-y 
    der_decoder_ten = torch.autograd.functional.jacobian(model.decode,z_prec_ten)
    der_decoder = der_decoder_ten.cpu().detach().numpy()
    der_decoder = np.array(der_decoder, dtype = np.float32)
    der_decoder = np.reshape(der_decoder,(28*28,ZDIMS))
    arg_1 = np.dot(np.transpose(der_decoder),np.transpose(A))   
    return np.dot(arg_1,arg_2)   
        
z_prec = z_in
d_z = 2
d_x = 2 

k = 0
while d_z > d_d_l:
    derivative = np.reshape(landweber_deep_deblurring(z_prec,A,y,x),ZDIMS)  
    z_new = z_prec - s_d_l * derivative
    d_z = np.sum((z_new-z_prec)**2)/np.sum((z_prec)**2)   
    z_prec = z_new
    k += 1

k = 0
while d_x > d_l:
    derivative = np.reshape(landweber_deblurring(x_prec,A,y,x),(28*28,1)) 
    x_new = x_prec - s_l * derivative
    d_x = np.sum((x_new-x_prec)**2)/np.sum((x_prec)**2)   
    x_prec = x_new
    k += 1

x_final_d_l = model.decode(torch.cuda.FloatTensor(z_prec)).cpu()
x_final_d_l = np.array(x_final_d_l.cpu().detach().numpy(),dtype = np.float32) 
x_final_d_l.squeeze()
x_final_d_l.resize(28,28)
x_final_d_l_255 = 255*x_final_d_l

psnr_d_l = PSNR(im_x_255,x_final_d_l_255)
ssim_d_l = ssim(im_x_255,x_final_d_l_255,multichannel="False")

x_final_l = np.array(x_prec,dtype = np.float32)
x_final_l = np.resize(x_final_l,(28,28))
# Normalization of x_final_l values
x_final_l_norm = (x_final_l-x_final_l.min())/(x_final_l.max()-x_final_l.min())
x_final_l_255 = 255*x_final_l_norm

psnr_l = PSNR(im_x_255,x_final_l_255)
ssim_l = ssim(im_x_255,x_final_l_255,multichannel="False")

# Table
table = [["", "PSNR", "SSIM"]]
table.append(["Starting point",psnr_original,ssim_original])
table.append(["Landweber",psnr_l,ssim_l])
table.append(["Deep Landweber",psnr_d_l,ssim_d_l])
T = tabulate(table, headers='firstrow', tablefmt='fancy_grid')
f = open(path+'table_PSNR='+str(psnr_original)+'.txt', 'w')
f.write(T)
f.close()

# Figure
fig, (ax2,ax3,ax4) = plt.subplots(1,3)
im2 = ax2.imshow(im_y, cmap="gray")
ax2.axis("off")
im3 = ax3.imshow(x_final_l, cmap="gray")
ax3.axis("off")
im4 = ax4.imshow(x_final_d_l, cmap="gray")
ax4.axis("off")
plt.savefig(path+"Deblurring_different_approaches_PSNR="+str(psnr_original)+".pdf")
plt.close()




