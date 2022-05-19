
import numpy as np
import os
import torch
import torch.utils.data
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from scipy import signal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True
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

x = np.reshape(x,(28*28,1))
N = 28 
#  Deblurring problem
weight = 0.005
weight = 0.1
weight = 0.5
weight = 0.7
e = weight*np.random.randn(28*28,1)

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
y = np.dot(A,x)+e

path = "/home/Desktop/Datas/"
im_y = np.array(y,dtype = np.float32)
im_y.squeeze()
im_y.resize(28,28)
np.save(path+"im_y_e="+str(weight)+".npy",im_y)




