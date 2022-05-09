
import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True

BATCH_SIZE = 1

# Vector of latent space dimensions S
ZDIMS_vett = np.arange(start=5,stop=105,step=5,dtype=int)

# Choose an activation function
act_fun = "myactfun"
# act_fun = "elu"
# act_fun = "relu"
# act_fun = "leaky_relu"

# Weight of the Kullback-Leibler Divergence
weight_KL = 5

def myactfun(x):
    return torch.abs(x)*torch.atan(x)

for ZDIMS in ZDIMS_vett:
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            # ENCODER
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, padding=3, stride=2)    
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=4, padding=1, stride=2)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, padding=1, stride=2)
            self.fc1 = nn.Linear(4*4*64, ZDIMS)
            self.fc2 = nn.Linear(4*4*64, ZDIMS)

            self.elu = nn.ELU()
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.leaky_relu = nn.LeakyReLU()
   
            # DECODER
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

        elif act_fun == "elu":
            def encode(self, x: Variable) -> (Variable, Variable):
                x = x.view(-1, 1, 28, 28)
                x = self.elu(self.conv1(x))
                x = self.elu(self.conv2(x))
                x = self.elu(self.conv3(x))
                x = x.view(-1, 4*4*64)
                mu_z = self.fc1(x)       
                logvar_z = self.fc2(x)    
                return mu_z, logvar_z    
            def decode(self, z: Variable) -> Variable:
                x = self.elu(self.fc3(z))
                x = x.view(-1,64,4,4)
                x = self.elu(self.convt1(x))
                x = self.elu(self.convt2(x))
                return self.sigmoid(self.convt3(x))

        elif act_fun == "relu":
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

        elif act_fun == "leaky_relu":
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

    # Load training set images
    SEED=1
    torch.manual_seed(SEED)
    if CUDA:
        torch.cuda.manual_seed(SEED)
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,transform=transforms.ToTensor()),
       batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
       batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    # Number of reconstruction
    n_im = 200
    i = 0  
    while i < n_im:
        M = np.zeros((28,28,3), dtype=float)
        for k in range(28):
            for h in range(28):
                M[k,h,:] = test_loader.dataset[i][0][0][k][h]
        plt.imshow(M, cmap = "gray")
        plt.savefig(path_title+str(i+1)+"_original.jpg")
        plt.close() 
        M_aux = np.array(M[:,:,0], dtype=float) 
        imgray_ten = Variable(torch.cuda.FloatTensor(M_aux).view(1,1,28,28))
        mu,logvar = model.encode(imgray_ten)
        z = model.reparameterize(mu,logvar).cuda()
        ricostr = model.decode(z).cpu().detach().numpy()
        ricostr = np.array(ricostr, dtype = np.float32)
        (ricostr.squeeze()).resize(28,28)
        plt.imshow(ricostr, cmap = "gray")
        plt.savefig(path_title+str(i+1)+"_reconstruction.jpg")
        plt.close()
        i+=1

    # Generator samples
    sample = Variable(torch.randn(16, ZDIMS))
    if CUDA:
        sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(16, 1, 28, 28),"/home/Desktop/Generation_"+act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+".jpg")




