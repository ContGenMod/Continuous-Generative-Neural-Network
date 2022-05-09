
import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True
SEED = 1
LOG_INTERVAL = 10
no_of_sample = 10

EPOCHS = 200

BATCH_SIZE = 2048

# Vector of latent space dimensions S
ZDIMS_vett = np.arange(start=5,stop=105,step=5,dtype=int)

# Choose an activation function
act_fun = "myactfun"
# act_fun = "elu"
# act_fun = "relu"
# act_fun = "leaky_relu"

# Weight of the Kullback-Leibler Divergence
weight_KL = 5

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# ------------------------------------------------------------------------------------------------------------------------
# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,transform=transforms.ToTensor()),
   batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
   batch_size=BATCH_SIZE, shuffle=True, **kwargs)
# ------------------------------------------------------------------------------------------------------------------------

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

    model = VAE()
    if CUDA:
        model.cuda()

    # Learning rate
    lr=0.005
    optimizer = optim.Adam(model.parameters(),lr=lr)

    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = Variable(data)
            if CUDA:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.cpu().data.numpy()
            train_loss /= len(test_loader.dataset)
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.10f}'.format(epoch, train_loss))
        y_a.append(train_loss)
        
    def test(epoch):
        model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(test_loader):
            if CUDA:
                data = data.cuda()
            data = Variable(data)
            recon_batch, mu, logvar = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.10f}'.format(test_loss))
        y_t.append(test_loss)

    y_a = []
    y_t = []
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

    path = "/home/Desktop/"
    title = act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)
    os.mkdir(path+title)
    path_title = path+title+"/"
    path_s = "state_dict_model_"+act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+".pt"
    torch.save(model.state_dict(),path_title+path_s)

    # Plot loss function training
    x = list(range(1,EPOCHS+1))
    x_a = np.array(x)
    y_a_a = np.array(y_a)
    y_t_a = np.array(y_t)
    y_a_KL_a = np.array(y_a_KL)
    y_t_KL_a = np.array(y_t_KL)
    plt.plot(x_a,y_a_a, label= "Train loss")
    plt.plot(x_a,y_t_a, label="Test loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    path_l = "Loss_Adam_lr="+str(lr)+"_"+act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+".jpg"
    plt.savefig(path_title+path_l)
    plt.close()

    
    



