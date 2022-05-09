
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True

ZDIMS_vett = np.arange(start=5,stop=105,step=5,dtype=int)
ZDIMS_vett = np.array(ZDIMS_vett)

# Choose an activation function
act_fun = "myactfun"
# act_fun = "elu"
# act_fun = "relu"
# act_fun = "leaky_relu"

# Weight of the Kullback-Leibler Divergence
weight_KL = 5

path = "/home/Desktop/"

n_im = 200

Ssim_arr = []
Ssim_arr_mean = []
Ssim_arr_std = []

for ZDIMS in ZDIMS_vett:
    h = 0    
    Ssim_arr_z = []
    path_z = "/home/Desktop/"+act_fun+"_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"/"
    Err_mean = 0
    Ssim_mean = 0
    while h < n_im:
        original = cv2.imread(path_z+str(h+1)+"_original.jpg")   
        reconstruction = cv2.imread(path_z+str(h+1)+"_reconstruction.jpg")
        Ssim_arr_z.append(ssim(original,reconstruction,multichannel=True))
        h+=1

    Ssim_arr.append(np.array(Ssim_arr_z))
    Ssim_mean = np.mean(Ssim_arr_z)
    Ssim_std = np.std(Ssim_arr_z)
    Ssim_arr_mean.append(Ssim_mean)
    Ssim_arr_std.append(Ssim_std)

x_vett = np.array(ZDIMS_vett).astype(int)
y_s = np.array(Ssim_arr_mean)
std_s = np.array(Ssim_arr_std)
plt.plot(x_vett,y_s)
plt.fill_between(x_vett,y_s-std_s,y_s+std_s,alpha=.3)
plt.savefig(path+"Graph_ZDIMS_"+act_fun+"_weight_KL="+str(weight_KL)+"_SSIM.pdf")
plt.close()






