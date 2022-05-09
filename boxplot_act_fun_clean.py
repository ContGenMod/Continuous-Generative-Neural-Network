
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA = True

# Vector of latent space dimensions S
ZDIMS_vett = np.arange(start=5,stop=105,step=5,dtype=int)

# Weight of the Kullback-Leibler Divergence
weight_KL = 5

path = "/home/Desktop/"

n_im = 200

for ZDIMS in ZDIMS_vett:
    h = 0
    path_myactfun = "/home/Desktop/myactfun_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"/"
    path_elu = "/home/Desktop/elu_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"/"
    path_relu = "/home/Desktop/relu_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"/"
    path_leaky_relu = "/home/Desktop/leaky_relu_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"/"
    Ssim_myactfun_arr = []
    Ssim_elu_arr = []
    Ssim_relu_arr = []
    Ssim_leaky_relu_arr = []
    while h < n_im:
        original = cv2.imread(path_myactfun+str(h+1)+"_original.jpg")   
        reconstruction_myactfun = cv2.imread(path_myactfun+str(h+1)+"_reconstruction.jpg")
        reconstruction_elu = cv2.imread(path_elu+str(h+1)+"_reconstruction.jpg")  
        reconstruction_relu = cv2.imread(path_relu+str(h+1)+"_reconstruction.jpg")
        reconstruction_leaky_relu = cv2.imread(path_leaky_relu+str(h+1)+"_reconstruction.jpg")
        Ssim_myactfun_arr.append(ssim(original,reconstruction_myactfun,multichannel=True))
        Ssim_elu_arr.append(ssim(original,reconstruction_elu,multichannel=True))
        Ssim_relu_arr.append(ssim(original,reconstruction_relu,multichannel=True))
        Ssim_leaky_relu_arr.append(ssim(original,reconstruction_leaky_relu,multichannel=True))
        h+=1
    Ssim_myactfun = np.array(Ssim_myactfun_arr)
    Ssim_elu = np.array(Ssim_elu_arr)
    Ssim_relu = np.array(Ssim_relu_arr)
    Ssim_leaky_relu = np.array(Ssim_leaky_relu_arr)
    # Boxplot
    fig, axs = plt.subplots(1,4)
    fig.set_tight_layout(1)
    y_min = min(Ssim_myactfun.min(), Ssim_elu.min(), Ssim_leaky_relu.min(), Ssim_relu.min())
    sns.boxplot(data = Ssim_myactfun, ax = axs[0], showfliers=False)
    axs[0].set_ylim([0.97*y_min,1.0])
    sns.boxplot(data = Ssim_elu, ax = axs[1], showfliers=False)
    axs[1].set_ylim([0.97*y_min,1.0])
    sns.boxplot(data = Ssim_leaky_relu, ax = axs[2], showfliers=False)
    axs[2].set_ylim([0.97*y_min,1.0]) 
    sns.boxplot(data = Ssim_relu, ax = axs[3], showfliers=False)
    axs[3].set_ylim([0.97*y_min,1.0])
    plt.savefig(path+"Boxplot_ZDIMS="+str(ZDIMS)+"_weight_KL="+str(weight_KL)+"_SSIM.pdf")
    plt.close()


