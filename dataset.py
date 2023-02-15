
import numpy as np
import pywt

N = 1  # number of Vanishing moments of the Daubechies wavelet
n = 6  # number of downsampling done to obtain the scaling coefficients
wname = 'db'+str(N)

T = 1  # dimension of the support of the signals 
t = np.linspace(0,T,4096)  # discretized support of the signals

R = 2  # number of non zero Fourier coefficients --> 5 degrees of freedom

# Fouries series
def series_real_coeff(a0, a, b, t, T):
    tmp = np.ones_like(t) * a0 / 2.
    for k, (ak, bk) in enumerate(zip(a, b)):
        tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(
            2 * np.pi * (k + 1) * t / T)
    return tmp

# Generation of a dataset of smooth functions
len_t = len(t)
Ntrainingset = 10000  # number of elements of the training set
XTrain = np.zeros((len_t,1,1,Ntrainingset))
a0 = 0  # mean of each element of the training set
for i in range(Ntrainingset):
    a = np.zeros((R))
    b = np.zeros((R))
    for r in range(R):
        sigma = 1/(r+1)**3  # the variance decreases as the frequency increases
        a[r] = sigma * np.random.randn()
        b[r] = sigma * np.random.randn()
    XTrain[:,0,0,i] = series_real_coeff(a0,a,b,t,T)

Ntestset = 2000  # number of elements of the test set
XTest = np.zeros((len_t,1,1,Ntestset))
a0 = 0  # mean of each element of the test set
for i in range(Ntestset):
    a = np.zeros((R))
    b = np.zeros((R))
    for r in range(R):
        sigma = 1/(r+1)**3  # the variance decreases as the frequency increases
        a[r] = sigma * np.random.randn()
        b[r] = sigma * np.random.randn()
    XTest[:,0,0,i] = series_real_coeff(a0,a,b,t,T)

# Normalize dataset
min_X = min(np.amin(XTrain),np.amin(XTest))
max_X = max(np.amax(XTrain),np.amax(XTest))
XTrain_norm = (XTrain-min_X)/(max_X-min_X)
XTest_norm = (XTest-min_X)/(max_X-min_X)

# matlab functions
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

# Scaling coefficients of dataset
[c,l] = matlab_wavedec(XTrain[:,0,0,1],wname,n)
len_s_c = l[0]
l_true = l
c_true = c
XTrain_s_c = np.zeros((len_s_c,1,1,Ntrainingset))
for i in range(Ntrainingset):
    [c,l] = matlab_wavedec(XTrain[:,0,0,i],wname,n)
    XTrain_s_c[:,0,0,i] = c[0:l[0]]

XTest_s_c = np.zeros((len_s_c,1,1,Ntestset));
for i in range(Ntestset):
    [c,l] = matlab_wavedec(XTest[:,0,0,i],wname,n)
    XTest_s_c[:,0,0,i] = c[0:l[0]]

# Normalize dataset of the scaling coefficients
min_s_c = min(np.amin(XTrain_s_c),np.amin(XTest_s_c))
max_s_c = max(np.amax(XTrain_s_c),np.amax(XTest_s_c))
XTrain_s_c_norm = (XTrain_s_c-min_s_c)/(max_s_c-min_s_c)
XTest_s_c_norm = (XTest_s_c-min_s_c)/(max_s_c-min_s_c)

# Save datasets
path = '/home/Desktop/Dataset/n='+str(n)+'/N='+str(N)+'/'
np.save(path+'XTrain.npy',XTrain) 
np.save(path+'XTest.npy',XTest)
np.save(path+'XTrain_norm.npy',XTrain_norm) 
np.save(path+'XTest_norm.npy',XTest_norm) 
np.save(path+'min_X.npy',min_X) 
np.save(path+'max_X.npy',max_X) 
np.save(path+'min_s_c.npy',min_s_c) 
np.save(path+'max_s_c.npy',max_s_c) 
np.save(path+'t.npy',t) 
np.save(path+'len_t.npy',len_t) 
np.save(path+'l_true.npy',l_true) 
np.save(path+'c_true.npy',c_true) 
np.save(path+'N.npy',N) 
np.save(path+'len_s_c.npy',len_s_c) 
np.save(path+'Ntrainingset.npy',Ntrainingset) 
np.save(path+'Ntestset.npy',Ntestset)
np.save(path+'XTrain_s_c.npy',XTrain_s_c) 
np.save(path+'XTest_s_c.npy',XTest_s_c) 
np.save(path+'XTrain_s_c_norm.npy',XTrain_s_c_norm) 
np.save(path+'XTest_s_c_norm.npy',XTest_s_c_norm) 
np.save(path+'n.npy',n) 



