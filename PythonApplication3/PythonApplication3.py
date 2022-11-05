import numpy as np
from keras.datasets import nmnist # импорт дата сет
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()
traindata = np.zeros((60000,28,28))
testdata = np.zeros((10000,28,28))
Noises = ["gaussian", "speckle"]
traindata[idx] = add_noise(xtrain[idx],k = noises[noise_id])
testdata[idx] = add_noise(xtest[idx],k = noises[noise_id])
def add_noise(img,noise_type="gaussian"):
    row,col = img.shape
    img = img.astype(np.float32)
    if noise_type == "gaussian":
        mean = -5.9
        var = 35
        sigma = var ** .5
        noise = np.random.normal(mean, sigma, (row,col))
        noise = noise.reshape(row,col)
        img = img + noise
        return img
    if noise_type =="speckle":
        noise = np.random.randn(row,col)
        noise = noise.reshape(row,col)
        img = img + img * noise
    return img
n, m = 2,2
f, axes = plt.subplots(n, m)
axes[0,0].imshow(xtrain[1100], cmap = "gray" )
axes[1,0].imshow(traindata[1100], cmap = "gray")
tsfms = transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize(0.1307, 0.3081)
     ])
trainset = noisedDataset(traindata,xtrain,ytrain,tsfms)
testset = noisedDataset (testdata,xtest,ytest,tsfms)
batch_size:
    trainloader = DataLoader(trainset, batch_size = 32,
                             shuffle = True)
    testloader = DataLoader(testset, batch_size = 1,
                            shuffle = True)

rw, cl = 28, 28
n = 6
name = []
AryIzo = np.zeros((n, rw, cl))
for k in range(n):
    name.append ("Izo" + str(k) + ".jpg")
    izo = Image.open(name[k])
    pix = izo.load()
    for i in range(rw):
        for j in range(cl):
            AryIzo[k,i,j] = (pix[i,j][0]+ pix[i,j][1]+ pix[i,j][2])/3

