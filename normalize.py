import os
import numpy as np
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from datasets.cyclegan import CycleGANDataset

if __name__ == "__main__":
    ds = CycleGANDataset('data',transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=10)
    mean = 0.0
    meansq = 0.0
    count = 0
    from functools import reduce
    for index, data in enumerate(dl):
        #mean= reduce((lambda x, y: x + y), data)
        mean = data.sum()
        #inter = reduce((lambda x, y: x**2 + y), data)
        meansq = meansq + (data**2).sum()
        count += np.prod(data.shape)
    total_mean = mean/count
    total_var = (meansq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print([total_mean,total_var,total_std])