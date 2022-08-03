from datasets.numpydatasets import Numpy2dDataSet, Numpy3dDataSet
from datasets.cyclegan import CycleGANDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import numpy as np

if __name__ == "__main__":  
    ds = CycleGANDataset('data',transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((180,225)),transforms.ToTensor(),transforms.Normalize(0.0085,0.2753)]))
    dl = DataLoader(ds, batch_size=10)
    ex = next(iter(dl))
    img = ex[1].numpy().squeeze() 
    PIL.Image.fromarray((img * 255).astype(np.uint8))
    print(img)