import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int):
    transform = torchvision.transforms.Compose([
        ToTensor(),
        Lambda(lambda x: (x - 0.5) * 2)
    ])
    dataset = datasets.MNIST(
        root='./data', 
        train=True,
        transform=transform,
        download=True
    )
    return DataLoader(dataset, batch_size, shuffle=True)


def get_img_shape():
    return (1, 28, 28) # (channels, height, width)