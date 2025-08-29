import torch
import torchvision
import random
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset

mnist_mean = 0.1307
mnist_std = 0.3081

to_pil = transforms.ToPILImage()

class DynamicWMNIST(torchvision.datasets.MNIST):
    def __init__(self, root="./data", train=True, min_digits=1, max_digits=3, dataset_size=500000, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.min_digits = min_digits
        self.max_digits = max_digits
        self.dataset_size = dataset_size
        self.transform = transform
        self.values = list(range(min_digits, max_digits+1))
        self.weights = [0.2, 0.3, 0.5]

        self.label_to_indices = {
            i: torch.where(self.targets == i)[0] for i in range(10)
        }

    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        n_digits = random.choices(self.values, weights=self.weights, k=1)[0]
        number = random.randint(0 if n_digits == 1 else 10**(n_digits-1), 10**n_digits - 1)
        digits = list(str(number))

        digit_images = []
        for d in digits:
            label = int(d)
            indices = self.label_to_indices[label]
            chosen_idx = indices[torch.randint(len(indices), (1,)).item()]
            img = self.data[chosen_idx]
            img = img.unsqueeze(0)
            digit_images.append(img)
        concat_img = torch.cat(digit_images, dim=2)

        if self.transform:
            concat_img = self.transform(concat_img)

        return concat_img, number
    
class DynamicDMNIST(torchvision.datasets.MNIST):
    def __init__(self, root="./data", train=True, min_digits=1, max_digits=3, dataset_size=150000, transform=None, download=True):
        super().__init__(root=root, train=train, transform=transform, download=download)

        self.min_digits = min_digits
        self.max_digits = max_digits
        self.dataset_size = dataset_size
        self.transform = transform

        self.label_to_indices = {
            i: torch.where(self.targets == i)[0] for i in range(10)
        }

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        n_digits = random.randint(self.min_digits,self.max_digits)
        number = random.randint(0 if n_digits == 1 else 10**(n_digits-1), 10**n_digits - 1)
        digits = list(str(number))

        digit_images = []
        for d in digits:
            label = int(d)
            indices = self.label_to_indices[label]
            chosen_idx = indices[torch.randint(len(indices), (1,)).item()]
            img = self.data[chosen_idx]
            img = img.unsqueeze(0)
            digit_images.append(img)
        for i in range(n_digits, self.max_digits):
            img = torch.zeros((28, 28))
            img = img.unsqueeze(0)
            digit_images.append(img)
        concat_img = torch.cat(digit_images, dim=0)

        if self.transform:
            concat_img = self.transform(concat_img)

        return concat_img, number

class testMNISTDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # leggo le cartelle e ordino numericamente
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], key=int)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # creo lista (path_file, label)
        for cls_name in classes:
            cls_idx = self.class_to_idx[cls_name]
            folder_path = os.path.join(root, cls_name)
            files = sorted(os.listdir(folder_path), key = lambda x: int(x.removesuffix('.png')))
            for fname in files:
                self.samples.append((os.path.join(folder_path, fname), cls_idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, label
