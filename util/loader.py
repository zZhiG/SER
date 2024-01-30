import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
import random
import numpy as np
from mytransforms import PhotoMetricDistortion
from scipy.ndimage import convolve, uniform_filter
import cv2

seed = np.random.randint(1459343089)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_filter(image, size=5, sigma=1.):
    kernel = gaussian_kernel(size, sigma)
    filtered_image = convolve(image, kernel)
    return filtered_image

class Datasetloader(Dataset):
    def __init__(self, root_images, root_masks, h, w):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []

        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            self.images.append(img_file)
            self.labels.append(mask_file)

        print(f'load-images:{len(self.images)} and label:{len(self.labels)}')

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]
        image = Image.open(image).convert('LAB')
        A_img = image.split()[1]
        B_img = image.split()[2]
        image = image.split()[0]
        image = np.array(image)
        L_img = gaussian_filter(image)
        L_img = uniform_filter(L_img, size=3)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(L_img)
        gamma = 1.5
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        clahe_image = cv2.LUT(clahe_image, table)
        clahe_image = Image.fromarray(clahe_image)
        image = Image.merge('LAB', [clahe_image, A_img, B_img]).convert('RGB')
        image = np.array(image)
        image = PhotoMetricDistortion(image)
        image = Image.fromarray(image).convert('RGB')

        mask = Image.open(mask)
        img_tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25)), interpolation=InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(16, fill=144),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])
        lb_tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25)), interpolation=InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(16, fill=0),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        image = image
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        set_seed(1459343089)
        img = img_tf(image)
        img = norm(img)

        set_seed(1459343089)
        mask = lb_tf(mask)
        mask[mask>0] = 1
        mask[mask<0] = 0

        return (img, mask)

