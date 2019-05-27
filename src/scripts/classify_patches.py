import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import imageio

import os
import pickle
import argparse
from classifier import load_model

class Rescale(object):
    """
    Resize the image to the given size.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size*h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(img, (new_h, new_w))
        return img


class Normalize(object):
    """
    As we use Imagenet pretrained model we would normalize our images with the 
    mean and std of the Imagenet dataset
    """
    def __init__(self, mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        return (img - self.mean) / self.std


class ToTensor(object):
    """
    Convert numpy arrays to PyTorch tensors
    """
    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)


class ParkingDataset(Dataset):
    """
    Dataset class for making a dataloader for out test images using the processed labels
    """
    def __init__(self, name='parking1', transform=None):
        img_dir = f'Data/{name}/test/'
        label_path = f'Data/labels/{name}_processed.txt'
        
        self.img_dir = img_dir
        temp_imgs = os.listdir(img_dir)
        self.imgs = []
        for x in temp_imgs:
            if '_result' in x:
                continue
            self.imgs.append(x)
        self.imgs.sort()

        with open(label_path, 'rb') as f:
            self.labels = pickle.load(f)
        self.transform = transform
        print(self.labels)
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = io.imread(self.img_dir + img_path)
        
        imgs = torch.zeros(len(self.labels), 3, 150, 150, dtype=torch.float)
        
        for i, label in enumerate(self.labels):
            temp = img[label[1]:label[3], label[0]:label[2]]
            if self.transform is not None:
                temp = self.transform(temp)
            imgs[i] = temp
        
        return imgs, img, self.labels, self.img_dir+img_path


def get_preds(data, cuda=False):
    imgs, img, labels, img_path = data
    if cuda:
        imgs = imgs.to(device)
    out = model(imgs)
    if cuda:
        out = out.cpu()
    out = out.data.numpy()
    out = np.argmax(out, axis=1)
    return out, img, labels, img_path


def draw_results(preds, img, labels, img_name=None, save=False, show=False, speed=2):
    imgcv = np.copy(img)
    for pred, label in zip(preds, labels):
        if pred == 0:
            cv2.rectangle(imgcv,
                          (label[0], label[1]), (label[2], label[3]),
                          (255,0,0),
                          2)
        else:
            cv2.rectangle(imgcv,
                          (label[0], label[1]), (label[2], label[3]),
                          (0,0,255),
                          2)

    if save:
        imageio.imwrite(f'{img_name}_result.jpg', imgcv)

    if show:
        plt.imshow(imgcv)
        plt.pause(speed)

def classify_patches(directory='parking1', num=-1, show=False, save=False, cuda=False, speed=2):
    # Get dataloader
    data = ParkingDataset(directory, transform=transforms.Compose([Rescale((150,150)), 
                                                                   Normalize(),
                                                                   ToTensor()]))
    # Construct model
    model = load_model()
    device = torch.device('cuda')
    if cuda:
        model = model.to(device)
    
    if num == -1:
        count = len(data)
    else:
        count = num

    print('Finished loading the model')

    # Run model for test images
    for i in range(count):
        print('Started test image ', i+1)
        preds, img, labels, img_path = get_preds(data[i], cuda=cuda)
        draw_results(preds, img, labels, img_name=img_path, save=save, show=show, speed=speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify patches of parking slots')
    parser.add_argument('-f', '--directory', type=str, default='parking1', help='name of the parking slot to use')
    parser.add_argument('-n', '--num', type=int, default=-1, help='number of images to use for testing')
    parser.add_argument('-s', '--speed', type=float, default=2, help='Interval for pause if --show is pecified')
    parser.add_argument('--show', action='store_true', help='show predictions')
    parser.add_argument('--save', action='store_true', help='save predictions')
    parser.add_argument('--cuda', action='store_true', help='If specified then use cuda to run the model')
    args = parser.parse_args()

    # Get the dataloader
    data = ParkingDataset(args.directory, transform=transforms.Compose([Rescale((150,150)), 
                                                                        Normalize(),
                                                                        ToTensor()]))
    # Construct model
    model = load_model()
    device = torch.device('cuda')
    if args.cuda:
        model = model.to(device)
    
    if args.num == -1:
        count = len(data)
    else:
        count = args.num

    print('Finished loading the model')

    # Run the model for the test images
    for i in range(count):
        print('Started test image ', i+1)
        preds, img, labels, img_path = get_preds(data[i], cuda=args.cuda)
        draw_results(preds, img, labels, img_name=img_path, save=args.save, show=args.show, speed=args.speed)
