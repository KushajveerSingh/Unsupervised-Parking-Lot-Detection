import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pickle
from skimage import io, transform

from training.occu_classifier import get_occu_model
from training.color_classifier import get_color_model
from training.pose_classifier import get_pose_model

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

def get_patches(img, dir):
    label = f'{dir}/labels/split_processed.txt'
    with open(label, 'rb') as f:
        labels = pickle.load(f)
    print(img.shape)
    tfms = transforms.Compose([Rescale((150, 150)), Normalize(), ToTensor()])
    imgs = torch.zeros(len(labels), 3, 150, 150, dtype=torch.float)

    for i, label in enumerate(labels):
        temp = img[label[1]:label[3], label[0]:label[2]]
        temp = tfms(temp)
        imgs[i] = temp
    
    return imgs, labels

def softmax(x):
    temp = np.exp(x - np.max(x))
    return temp / temp.sum()

def do_inference(img, dir):
    # img = np.transpose(img, (1, 0, 2))
    input, labels = get_patches(img, dir)

    occu_model = get_occu_model()
    color_model = get_color_model()
    pose_model = get_pose_model()
    
    # Get center locs
    outs = []
    for label in labels:
        outs.append([(label[0]+label[2])//2, (label[1]+label[3])//2])

    # Do occupancy classification
    out = occu_model(input)
    out = out.cpu().data.numpy()
    occu_out = np.argmax(out, axis=1)

    # Do color detection
    out = color_model(input)
    out = out.cpu().data.numpy()
    idxs = np.argmax(out, axis=1)
    color_out = []
    for i, idx in enumerate(idxs):
        temp = softmax(out[i])
        val = temp[idx]
        if val <= 0.136:
            color_out.append(8)
        else:
            color_out.append(idx)

    # Do pose estimation
    out = pose_model(input)
    out = out.cpu().data.numpy()
    pose_out = np.argmax(out, axis=1)
    
    final_outs = []
    for l, o, c, p in zip(outs, occu_out, color_out, pose_out):
        temp = l + [o] + [c] + [p]
        final_outs.append(temp)

    return final_outs

    