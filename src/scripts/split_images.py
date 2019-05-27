import os
import imageio
import numpy as np 
import argparse

def get_imgs(name='parking3_split'):
    """
    Get the images that we want to train on from the train folder. By default, all the images
    that saved using other functions are also stored in the same directory, so we remove them
    from the list of training images.

    The excluded images include
    1.  *_m2det.jpg :- These are the images saved by m2det object detection model with the bounding
        boxes as predicted by m2det.
    2.  *_split_*.jpg :- These are the images that we get after spliting our original image into 
        smaller 3x3 images
    3.  *_result.jpg :- These are the images that are saved after processing the labels. So these
        images contain the final bounding boxes that our classifier model will use.
    
    Arguments:
        name :- The name of the parking_space/folder to use in 'Data/' folder
    
    Returns:
        imgs :- list of images that we require for training our model
    """
    path = f'Data/{name}/train/'
    temp = os.listdir(path)
    imgs = []

    exclude = ['_m2det', '_split', '_result']
    for img in temp:
        if any(x in img for x in exclude):
            continue
        else:
            imgs.append(path + img)
    return imgs


def split_images(directory='parking3_split', save_path='None'):
    """
    Split the image into an overlapping 3x3 grid according to the following strategy. For the width,
    3 splits are done from 0 to 0.5, 0.25 to 0.75, 0.75 to 1.0 percent and same is done for height. 
    Currently, only code with predefined splits has been tested.

    For saving the images the following notation is used {count}_{height}_{width}_split_{image_name}.jpg.
    Here,
    1.  count :- the number of image being split (starts from 1)
    2.  height, width :- In order to save computation at the time of combining these splits the height and
        width which were used to split these images are also stored in the filename. So when combinging thse
        splits we can simply add width and height to the bounding box values and we would get the box
        coordinates for the original unsplit image
    3.  image_name :- The original name of the image

    Arguments:
        directory :- The name of the parking_space/folder to use in 'Data/' folder
        save_path :- the directory where the new split images must be saved, default is the Data/{name}/train
    """
    img_names = get_imgs(directory)
    if save_path is 'None':
        tmp = img_names[0]
        index = len(tmp) - tmp[::-1].find('/')
        save_path = tmp[:index]
    
    splits_w = [[.0, .5], [.25, .75], [.5, 1.]]
    splits_h = [[.0, .5], [.25, .75], [.5, 1.]]    
    
    for num, image in enumerate(img_names):
        im = imageio.imread(image)
        h, w = im.shape[:2]
        
        
        name = image.split('/')[-1]
        for i, s_h in enumerate(splits_h):
            for j, s_w in enumerate(splits_w):
                img = im[int(h*s_h[0]):int(h*s_h[1]), int(w*s_w[0]):int(w*s_w[1]), :]
                new_w = int(s_w[0]*w)
                new_h = int(s_h[0]*h)
                
                save_name = os.path.join(save_path, f'{num}_{new_h}_{new_w}_split_{name}')
                
                imageio.imwrite(save_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split images into a grid of 3x3')
    parser.add_argument('-f', '--directory', type=str, default='parking3_split')
    parser.add_argument('-s', '--save-path', type=str, default='None')
    args = parser.parse_args()

    split_images(args.directory, args.save_path)