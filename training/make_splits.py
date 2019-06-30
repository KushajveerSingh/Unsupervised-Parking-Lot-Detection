import os
import imageio
import numpy as np 
import argparse


def make_splits(path):
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
        path :- The name of working folder
    """
    temp = os.listdir(path)
    img_names = []
    for img in temp:
        img_names.append(path + '/' + img)

    temp = img_names[0]
    index = len(temp) - temp[::-1].find('/')
    save_path = temp[:index]
    
    splits_w = [[.0, .5], [.25, .75], [.5, 1.]]
    splits_h = [[.0, .5], [.25, .75], [.5, 1.]]    
    
    for num, image in enumerate(img_names):
        im = imageio.imread(image)
        h, w = im.shape[:2]
        
        name = image.split('/')[-1]
        for s_h in splits_h:
            for s_w in splits_w:
                img = im[int(h*s_h[0]):int(h*s_h[1]), int(w*s_w[0]):int(w*s_w[1]), :]
                new_w = int(s_w[0]*w)
                new_h = int(s_h[0]*h)
                
                save_name = os.path.join(save_path, f'{num}_{new_h}_{new_w}_split_{name}')
                imageio.imwrite(save_name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split images into a grid of 3x3')
    parser.add_argument('--path', type=str, help='The name of working cirectory')
    args = parser.parse_args()

    make_splits(args.path)