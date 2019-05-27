import os
import pickle
import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt 

def get_img_lbls(name='parking1'):
    """
    Get the reference image used while training and the labels for that parking
    slot. In case multiple images were used during trianing only one image is 
    used.
    
    Arguments:
        name :- The parking slots are names as parking1, parking2 and so on.
                name=1 means we are referring to parking1.
    Return:
        img :- image with dimensions (height, width, channels)
        labels :- a 3D list containing a 2D list for every image
                  used in the training process.
                  The labels for each box are represented as
                  (x_min, y_min, x_max, y_max) which refers to the
                  top left corner and bottom right corner of the img
    """
    path = 'Data'
    path_lbl = path + f'/labels/{name}.txt'
    path_img = path + f'/{name}/train/'
    img = path_img + os.listdir(path_img)[0]
    
    with open(path_lbl, 'rb') as f:
        labels = pickle.load(f)
    img = imageio.imread(img)
    
    return img, labels


def clean_labels(labels, size, threshold=0.5):
    """
    M2Det will give labels that contain values outside the image dimensions
    for the cars at the edges. To deal with these edge cases, we will remove
    some labels and modify the other labels to contain only the pixels in the
    image.
    
    To remove the labels we compare the IOU of the actual box and the box
    with dimensions inside the image, if that ratio is less than threshold 
    than we would remove that label
    
    Arguments:
        labels :- output of get_img_lbls.
            box coordinates are represented as [x_min, y_min, x_max, y_max]
        size :- (height, width) tuple of the reference image
        threshold :-
        
    Returns:
        labels_cleaned :- labels following the rules as described above
    """
    h, w, = size
    labels_cleaned = []
    
    for locs in labels:
        label = []
        for loc in locs:
            x_min, y_min, x_max, y_max = loc
            
            # Check for the left edge
            if x_min < 0:
                temp = 0
                area_in  = (x_max - temp ) * (y_max - y_min)
                area_out = (x_max - x_min) * (y_max - y_min)
                if area_in/area_out < threshold:
                    continue
                else:
                    x_min = 0
                
            # Check for top edge
            if y_min < 0:
                temp = 0
                area_in  = (x_max - x_min) * (y_max - temp )
                area_out = (x_max - x_min) * (y_max - y_min)
                if area_in/area_out < threshold:
                    continue
                else:
                    y_min = 0
                    
            # Check for right edge
            if x_max >= w:
                temp = w-1
                area_in  = (temp  - x_min) * (y_max - y_min)
                area_out = (x_max - x_min) * (y_max - y_min)
                if area_in/area_out < threshold:
                    continue
                else:
                    x_max = w-1
            
            # Check for bottom edge
            if y_max >= h:
                temp = h-1
                area_in  = (x_max - x_min) * (temp  - y_min)
                area_out = (x_max - x_min) * (y_max - y_min)
                if area_in/area_out < threshold:
                    continue
                else:
                    y_max = h-1
            
            label.append([x_min, y_min, x_max, y_max])
        labels_cleaned.append(label)
    
    return labels_cleaned


def corner_to_center_coords(labels, mode=1):
    """
    Convert (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height)
    """
    center_list = []
    if mode == 1:
        for label in labels:
            temp = []
            for x in label:
                temp.append([(x[0]+x[2])//2,
                            (x[1]+x[3])//2,
                            x[2] - x[0],
                            x[3] - x[1]])
            center_list.append(temp)
    else:
        for x in labels:
            center_list.append([(x[0]+x[2])//2,
                         (x[1]+x[3])//2,
                         x[2] - x[0],
                         x[3] - x[1]])
    return center_list


def center_to_corner_coords(labels, mode=1):
    """
    Convert (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
    """
    corner_list = []
    if mode == 1:
        for x in labels:
            corner_list.append([x[0]-x[2]//2,
                                x[1]-x[3]//2,
                                x[0]+x[2]//2,
                                x[1]+x[3]//2])
    elif mode == 2:
        x = labels[0]
        corner_list.append([x[0]-x[2]//2,
                                x[1]-x[3]//2,
                                x[0]+x[2]//2,
                                x[1]+x[3]//2])
    else:
        for label in labels:
            temp = []
            for x in label:
                temp.append([x[0]-x[2]//2,
                             x[1]-x[3]//2,
                             x[0]+x[2]//2,
                             x[1]+x[3]//2])
            corner_list.append(temp)
    return corner_list


def convert_2d_to_1d(labels, w, mode=1):
    """
    Convert 2D (x,y) coordinates to 1D.
    
    Arguments:
        labels :- (center_x, center_y, width, height)
        w :- width of the image used while training
    
    Returns:
        converted_list :- A 1D mapping of the above 2D data along with the 
                width and height of the bounding box
                (center_x+center_y*w, width, height)
    """
    converted_list = []
    if mode == 1:
        for label in labels:
            temp = []
            for x in label:
                temp.append([x[0] + x[1]*w, x[2], x[3]])
            converted_list.append(temp)
    else:
        for x in labels:
            converted_list.append([x[0] + x[1]*w, x[2], x[3]])
    return converted_list


def convert_1d_to_2d(labels, w, mode=1):
    """
    Convert 1D coordinates to 2D (x,y) coords
    
    Arguments:
        labels :- (center_x+center_y*w, width, height)
        w :- width of the image used while training
    
    Returns:
        converted_list :- A 2D mapping of the above 1D data along with the 
                width and height of the bounding box
                (center_x, center_y, width, height)
    """
    converted_list = []
    if mode == 1:
        for x in labels:
            converted_list.append([x[0]%w, x[0]//w, x[1], x[2]])
    elif mode == 2:
        x = labels
        converted_list.append([x[0]%w, x[0]//w, x[1], x[2]])
    else:
        for label in labels:
            temp = []
            for x in label:
                temp.append([x[0]%w, x[0]//w, x[1], x[2]])
            converted_list.append(temp)
    return converted_list


def IOU(coord1, coord2):
    x_left = max(coord1[0], coord2[0])
    y_top = max(coord1[1], coord2[1])
    x_right = min(coord1[2], coord2[2])
    y_bottom = min(coord1[3], coord2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    coord1_area = (coord1[2] - coord1[0]) * (coord1[3] - coord1[1])
    coord2_area = (coord2[2] - coord2[0]) * (coord2[3] - coord2[1])
    
    iou = inter_area / float(coord1_area + coord2_area - inter_area)
    return iou


def merge_labels(labels, w, thresh=None):
    labels_center = corner_to_center_coords(labels)
    labels_1d = convert_2d_to_1d(labels_center, w)
    if thresh is None: 
        thresh = 0.6
    
    labels_merged = labels_1d[0]

    for i, label in enumerate(labels_1d):
        if i == 0: 
            continue
            
        next_merge = []
        for x in label:
            diff = np.inf
            index = 0
            for ind, l in enumerate(labels_merged):
                if abs(x[0] - l[0]) < diff:
                    diff = abs(x[0] - l[0])
                    index = ind
                    
            lm = labels_merged[index]
            lm_center = convert_1d_to_2d(lm, w, mode=2)
            lm_corner = center_to_corner_coords(lm_center, mode=2)[0]
            x_center = convert_1d_to_2d(x, w, mode=2)
            x_corner = center_to_corner_coords(x_center, mode=2)[0]
            
            iou = IOU(x_corner, lm_corner)
            if iou >= thresh:
                continue
            next_merge.append(x)
            
        labels_merged = labels_merged + next_merge
    
            
    labels_2d = convert_1d_to_2d(labels_merged, w)
    labels_corners = center_to_corner_coords(labels_2d)
    
    return labels_corners


def final_clean(labels, w, thresh=None):
    labels_center = corner_to_center_coords(labels, mode=2)
    labels_1d = convert_2d_to_1d(labels_center, w, mode=2)
    if thresh is None: 
        thresh = 0.6
    labels_merged = []

    for i, x in enumerate(labels_1d):
        if i == 0:
            labels_merged.append(x)
            continue

        next_merge = []
        diff = np.inf
        index = 0
        for ind, l in enumerate(labels_merged):
            if abs(x[0] - l[0]) < diff:
                diff = abs(x[0] - l[0])
                index = ind

        lm = labels_merged[index]
        lm_center = convert_1d_to_2d(lm, w, mode=2)
        lm_corner = center_to_corner_coords(lm_center, mode=2)[0]
        x_center = convert_1d_to_2d(x, w, mode=2)
        x_corner = center_to_corner_coords(x_center, mode=2)[0]

        iou = IOU(x_corner, lm_corner)
        if iou >= thresh:
            continue
        next_merge.append(x)
    labels_merged = labels_merged + next_merge

    labels_2d = convert_1d_to_2d(labels_merged, w)
    labels_corners = center_to_corner_coords(labels_2d)
    
    return labels_corners


def save_labels(labels, name='parking1'):
    path = Path('Data/labels/')
    save_name = path/f'{name}_processed.txt'
    with open(save_name, 'wb') as f:
        pickle.dump(labels, f)

def draw_detection(image, labels, name='parking1', save=False, show=False):
    img = np.copy(image)
    
    for label in labels:
        cv2.rectangle(img, 
                     (label[0], label[1]), (label[2], label[3]),
                      (0,0,255),
                      2)

    if save:
        imageio.imwrite(f'Data/{name}/train/merged_m2det.jpg', img)

    if show:
        plt.imshow(img)
        plt.pause(2)


def process_labels(directory='parking1', threshold=0.6, save=False, show=False):
    img, labels = get_img_lbls(directory)
    h, w, _ = img.shape
    labels_cleaned = clean_labels(labels, (h,w))
    merged_labels = merge_labels(labels_cleaned, w, threshold)
    final_labels = final_clean(merged_labels, w, threshold)
    save_labels(final_labels, name=directory)
    draw_detection(img, final_labels, directory, save, show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process labels before classification")
    parser.add_argument('-f', '--directory', type=str, default='parking1', help='name of the parking sapce to use')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Lower limit on the allowed IOU')
    parser.add_argument('--save', action='store_true', help='save final prediction of parking spaces to disk')
    parser.add_argument('--show', action='store_true', help='show final detections of parking spaces')
    args = parser.parse_args()

    img, labels = get_img_lbls(args.directory)
    h, w, _ = img.shape
    labels_cleaned = clean_labels(labels, (h,w))
    merged_labels = merge_labels(labels_cleaned, w, args.threshold)
    # final_labels = final_clean(merged_labels, w, args.threshold)
    save_labels(merged_labels, name=args.directory)
    draw_detection(img, merged_labels, args.directory, args.save, args.show)