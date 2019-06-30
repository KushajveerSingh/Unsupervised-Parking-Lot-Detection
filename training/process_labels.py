import os
import numpy as np
import cv2
import pickle
import argparse
import imageio
import matplotlib.pyplot as plt

def merge_images(locs, dir):
    """
    Suppose we split out image into 2 equal parts. Now we obtain the bounding box
    coordinates for the each of the two parts.

    `merge_images' changes the coordinates of the boxes of these two images so that
    each box coordinates corresponds to the coordinates of the upsplit image. And in
    the end combine all the images into a single image. 

    Arguments:
        locs :- contains the coordinates of the boudning boxes of all the images
                including the splits that we created in 'make_split.py'. 
                
                locs[img_name] = [list of box coordinates in that image]
                Box coordinates are (x_min, y_min, x_max, y_max) 
    
    Returns: 
        [list of adjusted box coordinates].
        Box coordinates are (x_min, y_min, x_max, y_max) 
    """
    names = list(locs.keys())
    names.sort()
    merged_locs = {}

    # Store the original images
    for name in names:
        if 'split' not in name:
            merged_locs[name] = locs[name]
            temp_img_name = name.split('/')[-1]

    for name in names:
        if 'split' in name:
            # All our split images have `_split` in their name, so we only need to
            # process them
            loc_new = []
            loc = locs[name]
            img_name = name.split('/')[-1]
            # Every split image is saved with the width and height with which we have
            # to multiple the box coordinates so as to get coords of unsplit image
            h, w = [int(x) for x in img_name.split('_')[1:3]]
            
            for x in loc:
                loc_new.append([x[0]+w, x[1]+h, x[2]+w, x[3]+h])
            
            dir_index = len(name) - name[::-1].find('/')
            orig_name = name[:dir_index] + img_name.split('_split_')[-1]
            merged_locs[orig_name] = merged_locs[orig_name] + loc_new

    merged = []
    # Merge images
    for k, v in merged_locs.items():
        merged = merged + v
  
    # Clean the labels for values outside image dimensions
    img = imageio.imread(os.path.join(dir, 'images', temp_img_name))
    h, w = img.shape[:2]
    for x in merged:
        if x[0] < 0:   x[0] = 0
        if x[1] < 0:   x[1] = 0
        if x[2] >= w:  x[2] = w-1
        if x[3] >= h:  x[3] = h-1
    return merged

def IOU(coord1, coord2, area_inter=False): 
    """
    Compute the Intersection Over Union (IOU) between two box cooridnates where the coords
    are of the form (x_min, y_min, x_max, y_max).

    IOU is computed by finding the area of intersection between two bounding boxes and then
    dividing the area of intersection by the union of the area of the two bounding boxes.

    Arguments:
        coord1 :- First bounding box (x_min, y_min, x_max, y_max)
        coord2 :- Second bounding box (x_min, y_min, x_max, y_max)
        area_inter :- Bool. If True then return the area of intersection of two bounding boxes
    
    Returns:
        if area_inter == False:
            IOU between the two bounding boxes as a float value
        if area_inter == True:
            the are of intersection between the two bounding boxes is returned
    """
    x_left = max(coord1[0], coord2[0])
    y_top = max(coord1[1], coord2[1])
    x_right = min(coord1[2], coord2[2])
    y_bottom = min(coord1[3], coord2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return .0
    
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    if area_inter:
        return inter_area

    coord1_area = (coord1[2] - coord1[0]) * (coord1[3] - coord1[1])
    coord2_area = (coord2[2] - coord2[0]) * (coord2[3] - coord2[1])
    iou = inter_area / float(coord1_area + coord2_area - inter_area)

    return iou
    
def clean_labels(locs, thresh=0.5):
    """
    This function removes all the overlapping and unncessary boudning boxes. 

    This function is a brute force method where we compare the overlapping area between two boudning boxes
    and if that overlap is greater than 'thresh' than one of the boudning boxes is removed according to a
    predefined strategy. 

    The strategy to remove bounding boxes can be simplified as, removing the smaller bounding box which
    overlaps a bigger bounding box.

    Arguments:
        locs :- [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
        thresh :- the maximum amount of overlap between two boudning boxes (0.0 to 1.0 legal values)
    
    Returns:
        The final bounding boxes for the training images, where all duplicates and unnecessary bounding 
        boxes have been removed.

        locs[img_name] = [list of box coordinates]
        Box coordinates are (x_min, y_min, x_max, y_max)
    """
    assert thresh >= 0.0 and thresh <= 1.0
    final_locs = locs

    for x in locs:            
        overlap = []
        if x in overlap:
            continue
        
        for y in locs:
            if y == x:
                continue

            y_w = y[2] - y[0]
            y_h = y[3] - y[1]
            area = IOU(x, y, area_inter=True)

            if area != 0.0 and area/(y_w*y_h) > thresh:
                overlap.append(y)
                final_locs.remove(y)

    return final_locs

def NMS(locs, thresh=0.5):
    if len(locs) == 0:
        return []
    
    # Initialize list of the picked boxes
    pick = []

    # Grab the coords of bounding boxes
    x1 = locs[:, 0]
    y1 = locs[:, 1]
    x2 = locs[:, 2]
    y2 = locs[:, 3]

    # Compute area of bounding boxes and sort the boudning boxes
    # by the bottom-right y-coord of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while indices remain in idxs list
    while len(idxs) > 0:
        # Grab last index in the indexed list and add the index 
        # value to the list of picked indexes
        pass

def draw_detection(locs, img_name, dir_name, show, save):
    """
    Function to draw bounding boxes on the original images and option to save them to disk

    Arguments:
        locs :- [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
        show :- Bool. If True then the images are shown in a new window
        save :- Bool. If True then the images with the bounding boxes are saved to temp/label_images
    """
    img = imageio.imread(img_name)
    for x in locs:
        cv2.rectangle(img,
                      (x[0], x[1]), (x[2], x[3]),
                      (0,0,255),
                      2)
        
    if show:
        plt.imshow(img)
        plt.pause(2)

    if save:
        os.makedirs(dir_name+'/labels_images', exist_ok=True)
        name = dir_name + '/labels_images/merged_labels.jpg'
        imageio.imwrite(name, img)


def process_labels(dir, thresh=0.5, show=False, save=False):
    """
    Process the raw labels that the object detection model gives. The raw labels are stored in a pickle format
    as a binary text file. 

    The following steps are taken to get the processed labels:
    1.  Clean the labels. This is done so as to remove the labels that have box coordinates outside the image
        dimensions. (Done by 'utils.clean_labels')
    2.  Merge the labels from the image splits with the original image. If we made 9 splits then the bounding
        boxes from these 9 images/splis are concatenated with the boudning boxs of the original image. (Done 
        by 'adjust_dims' and 'merge_images')
    3.  Now to clean the labels from duplicats and overlaps, a two step procedure is used.
    4.  First, the bounding boxes are sampled by removing the boxes that have Intersection Over Union (IOU) with
        other boxes greater than 'thresh1'. This has some limitations as it is done in a sequential manner. To 
        overcome these limitations second cleaning step is used.
    5.  Second, a brute force method of comparing the area of intersection between paris of boudning boxes is
        used, which removes all the bounding boxes that have overlaps greater than 'thresh2'.

    Arguments:
        dir    :- the name of the parking_slot/folder in the 'Data/' folder that you want to use
        thresh :- threshold for the maximum overlap between two bounding boxes (0.0 to 1.0 legal values)
        show   :- Bool. If True then the images with processed labels are shown in a new window
        save   :- Bool. If True then the images with processed labels are stored to disk, in the same folder
                  as the train images. The images are stored as "Data/{directory}/train/*_result.jpg"
    """
    # Get locs
    with open(f'{dir}/labels/split.txt', 'rb') as f:
        locs = pickle.load(f)

    names = list(locs.keys())
    for name in names:
        if 'split' not in name:
            img_name = name
            break

    # Combine the bounding boxes from the split images with the originl image
    locs = merge_images(locs, dir)

    # Clean Labels to remove duplicates and overlapping boxes
    locs = clean_labels(locs, thresh=thresh)

    locs = clean_labels(locs, thresh=thresh)
    # Save locs
    save_path = f'{dir}/labels/split_processed.txt'
    with open(save_path, 'wb') as f:
        pickle.dump(locs, f)
    
    # Show locs
    draw_detection(locs, img_name, dir, show, save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dir', type=str, default='temp')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    process_labels(dir=args.dir, thresh=args.thresh, show=args.show, save=args.save)