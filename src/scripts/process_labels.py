import os
import cv2
import pickle
import argparse
import imageio
import matplotlib.pyplot as plt

def clean_labels(locs):
    """
    Adjust the values if ant in box coordinaes that fall outside the image dimensions.

    Arguments:
        locs :- loc[img_path] = [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
    
    Returns:
        clean_list :- clean_list[img_path] = [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
    """
    clean_list = {}
    for k, v in locs.items():
        img_path = k
        img = imageio.imread(img_path)
        h, w = img.shape[:2]
        clean_list[img_path] = []
        
        for x in v:
            if x[0] < 0:   x[0] = 0
            if x[1] < 0:   x[1] = 0
            if x[2] >= w:  x[2] = w-1
            if x[3] >= h:  x[3] = h-1
            clean_list[img_path].append(x)
            
    return clean_list
        
def corner_to_center(locs):
    """
    Convert (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height)

    Arguments: 
        locs :- Dictionary with keys = image_paths and the values as a list of all the
                bounding boxes for that image
                locs[img_path] = [list of bounding boxes in the image]
                Box coordinates are (x_min, y_min, x_max, y_max)
    
    Returns:
        center_list :- center_list[img_path] = [list of box coordinates]
                Box coordinates are (center_x, center_y, width, height)
    """
    center_list = {}
    for k, v in locs.items():
        center_list[k] = []
        for x in v:
            center_list[k].append([(x[0]+x[2]) // 2,
                                (x[1]+x[3]) // 2,
                                 x[2] - x[0],
                                 x[3] - x[1]])
    return center_list

def center_to_corner(locs):
    """
    Convert (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)

    Arguments: 
        locs :- Dictionary with keys = image_paths and the values as a list of all the
                bounding boxes for that image
                locs[img_path] = [list of bounding boxes in the image]
                Box coordinates are (center_x, center_y, x_max, y_max)

    Returns:
        corner_list :- corner_list[img_path] = [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
    """
    corner_list = {}
    for k, v in locs:
        corner_list[k] = []
        for x in v:
            corner_list[k].append([x[0] - x[2]//2,
                                x[1] - x[3]//2,
                                x[0] + x[2]//2,
                                x[1] + x[3]//2])
    return corner_list

def center_to_corner_list(locs):
    corner_list = []
    for x in locs:
        corner_list.append([x[0] - x[2]//2,
                            x[1] - x[3]//2,
                            x[0] + x[2]//2,
                            x[1] + x[3]//2])
    return corner_list

def get_corner(loc):
    """
    Convert (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)

    Arguments:
        loc :- A 4 element list containing box coordinates as (center_x, center_y, w, h)
    """
    return [loc[0] - loc[2]//2,
            loc[1] - loc[3]//2,
            loc[0] + loc[2]//2,
            loc[1] + loc[3]//2]

def min_euclid(coord, locs):
    """
    Find the bounding box coordinate closes to the given coord

    Arguments:
        coord :- the reference box coordinate for which we want to find the coordinate
                closes to it
                (center_x, center_y, width, height)
        locs :- List of the boudning box coordinates from which we want to find the closes
                coordinate to coord
                [List of coordinates] of format (center_x, center_y, width, height)
    """
    x = coord[0]
    y = coord[1]
    min_dist = 1e8
    
    for loc in locs:
        dist = (x - loc[0])**2 + (y - loc[1])**2
        if dist < min_dist:
            min_dist = dist
            min_loc = loc
    return min_loc

def IOU(coord1, coord2, area_inter=False): 
    """
    Compute the Intersection Over Union (IOU) between two box cooridnates where the coords
    are of the form (x_min, y_min, x_max, y_max).

    IOU is computed by finding the area of intersection between two bounding boxes and then
    dividing the area of intersection by the union of the area of the two bounding boxes.

    Arguments:
        coord1 :- Firstt bounding box (x_min, y_min, x_max, y_max)
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

def adjust_dims(locs):
    """
    Suppose we split out image into 2 equal parts. Now we obtain the bounding box
    coordinates for the each of the two parts.

    `adjust_dims' changes the coordinates of the boxes of these two images so that
    each box coordinates corresponds to the coordinates of the upsplit image.

    Arguments:
        locs :- contains the coordinates of the boudning boxes of all the images
                including the splits that we created in 'make_split.py'. 
                
                locs[img_name] = [list of box coordinates in that image]
                Box coordinates are (x_min, y_min, x_max, y_max) 
    
    Returns: 
        The box coodinates are adjusted so that each box coordinate in an image patch
        corresponds to the box coordinate as it no splits were made.

        locs[img_name] = [list of adjusted box coordinates].
        Box coordinates are (x_min, y_min, x_max, y_max) 
    """
    names = list(locs.keys())
    tmp = names[0]
    dir_index = len(tmp) - tmp[::-1].find('/')
    names.sort()
    
    for name in names:
        if 'split' in name[dir_index:]:
            # All out split images have `_split` in their name, so we only need to
            # process them
            loc_new = []
            loc = locs[name]
            img_name = name[dir_index:]
            # Every split image is saved with the width and height with which we have
            # to multiple the box coordinates so as to get coords of unsplit image
            h, w = [int(x) for x in img_name.split('_')[1:3]]
            
            for x in loc:
                loc_new.append([x[0]+w, x[1]+h, x[2]+w, x[3]+h])
            locs[name] = loc_new
    return locs


def merge_images(locs):
    """
    Combine all the bounding box coordinates of the splits with the original image. If we
    had originally 2 images for trianing and them we split each of these images into 9 parts,
    then this function concatenates all the bounding boxes from the 9 splits and the original
    image and returns the boudnding boxes for the original 2 images.

    Arguments:
        locs :- coordinates of the boudnding boxes of all images adjusted by using 'adjust_dims'
                
                locs[img_name] = [list of adjusted box coordinates in the images]
                Box coordinates are (x_min, y_min, x_max, y_max)
    
    Returns:
        Concatenation of all the box coordinates of the original image with the splits of that 
        image.

        locs[img_name] = [list of concatenated box coordinates]
        Box coordinates are (x_min, y_min, x_max, y_max)
    """
    names = list(locs.keys())
    tmp = names[0]
    dir_index = len(tmp) - tmp[::-1].find('/')
    dir_name = tmp[:dir_index]
    
    m_locs = {}
    for name in names:
        img_name = name[dir_index:]
        
        if 'split' in img_name:
            ind = img_name.find('split_') + 6
            orig_name = img_name[ind:]
            img = dir_name+orig_name
            
            # We store the complete path to the image as key in the dictionary. As it save the
            # overhead when we load these images, where we can simply read them from the keys
            # of the dictionary
            if img in m_locs:
                m_locs[img] = m_locs[img] + locs[name]
            else:
                m_locs[img] = locs[name]
    
    names = list(m_locs.keys())
    final_locs = {}
    for name in names:
        final_locs[name] = m_locs[name] + locs[name]
        
    return final_locs
    

def merge_labels(locs, thresh=0.3):
    """
    By concatenating all the box coordinates using 'merge_images' we end up with a lot of duplicate
    box coordinates. This function is part1 of a two step cleaning procedure. Here we use remove most
    of the irrelevant bounding boxes by comparing IOU(intersection over union) of two boudning boxes.

    This step removes a lot of unncessary and duplicate labels. The limitation of this function is that
    we merge all the box coordinates in a sequential linear manner, where we can suffer from things
    like bad starts. The limitations of this function are covered up by `final_clean` function.

    This function works by creating a new empty list and adding box coordinates to it such that no pair
    of box coordinates have IOU greater than `thresh`.
    
    Arguments:
        locs :- As returned by `merge_images`. All the box coordinates are concatenated with the orinal
                image from which splits were made in the first place.

                locs[img_name] = [list of concatenated box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
        thresh :- The threshold for the IOU. If the IOU value is greater than thresh than that boudning
                boxes are ignored. (0.0 to 1.0 the legal values)
    
    Returns:
        locs[img_name] = [list of box coordinates]
        Box coordinates are (x_min, y_min, x_max, y_max)
    """
    assert thresh <= 1.0 and thresh >= 0.0
    final_locs = {}
    for k, v in locs.items():
        final_locs[k] = []
        
        for x in v:
            if len(final_locs[k]) == 0:
                final_locs[k].append(x)
            
            min_loc = min_euclid(x, final_locs[k])
            
            x_corner = get_corner(x)
            min_loc_corner = get_corner(min_loc)
            
            iou = IOU(x_corner, min_loc_corner)
            if iou < thresh:
                final_locs[k].append(x)

    return final_locs
    
def merge(locs):
    thresh = 0.3
    final_locs = []
    for x in locs:
        if len(final_locs) == 0:
            final_locs.append(x)
        
        min_loc = min_euclid(x, final_locs)
        
        x_corner = get_corner(x)
        min_loc_corner = get_corner(min_loc)
        
        iou = IOU(x_corner, min_loc_corner)
        if iou < thresh:
            final_locs.append(x)

    return final_locs

def final_clean(locs, thresh=0.5):
    """
    This function is part2 of a two step clearning prcedure. This function removes all the overlapping
    and unncessary boudning boxes. 

    This function is a brute force method where we compare the overlapping area between two boudning boxes
    and if that overlap is greater than 'thresh' than one of the boudning boxes is removed according to a
    predefined strategy. 

    The strategy to remove bounding boxes can be simplified as, removing the smaller bounding box which
    overlaps a bigger bounding box.

    Arguments:
        locs :- locs as returned by `merge_labels`. 
                locs[img_name] = [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
        thresh :- the maximum amount of overlap between two boudning boxes (0.0 to 1.0 legal values)
    
    Returns:
        The final bounding boxes for the training images, where all duplicates and unnecessary bounding 
        boxes have been removed.

        locs[img_name] = [list of box coordinates]
        Box coordinates are (x_min, y_min, x_max, y_max)
    """
    assert thresh >= 0.0 and thresh <= 1.0
    final_locs = {}
    for k, v in locs.items():            
        final_locs[k] = v
        overlap = []
        for i, x in enumerate(v):
            if x in overlap:
                continue
            
            x_corner = get_corner(x)
            
            for j, y in enumerate(v):
                if y == x:
                    continue
                y_corner = get_corner(y)
                area = IOU(x_corner, y_corner, area_inter=True)
                if area != 0.0:
                    if area/(y[2]*y[3]) > thresh:
                        overlap.append(y)
                        final_locs[k].remove(y)
    return final_locs

def final(locs):
    final_locs = locs
    overlap = []
    for i, x in enumerate(locs):
        if x in overlap:
            continue
        x_corner = get_corner(x)

        for j, y in enumerate(locs):
            if y == x:
                continue
            y_corner = get_corner(y)
            area = IOU(x_corner, y_corner, area_inter=True)
            if area != 0.0:
                if area/(y[2]*y[3]) > 0.5:
                    overlap.append(y)
                    final_locs.remove(y)
    return final_locs

def combine_splits(name='parking3_split', thresh1=0.3, thresh2=0.5):
    """
    Process the raw labels that the object detection model gives.

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
        name :- the name of the folder/parking_lot that you using in `Data` folder
        thresh1 :- the threshold for the IOU overlap between two boudning boxes(0.0 to 1.0 legal values)
        thresh2 :- thr threshold for the maximum overlap between two boudnin boxes (0.0 to 1.0 legal values)

    Returns:
        locs :- A dictionary mapping the path to images with the list of bounding boxes in that image
            locs[img_path] = [list of bounding boxes in the image]
            Box coordinates are (x_min, y_min, x_max, y_max)
    """
    assert thresh1 >= 0.0 and thresh1 <= 1.0
    assert thresh2 >= 0.0 and thresh2 <= 1.0

    with open(f'Data/labels/{name}_split.txt', 'rb') as f:
        locs = pickle.load(f)

    locs = adjust_dims(locs)
    locs = merge_images(locs)
    locs = clean_labels(locs)
    locs = corner_to_center(locs)
    locs = merge_labels(locs, thresh=thresh1)
    locs = final_clean(locs, thresh=thresh2)

    final_locs = []
    for k, v in locs.items():
        final_locs = final_locs + v
    final_locs = merge(final_locs)
    final_locs = final(final_locs)

    locs = center_to_corner_list(final_locs)
    return locs


def draw_detection(locs, show, save):
    """
    Function to draw bounding boxes on the original images and option to save them to disk

    Arguments:
        locs :- locs[img_path] = [list of box coordinates]
                Box coordinates are (x_min, y_min, x_max, y_max)
        show :- Bool. If True then the images are shown in a new window
        save :- Bool. If True then the images with the bounding boxes are saved to disk.
                The images are saved in the same directory as the train images i.e.
                `Data/parking3/train/`
    """
    for k, v in locs.items():
        img = imageio.imread(k)
        for x in v:
            cv2.rectangle(img,
                          (x[0], x[1]), (x[2], x[3]),
                          (0,0,255),
                          2)
        
        if show:
            plt.imshow(img)
            plt.pause(2)

        if save:
            name = k.split('.jpg')[0] + '_result.jpg'
            imageio.imwrite(name, img)


def process_labels(directory='parking3_split', save_dir='Data/labels/', thresh1=0.3, thresh2=0.5, show=False, save=False):
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
        directory :- the name of the parking_slot/folder in the 'Data/' folder that you want to use
        save_dir  :- the path to directory where the processed labels should be stored. The labels are stored as 
                     "{save_dir}_processed.txt"
        thresh1   :- threshold for the maximum IOU between two bounding boxes (0.0 to 1.0 legal values)
        thresh2   :- threshold for the maximum overlap between two bounding boxes (0.0 to 1.0 legal values)
        show      :- Bool. If True then the images with processed labels are shown in a new window
        save      :- Bool. If True then the images with processed labels are stored to disk, in the same folder
                     as the train images. The images are stored as "Data/{directory}/train/*_result.jpg"
    """
    # Get locs
    locs = combine_splits(name=directory, thresh1=thresh1, thresh2=thresh2)

    # Save locs
    save_name = f'{directory}_processed.txt'
    file = os.path.join(save_dir, save_name)
    with open(file, 'wb') as f:
        pickle.dump(locs, f)
    
    # Show locs
    # draw_detection(locs, show, save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--directory', type=str, default='parking3_split')
    parser.add_argument('-s', '--save-dir', type=str, default='Data/labels/')
    parser.add_argument('--thresh1', type=float, default=0.3)
    parser.add_argument('--thresh2', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # Get locs
    locs = combine_splits(name=args.directory, thresh1=args.thresh1, thresh2=args.thresh2)

    # Save locs
    save_name = f'{args.directory}_processed.txt'
    file = os.path.join(args.save_dir, save_name)
    with open(file, 'wb') as f:
        pickle.dump(locs, f)
    
    # Show locs
    # draw_detection(locs, args.show, args.save)  