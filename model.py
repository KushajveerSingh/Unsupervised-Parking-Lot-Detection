import cv2
import numpy as np

import os
import argparse
import shutil
import imageio

from training.make_splits import make_splits
from training.m2det.detect_parking_spaces import detect_parking_spaces
from training.process_labels import process_labels
from training.inference import do_inference

class Model(object):
    def __init__(self, 
                 dir='data/temp', 
                 make_splits=False, 
                 gpu=False,
                 detection_thresh=0.2, 
                 detection_save=False,
                 detection_show=False,
                 label_thresh=0.5,
                 label_show=False,
                 label_save=False):
        self.dir              = dir
        self.gpu              = gpu
        self.make_splits      = make_splits
        self.detection_thresh = detection_thresh
        self.detection_save   = detection_save
        self.detection_show   = detection_show
        self.label_thresh     = label_thresh
        self.label_show       = label_show
        self.label_save       = label_save

        # Some directory constants
        self.images_dir = os.path.join(dir, 'images')
        self.labels_dir = os.path.join(dir, 'labels')

        # Create the required directories
        os.makedirs(f'{self.images_dir}', exist_ok=True)
        os.makedirs(f'{self.labels_dir}', exist_ok=True)


    def predict(self, video_path, x:np.ndarray=None):
        # Convert video to images for ease
        cap = cv2.VideoCapture(video_path)
        
        success, image = cap.read()
        count = 0
        while success:
            cv2.imwrite(f'{self.images_dir}/{count:03d}.jpg', image)
            success, image = cap.read()
            count += 1

        # Split the images if True
        if self.make_splits:
            make_splits(self.images_dir)
        
        # Module1: Object Detection Module
        # Run m2det script to get the object detection labels
        detect_parking_spaces(dir       =self.dir, 
                              threshold =self.detection_thresh, 
                              save      = self.detection_save,
                              show      = self.detection_show,
                              gpu       = self.gpu)
        
        # After running module1 you now have got the unprocessed labels
        # that contain the a mapping from image name to the list of 
        # bounding boxes in that image.

        # Module2: Label Processing Module
        process_labels(dir    = self.dir, 
                       thresh = self.label_thresh, 
                       show   = self.label_show, 
                       save   = self.label_save)
        x = cv2.imread(x)
        outs = do_inference(x, self.dir)
        
        # Remove the temporary image files
        shutil.rmtree(self.images_dir, ignore_errors=True)

        return outs
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python Wrapper for this project')
    parser.add_argument('--video-path', type=str, help='Path to the video file for training the parking space detection')
    parser.add_argument('--dir', type=str, default='temp', help='Path to directory for storing outputs and temporary data')
    parser.add_argument('--make-splits', action='store_true', help='If the option is selected, then the image is split into 3x3 grid')
    parser.add_argument('--gpu', action='store_true', help='If true then GPU is used for testing')
    parser.add_argument('--detection-thresh', type=int, default=0.2, help='The maximum IOU value between two bounding boxes during the object detection phase')
    parser.add_argument('--detection-save', action='store_true', help='If true then the images with bounding boxes are saved to dir/detection_images')
    parser.add_argument('--detection-show', action='store_true', help='If true then the images with bounding boxes are shown in a new window')
    parser.add_argument('--label-thresh', type=int, default=0.5, help='Threshold for the maximum relative overlap betweenm two bounding boxes (0.0 - 1.0)')
    parser.add_argument('--label-save', action='store_true', help='If true then a image with merged bounding boxes is saved to dir/labels_image/')
    parser.add_argument('--label-show', action='store_true', help='If true then a image with merged bounding boxes is shown in a new window')
    args = parser.parse_args()

    model = Model(
        dir              = args.dir,
        make_splits      = args.make_splits,
        gpu              = args.gpu,
        detection_thresh = args.detection_thresh,
        detection_save   = args.detection_save,
        detection_show   = args.detection_show,
        label_thresh     = args.label_thresh,
        label_save       = args.label_save,
        label_show       = args.label_show,
    )

    outs = model.predict(video_path=args.video_path, x='data/images/test.jpg')
    print(outs)