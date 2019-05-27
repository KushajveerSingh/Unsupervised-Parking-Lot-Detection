# Model Demonstration and How to use on your own dataset

This page will guide you on how you can use these models on your own dataset and also will show various demonstrations of the model on various parking spaces.

All your data would go in the `Data` folder. To make things simpler, parking spaces are nameed as `parking1`, `parking2` and so on. Each `parking{x}` refers to a seerate parking space. 

There are 2 subfolders in each `parking{x}` folder, namely `train` and `test`. The `train` folder would contain images that you want to use to get the locations of all the parking spaces. And `test` folder contains the images that are used for testing the resutls.

For demonstration purposes I guide you through `parking1`. In order to test my model on various cases I put images from rainy day, sunny day, normal and images taken in the night.

Follow these 3 steps in order to get the parking detection with all the visualizations for the specified parking slot. To get parking spaces, place images in the `Data/parking1/train/` that occur are used for training. Next place images that you want to test on in the `Data/parking/test/`.

1. Detect parking spaces
    ```python
    python src/m2det/parking_detect.py -f=parking1 --show --save
    ```
    
    Arguments that you can specify:
    | Argument | Useage                                                                                                            | Default    |
    |:--------:|:-----------------------------------------------------------------------------------------------------------------:| ---------- |
    |-f        | pakring slot you want to use                                                                                      | parking1   |
    |-s        | save directory, where you want to save labels                                                                     | Data/labels|
    |--show    | if specified than images with predicted bounded boxes also shown                                                  |            |
    |--save    | if specified the images with bounding boxes are also saved to disk in the same folder as images used for training |            |
    |-t        | the threshold for the bounding box predictions, default works good in most cases                                  | 0.2        |

    Now you will get your labels i.e. the parking spaces as a .txt file in the `Data/labels/parking1.txt`. For the boudning boxes the convention of (x_min, y_min, x_max, y_max) is followed i.e. the top left corner and the bottom right corner of the image.

    If the model some of the spaces, you can either try increasing the number of training images with varying number of cars or you can decrease the threshold and see if that car get's recognized.

2. Get the parking spaces for the specified parking slot. This script combines labels from multiple images      and also some preprocessing is done on the labels. 

    In implementation, the 2D (x,y) coordinates are converted to 1D for easy comparisons. After running this script, you would get a `parking1_processed.txt` in the `Data/labels/parking1_processed.txt` folder.

    ```python
    python src/scripts/process_labels2.py -f=parking1 --show --save
    ```

    Arguments that you can specify:
    | Argument | Useage                                                                                                            | Default    |
    |:--------:|:-----------------------------------------------------------------------------------------------------------------:| ---------- |
    | -f | parking slot you want to use | parking1 |
    | -t | threshold for the IOU that specifies the maximum overalap between two bounding boxes | 0.6 (0.0 - 1.0)|
    | --show | if specified the images with predicted boxes are shown| |
    | --save | if specified the images with bounding boxes are saved to disk in the train/parking1 folde| |
    
3. Classify the patches found in the step 2 using an image classification model.
    ```python
    python src/scripts/classify_patches.py -f=parking1
    ```

    Arguments that you can use:
    | Argument | Useage                                                                                                            | Default    |
    |:--------:|:-----------------------------------------------------------------------------------------------------------------:| ---------- |
    | -f | parking slot you want to use | parking1 |
    | -n | number of images in the test folder to use | -1 (all) |
    | --show | if specified the images with classification predictions are shown | |
    | --save | if specified the images with classification predictions are saved in the test folder with `*_result.jpg` name | |
    | --cuda | if specified use GPU for testing otherwise CPU is used | |
    | -s | the interval in seconds between two plots | 2 |


This project is focused on using a modular approach. As you can see there are three main modules, namely

1. Object Detection Module :- Responsible for detecting the cars. You can use any off-the-shelf object detection model and swap it in here. This provides a large amount of control over what models we want to use
2. Label Processor Module :- Responsible for combining labels from different images and cleaning the labels in cases the values of the predicted boxes are outside the dimensions of the image. This module does not need any editing.
3. Classifying Module :- Responsible for classifying the patches found from the above step as occupied or not. Resnet50 is used (superconvergence is used while training). If you want to change the classifier all you need to do is just make a function that returns your model.
    ```python
    from classifier import load_model

    # load_model
    def load_model():
        model = resnet50()
        model.load_state_dict(torch.load('src/scripts/f_classifier.pth'))
        model.eval()
        return model
    ```