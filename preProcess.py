import cv2
import matplotlib.pyplot as plt
import numpy as np

#for reading csv 
import pandas as pd


DATA_PATH = './data/driving_log.csv'
IMG_PATH = './data/'


#the steering difference between left/right and center camera
STEERING_DIFF = 0.3
#for data augumentation, after that I found keras.preprocessing.image.ImageDataGenerator

def ImgGenerator(img, steering):
    """
    Augumented Img
    """
    img = crop(img, (int(img.shape[0]*0.35), int(img.shape[0]*0.9)))
    img, steering = rndMirror(img, steering)
    img = rndBrightness(img)
    img = resize(img,(64,64))
    return img, steering

def nextBatchImg(batch_size=64):
    """
    Read in image path from .csv file, to generate batch of images.
    1. Randomly pick batch_size of timestamps
    2. Randomly pick one of these images from center, left or right for produce more turning occasions
    however, for the steering is not the same as the center, so I add a difference to it, and
    the value of steering_diff is a hyperparameter.
    Input
        batch_size: batch size
    Output
        batchImg
    """
    imgs = pd.read_csv(DATA_PATH)
    numOfImgs = len(imgs)
    rndIndices = np.random.randint(0, numOfImgs, batch_size)

    image_files_and_angles = []
    # to pick one of the img from center, left or right
    # and modify the steering, if the image is from left or right
    for index in rndIndices:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = imgs.iloc[index]['left'].strip()
            angle = imgs.iloc[index]['steering'] + STEERING_DIFF
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = imgs.iloc[index]['center'].strip()
            angle = imgs.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = imgs.iloc[index]['right'].strip()
            angle = imgs.iloc[index]['steering'] - STEERING_DIFF
            image_files_and_angles.append((img, angle))

    return image_files_and_angles

def imgBatchGenerator(batch_size=64):
    """
    Generator for batch of images
    return batch of image and steering
    """
    while True:
        X_batch = []
        y_batch = []
        imgs = nextBatchImg(batch_size)
        for img_file, angle in imgs:
            rawImage = plt.imread(IMG_PATH + img_file)
            rawAngle = angle
            newImage, newAngle = ImgGenerator(rawImage, rawAngle)
            X_batch.append(newImage)
            y_batch.append(newAngle)

        assert len(X_batch) == batch_size, 'len(X_batch) == batch_size should be True'

        yield np.array(X_batch), np.array(y_batch)



def crop(img, ROI):
    """
    Crop image to Region of interest
    Input
        img: source image
        ROI: the top and bottom of the image
    Output
        image in ROI
    """
    top, bottom = ROI
    return img[int(top):int(bottom), :]


def rndMirror(img, steering, mirrorProb = 0.5):
    """
    Random mirror image and steering
    Input
        img: source image
        steering: corresponding steering
        mirrorProb: the probility to mirror a img
    Output
        MirroredImg, MirrorSteering
    """
    flag = np.random.uniform(0, 1)
    if flag < mirrorProb:
        return np.fliplr(img), -steering
    else:
        return img, steering


def rndRotate(img, steering, maxRotationAngle=20):
    """
    Rotate Img
    Input
        img: source image
        steering: corresponding steering
        maxRotatationAngle: max rotatation angle in degrees
    Output
        RotatedImg, RotatedSteering
    """
    angle = np.random.uniform(-maxRotationAngle, maxRotationAngle+1)
    rad = np.pi / 180 * angle
    return scipy.ndimage.rotate(img, angle, reshape=False), steering-rad


def rndBrightness(img):
    """
    Random modify the brightness by gamma correction
    Input
        img: source image
    Output
        brightnessModifiedImg
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction to modify brightness
    return cv2.LUT(img, table)

def resize(img, newsize):
    """
    Resize Img
    Input
        img: source image
        newsize: tuple of resize size
    Output
        Resize image    
    """
    return cv2.resize(img, newsize)


