
"""This module defines project-level constants."""

import os

h = False
if h:
    DATA_DIR = '/Users/hartih/Documents/School/Deep learning/Final_project/dl2021-image-corpus-proj/'
else:
    DATA_DIR = '../../dl2021-image-corpus-proj/'
FILE_NAME = 'cnn_comb.pt'
PATH = DATA_DIR + FILE_NAME
ANNOTATIONS_DIR = DATA_DIR + 'annotations/'
IMAGES_DIR = DATA_DIR + 'images/'
TEST_IMAGES_DIR = DATA_DIR + 'image-test/images/'

# New fodlers for train, test, and dev sets
TRAIN_DIR = DATA_DIR + 'train/'
DEV_DIR = DATA_DIR + 'dev/'
TEST_DIR= DATA_DIR + 'test/'
PRED_DIR = DATA_DIR + 'pred/'
# PRED_DIR = TEST_DIR  # For testing purposes

#--- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.001
WEIGHT_DECAY = 0.00
MOMENTUM = 0.1

#--- fixed constants ---
NUM_CLASSES = 14
NUM_CHANNELS = 3

#--- Other ---
IMAGENET = False
NEW_SPLIT = False
PRETRAINED = False
IMAGE_FILE_NAMES = os.listdir(IMAGES_DIR)
TEST_IMAGE_FILE_NAMES = os.listdir(TEST_IMAGES_DIR)
ANNOTATIONS = ["baby",
                "bird",
                "car",
                "clouds",
                "dog",
                "female",
                "flower",
                "male",
                "night",
                "people",
                "portrait",
                "river",
                "sea",
                "tree"]
