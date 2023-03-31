"""Module for loading in and manipulating different CNN models."""

import os
import shutil
import random
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.nn.functional as F
import math
import cnn_models


class myDataset(torch.utils.data.Dataset):
    """Enable creating train, test, and dev test datasets for PyTorch."""
    def __init__(self, root_dir=IMAGES_DIR, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.images = [root_dir + img for img in os.listdir(root_dir)]                
      
    def __len__(self):
        return len(self.images)       

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = img = Image.open(img_path)     
       
        if self.transform:
            img = self.transform(img)     

        return img, dict_labels[img_path.split("/")[-1]]


def main(h: bool, epochs: int, new_split: bool, imageNet: bool, pretrained: bool, model: nn.Module, file_name: str):
    """Start the CNN model."""
    global_vars(h, epochs, file_name)
    create_labels()
    if new_split:
        split_data()
    mean, std = determine_mean_std(imageNet)
    train_loader, dev_loader, test_loader = transform_data(mean, std)
    if pretrained:
        model.load_state_dict(torch.load(PATH))
    else:
        train_dev_file = DATA_DIR + 'train.txt'
        test_dev_file = DATA_DIR + 'test.txt'
        with open(train_dev_file, 'w') as f:  # Empties the files if not pre-trained
            pass
        with open(test_dev_file, 'w') as f: 
            pass
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_function = nn.BCEWithLogitsLoss()

    dev_loss = math.inf
    dev_losses = []
    dev_accuracies = []
    stop_early = False
    for epoch in range(N_EPOCHS):
        if stop_early:
            break
        train(model, optimizer, loss_function, device, train_loader, epoch, stop_early)
        dev(model, loss_function, device, dev_loader, epoch, dev_loss, dev_losses, dev_accuracies, stop_early)
    test(model, loss_function, device, test_loader)




def global_vars(h: bool, epochs: int, file_name: str):
    """Define global variables."""
    global DATA_DIR 
    if h:
        DATA_DIR = '/Users/hartih/Documents/School/Deep learning/Final_project/dl2021-image-corpus-proj/'
    else:
        DATA_DIR = '../../dl2021-image-corpus-proj/'
    global PATH
    PATH = DATA_DIR + file_name
    global ANNOTATIONS_DIR 
    ANNOTATIONS_DIR = DATA_DIR + 'annotations/'
    global IMAGES_DIR 
    IMAGES_DIR = DATA_DIR + 'images/'

    # New fodlers for train, test, and dev sets
    global TRAIN_DIR 
    TRAIN_DIR = DATA_DIR + 'train/'
    global DEV_DIR 
    DEV_DIR = DATA_DIR + 'dev/'
    global TEST_DIR 
    TEST_DIR= DATA_DIR + 'test/'

    #--- hyperparameters ---
    global N_EPOCHS
    N_EPOCHS = epochs
    global BATCH_SIZE_TRAIN
    BATCH_SIZE_TRAIN = 100
    global BATCH_SIZE_TEST
    BATCH_SIZE_TEST = 100
    global LR
    LR = 0.001
    global WEIGHT_DECAY
    WEIGHT_DECAY = 0.00
    global MOMENTUM
    MOMENTUM = 0.1


    #--- fixed constants ---
    global NUM_CLASSES
    NUM_CLASSES = 14
    global NUM_CHANNELS
    NUM_CHANNELS = 3

    global annotations
    annotations = ["baby",
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


def create_labels():
    """Create labels for images."""
    global image_file_names
    image_file_names = os.listdir(IMAGES_DIR)
    global dict_labels
    dict_labels = {}
    for image_file_name in image_file_names:  # Initiate label tensors
        if os.path.isfile(IMAGES_DIR + image_file_name):
            dict_labels[image_file_name] = torch.zeros(14)
    for i in range(len(annotations)):  # Fill label tensors with 1's if found in one of the annotations text files
        with open(ANNOTATIONS_DIR + annotations[i] + ".txt") as f:
            for row in f:
                row = "im" + row.strip() + ".jpg"
                dict_labels[row][i] = 1


def split_data():
    """Split data between train, dev, and test sets and create directories for them."""
    os.makedirs(TRAIN_DIR)
    os.makedirs(TEST_DIR)
    os.makedirs(DEV_DIR)
    for image_file_name in image_file_names:
        if os.path.isfile(IMAGES_DIR + image_file_name):
            division = random.randint(1, 4)
            if division == 1 or division == 2:
                shutil.copyfile(IMAGES_DIR + image_file_name, TRAIN_DIR + image_file_name)
            if division == 3:
                shutil.copyfile(IMAGES_DIR + image_file_name, DEV_DIR + image_file_name)
            if division == 4:
                shutil.copyfile(IMAGES_DIR + image_file_name, TEST_DIR + image_file_name)


def determine_mean_std(imageNet: bool = False):
    """Choose between imageNet's values and calculating mean and std from our imageset."""
    if imageNet:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # Imagenet mean and std
    else:
        mean_transform = transforms.Compose([
                                                transforms.Grayscale(num_output_channels=3),
                                                transforms.Resize(256),                    
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])

        #Splitting these was the easiest due to memory allocation problem, you could only used IMAGES_DIR if this is not a problem
        train_mean_set = myDataset(TRAIN_DIR, transform=mean_transform)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_mean_set, batch_size=100, shuffle=False, num_workers=0)
        dev_mean_set = myDataset(DEV_DIR, transform=mean_transform)
        dev_data_loader = torch.utils.data.DataLoader(dataset=dev_mean_set, batch_size=100, shuffle=False, num_workers=0)
        test_mean_set = myDataset(TEST_DIR, transform=mean_transform)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_mean_set, batch_size=100, shuffle=False, num_workers=0)

        mean_train, std_train = calc_mean_std(train_data_loader)
        mean_dev, std_dev = calc_mean_std(dev_data_loader)
        mean_test, std_test = calc_mean_std(test_data_loader)
        mean = list((np.array(mean_train)+np.array(mean_dev)+np.array(mean_test))/3)
        std = list((np.array(std_train)+np.array(std_dev)+np.array(std_test))/3)
    return mean, std


def calc_mean_std(loader):
    """Calculate mean and std using our own dataset."""
    mean, std = 0, 0 
    for batch_num, (images, _) in enumerate(loader):
        mean += images.mean([0,2,3])
        std += images.std([0,2,3]) 
    return mean/(batch_num+1), std/(batch_num+1)


def calc_correct(pred: torch.Tensor, target: torch.Tensor):
    """Calculate the amount of correctly predicted as well as total predictions done."""
    pred = torch.sigmoid(pred)  # Since our neural network does not apply sigmoid
    correct_dict = {'tot': [0,0]}  # First number in value is correct ones, the second one is total amount
    correct_dict['tot_strict'] = [0,0] # All correct
    for i in range(len(pred)):
        all_correct = 0
        total = 0
        for j in range(len(pred[i])):
            estim_pred = 0 if float(pred[i][j]) < 0.5 else 1
            if annotations[j] not in correct_dict.keys():
                correct_dict[annotations[j]] = [0,0]
            correct_dict['tot'][1] += 1
            correct_dict[annotations[j]][1] += 1
            correct_dict['tot'][0] += int(estim_pred == target[i][j])
            correct_dict[annotations[j]][0] += int(estim_pred == target[i][j])
            all_correct += int(estim_pred == target[i][j])
            total += 1
        correct_dict['tot_strict'][1] += 1
        correct_dict['tot_strict'][0] += int(total==all_correct)
    return correct_dict


def class_evaluation(pred: torch.Tensor, target: torch.Tensor):
    """Calculate the true/false positive/negative values."""
    pred = torch.sigmoid(pred)
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    negative = 0
    positive = 0
    
    for i in range(len(pred)): # [100,14]
        for j in range(len(pred[i])):
            estim_pred = 0 if float(pred[i][j]) < 0.5 else 1
            # Negative target values
            if target[i][j] == 0:
                negative += 1
                if estim_pred == 1:
                    false_positive += 1
                if estim_pred == 0:
                    true_negative += 1
            # Positive target values
            if target[i][j] == 1:
                positive += 1
                if estim_pred == 1:
                    true_positive += 1
                if estim_pred == 0:
                    false_negative += 1
    
    result = {"true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "negative": negative,
            "positive": positive}
                    
    return result


def class_evaluation_by_annotation(pred: torch.Tensor, target: torch.Tensor):
    """Calculate the true/false positive/negative values for each annotation."""
    pred = torch.sigmoid(pred)
    # Initiate vslues for every annotation
    eval_dict = {}
    for a in range(len(annotations)):
        eval_dict[annotations[a]] = {"true_positive": 0,
                                        "false_positive": 0,
                                        "true_negative": 0,
                                        "false_negative": 0,
                                        "negative": 0,
                                        "positive": 0}
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            estim_pred = 0 if float(pred[i][j]) < 0.5 else 1
            # Negative target values
            if target[i][j] == 0:
                eval_dict[annotations[j]]["negative"] += 1
                if estim_pred == 1:
                    eval_dict[annotations[j]]["false_positive"] += 1
                if estim_pred == 0:
                    eval_dict[annotations[j]]["true_negative"] += 1
            # Positive target values
            if target[i][j] == 1:
                eval_dict[annotations[j]]["positive"] += 1
                if estim_pred == 1:
                    eval_dict[annotations[j]]["true_positive"] += 1
                if estim_pred == 0:
                    eval_dict[annotations[j]]["false_negative"] += 1
                    
    return eval_dict


def transform_data(mean: float, std: float):
    train_transform = transforms.Compose([            
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.Resize(256),                    
                                        transforms.CenterCrop(224),                
                                        transforms.ToTensor(),                     
                                        transforms.Normalize(                      
                                        mean=mean,                
                                        std=std
                                        )])
    test_transform = transforms.Compose([            
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.Resize(256),                    
                                            transforms.CenterCrop(224),                
                                            transforms.ToTensor(),                     
                                            transforms.Normalize(                      
                                            mean=mean,                
                                            std=std
                                            )])
    train_set = myDataset(TRAIN_DIR, transform=train_transform)
    test_set = myDataset(TEST_DIR, transform=train_transform)
    dev_set = myDataset(DEV_DIR, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, shuffle=False)
    return train_loader, dev_loader, test_loader


def train(model: nn.Module, optimizer, loss_function, device, train_loader, epoch: int, stop_early: bool):
    """Train the model"""
    train_dev_file = DATA_DIR + 'train.txt'
    model.train()
    with open(train_dev_file, 'a') as f:
        train_loss = 0
        train_correct = {annotation: [0,0] for annotation in annotations}
        train_correct['tot'] = [0,0]
        train_correct['tot_strict'] = [0,0]
        evaluation = {"true_positive": 0,
                        "false_positive": 0,
                        "true_negative": 0,
                        "false_negative": 0,
                        "negative": 0,
                        "positive": 0}
        total = 0
        model.train()
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_function(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += len(data)
            train_loss += loss.item()
            new_correct = calc_correct(pred, target)
            for annotation in annotations:
                new = new_correct[annotation]
                train_correct[annotation][0] += new[0]
                train_correct[annotation][1] += new[1]
            train_correct['tot'][0] += new_correct['tot'][0]
            train_correct['tot'][1] += new_correct['tot'][1]
            train_correct['tot_strict'][0] += new_correct['tot_strict'][0]
            train_correct['tot_strict'][1] += new_correct['tot_strict'][1]
            
            evaluations = class_evaluation(pred, target)
            evaluation["true_positive"] += evaluations["true_positive"]
            evaluation["false_positive"] += evaluations["false_positive"]
            evaluation["true_negative"] += evaluations["true_negative"]
            evaluation["false_negative"] += evaluations["false_negative"]
            evaluation["positive"] += evaluations["positive"]
            evaluation["negative"] += evaluations["negative"]

            f.write("------------------------")
            f.write('\n')
            f.write('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d) | Strict Acc: %.3f%% (%d/%d)' % 
                (epoch+1, batch_num+1, len(train_loader), train_loss / (batch_num + 1), 
                100. * train_correct['tot'][0] / train_correct['tot'][1], train_correct['tot'][0], train_correct['tot'][1],
                100. * train_correct['tot_strict'][0] / train_correct['tot_strict'][1], train_correct['tot_strict'][0], train_correct['tot_strict'][1]))
            f.write('\n')
            f.write('True positive rate: %.3f%% (%d/%d)' % 
                (100. * evaluation["true_positive"] / evaluation["positive"], evaluation["true_positive"], evaluation["positive"]) )
            f.write('\n')
            f.write('False negative rate: %.3f%% (%d/%d)' % 
                (100. * evaluation["false_negative"] / evaluation["positive"], evaluation["false_negative"], evaluation["positive"]) )
            f.write('\n')
            f.write('True negative rate: %.3f%% (%d/%d)' % 
                (100. * evaluation["true_negative"] / evaluation["negative"], evaluation["true_negative"], evaluation["negative"]) )
            f.write('\n')
            f.write('False positive rate: %.3f%% (%d/%d)' % 
                (100. * evaluation["false_positive"] / evaluation["negative"], evaluation["false_positive"], evaluation["negative"]) )
            f.write('\n')
            f.write("------------------------")
            f.write('\n')


def dev(model: nn.Module, loss_function, device, dev_loader, epoch: int, dev_loss: float, dev_losses: list[float], dev_accuracies: float, stop_early: bool):
    """Use dev set to check how model is performing."""
    train_dev_file = DATA_DIR + 'train.txt'
    model.eval()
    with open(train_dev_file, 'a') as f:
        cur_dev_loss = 0
        dev_correct = {annotation: [0,0] for annotation in annotations}
        dev_correct['tot'] = [0,0]
        dev_correct['tot_strict'] = [0,0]
        evaluation = {"true_positive": 0,
                        "false_positive": 0,
                        "true_negative": 0,
                        "false_negative": 0,
                        "negative": 0,
                        "positive": 0}
        total = 0
        model.eval()
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(dev_loader):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = loss_function(pred, target)

                cur_dev_loss += loss.item()
                new_dev_correct = calc_correct(pred, target)
                for annotation in annotations:
                    new = new_dev_correct[annotation]
                    dev_correct[annotation][0] += new[0]
                    dev_correct[annotation][1] += new[1]
                dev_correct['tot'][0] += new_dev_correct['tot'][0]
                dev_correct['tot'][1] += new_dev_correct['tot'][1]
                dev_correct['tot_strict'][0] += new_dev_correct['tot_strict'][0]
                dev_correct['tot_strict'][1] += new_dev_correct['tot_strict'][1]
                
                evaluations = class_evaluation(pred, target)
                evaluation["true_positive"] += evaluations["true_positive"]
                evaluation["false_positive"] += evaluations["false_positive"]
                evaluation["true_negative"] += evaluations["true_negative"]
                evaluation["false_negative"] += evaluations["false_negative"]
                evaluation["positive"] += evaluations["positive"]
                evaluation["negative"] += evaluations["negative"]

            current_loss = cur_dev_loss / (len(dev_loader) + 1)
            dev_losses.append(current_loss)
            current_accuracy = {annotation: 100. * dev_correct[annotation][0] / dev_correct[annotation][1]  for annotation in annotations}  # Accuracies for all classes 
            current_accuracy['tot'] = 100. * dev_correct['tot'][0] / dev_correct['tot'][1]
            current_accuracy['tot_strict'] = 100. * dev_correct['tot_strict'][0] / dev_correct['tot_strict'][1]
            dev_accuracies.append(current_accuracy)

            f.write("------------------------")
            f.write('\n')
            f.write('Evaluating: Batch %d/%d: Loss: %.4f | Dev Acc: %.3f%% (%d/%d) | Strict Acc: %.3f%% (%d/%d)' % 
            (batch_num+1, len(dev_loader), cur_dev_loss / (len(dev_loader) + 1), 
            100. * dev_correct['tot'][0] / dev_correct['tot'][1], dev_correct['tot'][0], dev_correct['tot'][1],
            100. * dev_correct['tot_strict'][0] / dev_correct['tot_strict'][1], dev_correct['tot_strict'][0], dev_correct['tot_strict'][1]))
            f.write('\n')
            f.write('True positive rate: %.3f%% (%d/%d)' % 
              (100. * evaluation["true_positive"] / evaluation["positive"], evaluation["true_positive"], evaluation["positive"]) )
            f.write('\n')
            f.write('False negative rate: %.3f%% (%d/%d)' % 
              (100. * evaluation["false_negative"] / evaluation["positive"], evaluation["false_negative"], evaluation["positive"]) )
            f.write('\n')
            f.write('True negative rate: %.3f%% (%d/%d)' % 
              (100. * evaluation["true_negative"] / evaluation["negative"], evaluation["true_negative"], evaluation["negative"]) )
            f.write('\n')
            f.write('False positive rate: %.3f%% (%d/%d)' % 
              (100. * evaluation["false_positive"] / evaluation["negative"], evaluation["false_positive"], evaluation["negative"]) )
            f.write('\n')
            f.write("------------------------")
            f.write('\n')


def test(model: nn.Module, loss_function, device, test_loader, dev_losses: list[float], dev_accuracies: list[float]):
    """Calculate the losses and accuracies using test dataset."""
    test_file = DATA_DIR + 'test.txt'
    test_loss = 0
    test_correct = {annotation: [0,0] for annotation in annotations}
    test_correct['tot'] = [0,0]
    test_correct['tot_strict'] = [0,0]
    evaluation = {"true_positive": 0,
                        "false_positive": 0,
                        "true_negative": 0,
                        "false_negative": 0,
                        "negative": 0,
                        "positive": 0}
    for a in range(len(annotations)):
        evaluation_by_annotation[annotations[a]] = {"true_positive": 0,
                                                    "false_positive": 0,
                                                    "true_negative": 0,
                                                    "false_negative": 0,
                                                    "negative": 0,
                                                    "positive": 0}

    evaluation_by_annotation = {}
    model.eval()
    with open(test_file, 'a') as f:
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = loss_function(pred, target)

                test_loss += loss.item()
                new_test_correct = calc_correct(pred, target)
                for annotation in annotations:
                    new = new_test_correct[annotation]
                    test_correct[annotation][0] += new[0]
                    test_correct[annotation][1] += new[1]
                test_correct['tot'][0] += new_test_correct['tot'][0]
                test_correct['tot'][1] += new_test_correct['tot'][1]
                test_correct['tot_strict'][0] += new_test_correct['tot_strict'][0]
                test_correct['tot_strict'][1] += new_test_correct['tot_strict'][1]
                
                evaluations = class_evaluation(pred, target)
                evaluation["true_positive"] += evaluations["true_positive"]
                evaluation["false_positive"] += evaluations["false_positive"]
                evaluation["true_negative"] += evaluations["true_negative"]
                evaluation["false_negative"] += evaluations["false_negative"]
                evaluation["positive"] += evaluations["positive"]
                evaluation["negative"] += evaluations["negative"]
                
                evaluations_by_annotation = class_evaluation_by_annotation(pred, target)
                print(evaluation_by_annotation)
                
                for a in range(len(annotations)):
                    evaluation_by_annotation[annotations[a]]["true_positive"] += evaluations_by_annotation[annotations[a]]["true_positive"]
                    evaluation_by_annotation[annotations[a]]["false_positive"] += evaluations_by_annotation[annotations[a]]["false_positive"]
                    evaluation_by_annotation[annotations[a]]["true_negative"] += evaluations_by_annotation[annotations[a]]["true_negative"]
                    evaluation_by_annotation[annotations[a]]["false_negative"] += evaluations_by_annotation[annotations[a]]["false_negative"]
                    evaluation_by_annotation[annotations[a]]["positive"] += evaluations_by_annotation[annotations[a]]["positive"]
                    evaluation_by_annotation[annotations[a]]["negative"] += evaluations_by_annotation[annotations[a]]["negative"]
                
                f.write("------------------------")
                f.write('\n')
                f.write('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d) | Strict Acc: %.3f%% (%d/%d)' % 
                    (batch_num+1, len(test_loader), test_loss / (batch_num + 1), 
                    100. * test_correct['tot'][0] / test_correct['tot'][1], test_correct['tot'][0], test_correct['tot'][1],
                    100. * test_correct['tot_strict'][0] / test_correct['tot_strict'][1], test_correct['tot_strict'][0], test_correct['tot_strict'][1]))
                f.write('\n')
                f.write('True positive rate: %.3f%% (%d/%d)' % 
                    (100. * evaluation["true_positive"] / evaluation["positive"], evaluation["true_positive"], evaluation["positive"]) )
                f.write('\n')
                f.write('False negative rate: %.3f%% (%d/%d)' % 
                    (100. * evaluation["false_negative"] / evaluation["positive"], evaluation["false_negative"], evaluation["positive"]) )
                f.write('\n')
                f.write('True negative rate: %.3f%% (%d/%d)' % 
                    (100. * evaluation["true_negative"] / evaluation["negative"], evaluation["true_negative"], evaluation["negative"]) )
                f.write('\n')
                f.write('False positive rate: %.3f%% (%d/%d)' % 
                    (100. * evaluation["false_positive"] / evaluation["negative"], evaluation["false_positive"], evaluation["negative"]) )
                f.write('\n')
                f.write("------------------------")
                f.write('\n')
        f.write(dev_losses)
        f.write('\n')
        f.write(dev_accuracies)  

if __name__ == "__main__":
    main(False, 5, False, False, True, cnn_models.CNN_COMB(14, 3), 'cnn_comb.pt')