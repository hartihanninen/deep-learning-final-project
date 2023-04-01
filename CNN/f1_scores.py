"""Create a .tsv file containing predictions of an image dataset."""

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
import cnn
from constants import *
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def main(model_name: str):
    """Start the CNN model."""
    if model_name == 'first':
        model = cnn_models.CNN_FIRST()
    elif model_name == 'base':
        model = cnn_models.CNN_BASE()
    elif model_name == 'pretrained':
        model = cnn_models.CNN_PRETRAINED()
    elif model_name == 'comb':
        model = cnn_models.CNN_COMB()
    dict_labels = cnn.create_labels()
    if IMAGENET:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        mean, std = [0.42986962, 0.42986962, 0.42986962], [0.2697042, 0.2697042, 0.2697042]
    test_loader = transform_data(mean, std, model_name, dict_labels)
    model.load_state_dict(torch.load(PATH))
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    create_predictions(model, device, test_loader)


def transform_data(mean: float, std: float, model_name: str, dict_labels): 
    """Transform images to usable matrix representation, change to match your model."""
    if model_name == 'first':
        transform = transforms.Compose([
                                            transforms.Grayscale(num_output_channels=3),
                                            transforms.ToTensor()])
    if model_name == 'base':
        transform = transforms.Compose([
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
    
    elif model_name == 'pretrained' or 'comb':
        transform = transforms.Compose([            #[1]
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),                    #[2]
            transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
            mean=mean,                #[6]
            std=std                 #[7]
            )]
        )
                                
    pred_set = cnn.myDataset(dict_labels=dict_labels, root_dir=TEST_DIR, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=pred_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
    return test_loader


def create_predictions(model: nn.Module, device, test_loader):
    """Create a .tsv file for prections based on a specified model."""
    test_correct = {annotation: [0,0] for annotation in ANNOTATIONS}
    test_correct['tot'] = [0,0]
    test_correct['tot_strict'] = [0,0]
    evaluation = {"true_positive": 0,
                    "false_positive": 0,
                    "true_negative": 0,
                    "false_negative": 0,
                    "negative": 0,
                    "positive": 0}
    model.eval()
    pred_array = []
    target_array = []
    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            pred_round = torch.round(torch.sigmoid(pred))
            if len(pred_array) == 0:
                pred_array = pred_round.numpy()
                target_array = target.numpy()
            else:
                pred_array = np.concatenate((pred_array, pred_round.numpy()))
                target_array = np.concatenate((target_array, target.numpy()))
            # new_test_correct = cnn.calc_correct(pred, target)
            # for annotation in ANNOTATIONS:
            #     new = new_test_correct[annotation]
            #     test_correct[annotation][0] += new[0]
            #     test_correct[annotation][1] += new[1]
            # test_correct['tot'][0] += new_test_correct['tot'][0]
            # test_correct['tot'][1] += new_test_correct['tot'][1]
            # test_correct['tot_strict'][0] += new_test_correct['tot_strict'][0]
            # test_correct['tot_strict'][1] += new_test_correct['tot_strict'][1]

            # evaluations = cnn.class_evaluation(pred, target)
            # evaluation["true_positive"] += evaluations["true_positive"]
            # evaluation["false_positive"] += evaluations["false_positive"]
            # evaluation["true_negative"] += evaluations["true_negative"]
            # evaluation["false_negative"] += evaluations["false_negative"]
            # evaluation["positive"] += evaluations["positive"]
            # evaluation["negative"] += evaluations["negative"]

            # print("------------------------")
            # print('Evaluating: Batch %d/%d: Test Acc: %.3f%% (%d/%d) | Strict Acc: %.3f%% (%d/%d)' % 
            #     (batch_num+1, len(test_loader), 
            #     100. * test_correct['tot'][0] / test_correct['tot'][1], test_correct['tot'][0], test_correct['tot'][1],
            #     100. * test_correct['tot_strict'][0] / test_correct['tot_strict'][1], test_correct['tot_strict'][0], test_correct['tot_strict'][1]))

            # print('True positive rate: %.3f%% (%d/%d)' % 
            #     (100. * evaluation["true_positive"] / evaluation["positive"], evaluation["true_positive"], evaluation["positive"]) )
            # print('False negative rate: %.3f%% (%d/%d)' % 
            #     (100. * evaluation["false_negative"] / evaluation["positive"], evaluation["false_negative"], evaluation["positive"]) )
            # print('True negative rate: %.3f%% (%d/%d)' % 
            #     (100. * evaluation["true_negative"] / evaluation["negative"], evaluation["true_negative"], evaluation["negative"]) )
            # print('False positive rate: %.3f%% (%d/%d)' % 
            #     (100. * evaluation["false_positive"] / evaluation["negative"], evaluation["false_positive"], evaluation["negative"]) )
            # print("------------------------")
    prec_ma, recall_ma, f1_ma, support_ma = precision_recall_fscore_support(target_array, pred_array, average='macro')
    prec_mi, recall_mi, f1_mi, support_mi = precision_recall_fscore_support(target_array, pred_array, average='micro')
    print('Macro:')
    print(prec_ma, recall_ma, f1_ma, support_ma)
    print("---------------------")
    print('Micro:')
    print(prec_mi, recall_mi, f1_mi, support_mi)
    


if __name__ == "__main__":
    main(model_name='comb')
    # y_true = np.array([[0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,1,0,0,0,0,1,0]])
    # y_pred = np.array([[0,0,0,1,0,0,0,0,0,0,0,1,0,0], [0,0,0,1,0,0,0,0,0,0,0,1,1,0]])
    # prec, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # print(prec, recall, f1, support)
    # prec, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    # print(prec, recall, f1, support)
