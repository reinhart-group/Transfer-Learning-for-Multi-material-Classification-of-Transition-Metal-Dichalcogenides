#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from torchvision import datasets, models, transforms

from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights

import torch.optim as optim


import random
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split


from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


from codes.utils import stratified_train_test_group_kfold_multilabel
from codes.utils import stratified_train_test_group_kfold
from codes.utils import model_test_classification
from codes.utils import accuracy_classification
from codes.utils import cnn_class_cross_val_final_test, f1score_cnn_fn, confusion_cnn_fn
from codes.utils import accuracy_multilabel, f1score_multilabel, confusion_multilabel
from codes.utils import data_loader_weighted_multilabel, accuracy_1atom_multilabel, label2class_multilabel
from codes.utils import features_and_targets_center, train_multilabel_full, multilabel_train_folds, multilabel_inference_folds
from codes.utils import sample_one_class


from codes.utils import train_multiclass_full, sample_one_multiclass, train_multiclass_full
from codes.utils import multiclass_train_folds, multiclass_inference_folds, data_loader_test_fn
from codes.utils import load_tiff_to_numpy, extract_data_from_image_files
#from codes.classification_codes import cnn_class_gridsearch

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score  
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from collections import Counter
import copy


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


import cv2
import matplotlib.pyplot as plt
from torch import topk
from PIL import Image


from sklearn.decomposition import NMF

from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
##%matplotlib inline

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def grey_to_rgb(numpy_image):
    
    image_rgb = np.repeat(numpy_image[..., np.newaxis], 3, -1)
    image_rgb = image_rgb.transpose(0, 3, 1, 2)/255
    
    return image_rgb


base_path = '/storage/group/wfr5091/default/iam5249/work/AFM_Latent/materials-classification05'

drive_prefix ='Data/Tif_Files/'

afm_files = glob.glob(os.path.join( drive_prefix, '*'))
print(f'found {len(afm_files)} raman files in Drive')

materials_dict = {'MoS2': [1, 0, 1, 0],'WS2': [0, 1, 1, 0], 'WSe2': [0, 1, 0, 1], 'MoSe2': [1, 0, 0, 1], 'Mo-WSe2': [1, 1, 0, 1]}
label_dict = {0: [1, 0, 1, 0], 1:[0, 1, 1, 0], 2:[0, 1, 0, 1], 3:[1, 0, 0, 1], 4:[1, 1, 0, 1]}


X, Y, sampleId, train_val_X, train_val_Y, train_val_groups, test_X, test_Y, test_group = extract_data_from_image_files(afm_files, materials_dict)


drop = 0

pretrain_model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

def model_from_pretrain(pretrain_model, drop):

    pretrain_model.fc = nn.Sequential(nn.Linear(2048, 100), #150, 1
                                     nn.ReLU(),

                                     nn.Dropout(p=drop),
                                     nn.Linear(100, 4),
                                    nn.Sigmoid()
                                     )
    pretrain_model.to(device)


    return pretrain_model



transform = transforms.Compose([
  transforms.ToPILImage(),
  #transforms.RandomRotation(degrees= (0, 180)),
  transforms.RandomHorizontalFlip(0.5),
  transforms.RandomVerticalFlip(0.5),
  #transforms.RandomRotation(90),
  
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  #transforms.Normalize(mean=mean, std=std),
    ])

transform_test = transforms.Compose([
      transforms.ToPILImage(),

      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      #transforms.Normalize(mean=mean, std=std),
        ])






batch_size = 1
def cnn_class_metrics(trained_model, data_loader, label_dict,data_type):

    accuracy = accuracy_multilabel(trained_model, data_loader, label_dict,data_type)
    confusion = confusion_multilabel(trained_model, data_loader, label_dict,data_type)
    f1score= f1score_multilabel(trained_model, data_loader,label_dict, data_type)


    return accuracy, confusion, f1score


print("Doing 10 folds training")

n_train, n_val = n_train_value, n_val_value
drop = 0

criterion = nn.BCELoss()

LR =[7.5e-5]
DROP =[0.0]
Batch_size =[32]

model_path = 'CNNI_Res152_fine5label-train_val_total.pth'
model_dir = os.path.join(base_path, 'Models')
model_dir = os.path.join(model_dir, model_path)
try:
    os.mkdir(model_dir)

except:
    os.path.isdir(model_dir)

    print(f'{model_dir} already exists!')

PATH = os.path.join(base_path, 'Models/model_CNNI_AFM_res152_2-3class-mose.pth') # pretrained model to be further fine-tuned

class_vary = 4

multilabel_train_folds(Batch_size, LR, DROP, criterion, n_train, n_val, train_val_X, train_val_Y, label_dict,train_val_groups, class_vary, base_path, model_path, pretrain_model,model_from_pretrain, PATH=PATH, less_volume = True, sequence=False, epochs=60, n_splits=5)


print(" ................")
print("Doing 10 folds inferencing")




Folds_acc, Folds_f1score, Folds_conf = multilabel_inference_folds(pretrain_model, model_from_pretrain, cnn_class_metrics, base_path, model_path, n_train, n_val, train_val_X, train_val_Y, test_X, test_Y,label_dict,train_val_groups, class_vary, batch_size = 1, less_volume=True, drop = 0.0, n_splits=5)

print(f"train_acc: {np.mean(Folds_acc['train']):.4f}, val_acc:  {np.mean(Folds_acc['val']):.4f}, test_acc: {np.mean(Folds_acc['test']):.4f}")

print(f"train_acc: {np.std(Folds_acc['train']):.4f}, val_acc:  {np.std(Folds_acc['val']):.4f}, test_acc: {np.std(Folds_acc['test']):.4f}")

print(f"train_f1: {np.mean(Folds_f1score['train']):.4f}, val_f1:  {np.mean(Folds_f1score['val']):.4f}, test_f1: {np.mean(Folds_f1score['test']):.4f}")

print(f"train_f1: {np.std(Folds_f1score['train']):.4f}, val_f1:  {np.std(Folds_f1score['val']):.4f}, test_f1: {np.std(Folds_f1score['test']):.4f}")

