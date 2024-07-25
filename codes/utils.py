import random
import numpy as np
import os
import pandas as pd
import pickle
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler

from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)



from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='not majority')


import copy
import tifffile

def load_tiff_to_numpy(filename):
    """
    Load a TIFF image (2D or 3D) into a NumPy array.
    
    Parameters:
    filename (str): The filename of the TIFF image to load.
    
    Returns:
    numpy.ndarray: The image data as a NumPy array.
    """
    # Read the TIFF file
    with tifffile.TiffFile(filename) as tif:
        # Get the image data as a NumPy array
        image_data = tif.asarray()
        
        # Get metadata (optional)
        metadata = tif.imagej_metadata
        
    #print(f"Original image shape: {image_data.shape}")
    #print(f"Original image data type: {image_data.dtype}")
    


    
    return image_data

def extract_data_from_image_files(afm_files, materials_dict):
    train_val_X, train_val_Y, train_val_groups = [], [], []
    test_X, test_Y, test_group = [], [], []
    X, Y, sampleId = [], [], []
    for index, item in enumerate(afm_files):
        #print(item)
        filesplit = item.split('/')[-1].split('_')#[0]#.split('_')
        sampleid = filesplit[3]
        #print(sampleid)
        if filesplit[1] in list(materials_dict.keys()):
            target = materials_dict[filesplit[1]]
            dataset = filesplit[-1].split('.')[0]
            numpy_afm = load_tiff_to_numpy(item)
            if dataset=='test':
                test_X.append(numpy_afm)
                test_Y.append(target)
                test_group.append(int(sampleid))

            else:
                train_val_X.append(numpy_afm)
                train_val_Y.append(target)
                train_val_groups.append(int(sampleid))
            X.append(numpy_afm)
            Y.append(target)
            sampleId.append(int(sampleid))

    train_val_X, train_val_Y, train_val_groups = np.array(train_val_X), np.array(train_val_Y), np.array(train_val_groups)       
    test_X, test_Y, test_group = np.array(test_X), np.array(test_Y), np.array(test_group)
    X, Y, sampleId = np.array(X), np.array(Y), np.array(sampleId)
    
    return X, Y, sampleId, train_val_X, train_val_Y, train_val_groups, test_X, test_Y, test_group


def stratified_train_test_group_kfold(X, Y, groups, n_splits, test_fold):
    """this fuction takes X, Y, groups, n_splits, test_fold, val_fold
    and returns the data sets for train, val and test
    X: the features, a numpy arrays
    Y: the targets, a numpy arrays
    groups: group identify of data points, a 1d numpy arrays
    n_splits: the part to which to split the data, 
    test is 1 part, Train is n_splits-1, val is 1/n_splits of Train
    test_fold: which fold is used for test from data, an integer
    """
#Splitting the data to Train and test    
    group_kfold1 = StratifiedGroupKFold(n_splits=n_splits)

    print(type(group_kfold1.split(X, Y, groups)))
    Train_indices = []
    test_indices = []
    for (i, j) in group_kfold1.split(X, Y, groups):
        Train_indices.append(i)
        test_indices.append(j)

    Train_X = X[Train_indices[test_fold]]
    Train_Y = Y[Train_indices[test_fold]]
    Train_groups = groups[Train_indices[test_fold]]

    test_X = X[test_indices[test_fold]]
    test_Y = Y[test_indices[test_fold]]
    
    return Train_groups, Train_X, Train_Y, test_X, test_Y

def label2class_multilabel(label_dict, label_idx, idx=False):
    label_class = None
    if not idx:
        label_idx = np.array([i for i in range(len(label_idx)) if label_idx[i]>0.5])
        #print('label_idx: ', label_idx)
        label_idx =np.sort(label_idx)
        #print('label_idx: ', label_idx)
    elif idx:
        label_idx = np.sort(label_idx)
    for key, value in label_dict.items():
        label = np.array([i for i in range(len(value)) if value[i]>0.5])
        label =np.sort(label)
        #print('label: ', label)
        for idx in range(len(label)):
            #if label[idx] in label_idx:
            if list(label) == list(label_idx):
                label_class = key
                return label_class
    if label_class == None:
        #print('warning! no class identified!')
        label_class = max(list(label_dict.keys()))+1
    return label_class


def stratified_train_test_group_kfold_multilabel(X, Y,label_dict, groups, n_splits, test_fold):
    
    Label_idx = []
    for labels in Y:
        label_idx = np.array([i for i in range(len(labels)) if labels[i]==1])
        Label_idx.append(label_idx)
    #Label_idx = np.array(Label_idx)
    
    Label_class = []
    for label_idx in Label_idx:
        label_class = label2class_multilabel(label_dict, label_idx, idx=True)
        Label_class.append(label_class)
        
    #Label_class = np.array(Label_class)
#Splitting the data to Train and test    
    group_kfold1 = StratifiedGroupKFold(n_splits=n_splits)

    print(type(group_kfold1.split(X, Label_class, groups)))
    Train_indices = []
    test_indices = []
    for (i, j) in group_kfold1.split(X, Label_class, groups):
        Train_indices.append(i)
        test_indices.append(j)

    Train_X = X[Train_indices[test_fold]]
    Train_Y = Y[Train_indices[test_fold]]
    Train_groups = groups[Train_indices[test_fold]]

    test_X = X[test_indices[test_fold]]
    test_Y = Y[test_indices[test_fold]]
    
    return Train_groups, Train_X, Train_Y, test_X, test_Y


def data_loader_weighted_multilabel(x, y, label_dict,transform, batch_size):
    target = torch.tensor(y, dtype=torch.float32)
    data = torch.tensor(x, dtype=torch.float32)
    Label_idx = []
    for labels in y:
        label_idx = np.array([i for i in range(len(labels)) if labels[i]==1])
        Label_idx.append(label_idx)
    #Label_idx = np.array(Label_idx)
    
    Label_class = []
    for label_idx in Label_idx:
        label_class = label2class_multilabel(label_dict, label_idx, idx=True)
        Label_class.append(label_class)
        
    #Label_class = np.array(Label_class)    
    
    labels_unique, class_sample_count = np.unique(Label_class, return_counts=True, axis=0)
    weight = [sum(class_sample_count) / c for c in class_sample_count]
    samples_weight = np.array([weight[t] for t in Label_class])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader




def pred2class(predicted):
    """the function bins the predicted value into the different classes"""
    #predicted = predicted.tolist()
    pred_class = []
    for index, item in enumerate(predicted):
        if item <= 925:# 0.5, 925
            pred_class.append(900)
        elif item <=975:# 1.5, 975
            pred_class.append(950)
        elif item >975:#1.5, 975
            pred_class.append(1000)    
    
    return pred_class
    
def data_loader_fn(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[t] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = torch.from_numpy(target)
    data = torch.from_numpy(data)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader


def data_loader_reg(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    class_dict = {900.0:0, 950.0:1, 1000.0:2}

    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[class_dict[t]] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = np.array(target).reshape(len(target),1)
    target = torch.tensor(target, dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader


def data_loader_test_fn(x, y, transform, batch_size, multilabel=True):
    target, data = 0, 0
    if multilabel:
        
        target = torch.tensor(y, dtype=torch.float32)
        data = torch.tensor(x, dtype=torch.float32)
        
    elif not multilabel:
        data = torch.tensor(x)
        target = torch.tensor(y)
    dataset = Dataset(data, target, transform)

    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = False, drop_last=False)#, num_workers= 2)
   
    return data_loader
    

def data_loader_test_reg(x, y, transform, batch_size):
	y = np.array(y).reshape(len(y),1)
	data = torch.tensor(x, dtype=torch.float32)
	target = torch.tensor(y, dtype=torch.float32)
	dataset = Dataset(data, target, transform)

	data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = False, drop_last=False)#, num_workers= 2)
	return data_loader
	
	
def data_loader_nnrak(x, y, transform, batch_size):
    target = np.array(y)
    data = np.array(x)
    labels_unique, class_sample_count = np.unique(target, return_counts=True)
    weight = [sum(class_sample_count) / c for c in class_sample_count]


    samples_weight = np.array([weight[t] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    target = np.array(target).reshape(len(target),1)
    target = torch.tensor(target, dtype=torch.float32)
    data = torch.tensor(data, dtype=torch.float32)
    #train_dataset = torch.utils.data.TensorDataset(data, target)

    dataset = Dataset(data, target, transform)
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=False)
    
    return data_loader
    
    
def ordinal_loss(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf
    
    predictions: List[List[float]], targets: List[float]"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        target = int(target)
        modified_target[i, 0:target+1] = 1
    loss = nn.MSELoss(reduction='none')(predictions, modified_target).sum(axis=1)
    loss = loss.sum()
    return loss  
    
def nnrank2label(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1] -> 0
    [0.60, 0.51, 0.1] -> 1
    [0.7, 0.7, 0.9] -> 2
    etc.
    pred: np.ndarray
    """
    class_pred = (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1
    return class_pred   
     
    
def accuracy_nnrank(trained_model, data_loader, data_type):
	correct = 0
	total = 0
	trained_model.eval()
	#with torch.no_grad():
	for data in data_loader:
		images, labels = data
		images, labels = images.cuda(), labels.cuda()

		outputs = trained_model(images)
		outputs = nnrank2label(outputs)
		labels =[np.rint(item)[0] for item in labels.cpu().numpy().tolist()]
		outputs = [np.rint(item) for item in outputs.cpu().detach().numpy().tolist()]
		for index, item in enumerate(labels):
			if labels[index]==outputs[index]:
	    			correct += 1
			total += 1
	accuracy = 100 * correct / total

	print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')

	return accuracy
	

	      
    	

transform = transforms.Compose([
      transforms.ToPILImage(),
      #transforms.RandomRotation(degrees= (0, 180)),
      transforms.RandomHorizontalFlip(0.5),
      transforms.RandomVerticalFlip(0.5),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

transform_test = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
      #transforms.Normalize(mean=mean, std=std),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class Dataset():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, transform):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples' 
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = self.transform(ID)
        y = self.labels[index]
        return X, y      




def data_loader_mlp(X, Y, batch_size, shuffle=True):
    data_tensor = torch.tensor(X, dtype=torch.float32)    #all_img test_x, test_y
    target_tensor = torch.tensor(Y)#, dtype = torch.float32)   #target
    
    
    dataset = Dataset_mlp(data_tensor, target_tensor)
    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = shuffle, drop_last=False)#model_micro.eval()
    return data_loader
    
def data_loader_mlp_reg(X, Y, batch_size, shuffle, drop_last):
    Y = Y.reshape(len(Y), 1)
    data_tensor = torch.tensor(X, dtype=torch.float32)    #all_img test_x, test_y
    target_tensor = torch.tensor(Y, dtype = torch.float32)   #target
    
    
    dataset = Dataset_mlp(data_tensor, target_tensor)
    data_loader = DataLoader(dataset,  batch_size = batch_size, shuffle = shuffle, drop_last=drop_last)#model_micro.eval()
    return data_loader
        
    
class Dataset_mlp():
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs


  def __len__(self):
        'Denotes the total number of samples' 
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.list_IDs[index]
        y = self.labels[index]
        return X, y  
   

def grey_to_rgb(numpy_image):
    
    image_rgb = np.repeat(numpy_image[..., np.newaxis], 3, -1)
    image_rgb = image_rgb.transpose(0, 3, 1, 2)/255
    
    return image_rgb


def features_and_targets_center(df_center, df_materials, materials_dict, less_class, less_class_size):
    filecenter = list(df_center['filename'])
    fileall = list(df_materials['filename'])
    materials = list(df_materials['materials'])
    data_image1 = np.array([np.array(item) for item in df_materials['AFM']])
    sampleId1 = np.array(df_materials['sampleId'])
    
    data_index = []
    for index, item in enumerate(materials):
        if len(item)==1 and item[0] in materials_dict.keys():
            data_index.append(index)
    center_index = []
    targets = []
    less_class_count = {}
    for index, item in enumerate(fileall):
        if index in data_index and materials[index][0] in less_class:
            less_class_count[materials[index][0]]=less_class_count.get(materials[index][0], 0) + 1
            if less_class_count[materials[index][0]] < less_class_size:
                targets.append(materials_dict[materials[index][0]])
                center_index.append(index)
        elif index in data_index and item in filecenter:
            targets.append(materials_dict[materials[index][0]])
            center_index.append(index)


    targets = np.array(targets)
    data_image = data_image1[center_index]
    sampleId = sampleId1[center_index]

    features = grey_to_rgb(data_image)

    print(np.unique(targets, return_counts=True, axis=0))
    
    print('features: ', features.shape, 'targets: ', targets.shape, 'groups: ', sampleId.shape)
    
    
    return features, targets, sampleId
    

def sample_one_class(X, Y, label_dict, number_sample, target_class):
    random.seed(41)
    sample_index = []
    sample_index_less = []
    for index, item in enumerate(Y):
        if label2class_multilabel(label_dict, item) == target_class:
            
            sample_index_less.append(index)
        elif label2class_multilabel(label_dict, item) != target_class:
            sample_index.append(index)
    sample_index_less = random.sample(sample_index_less, number_sample)
    sample_index += sample_index_less
    x = X[sample_index]
    y = Y[sample_index]
    #print(sample_dict)
    return x, y


def sample_one_multiclass(X, Y, number_sample, target_class):
    random.seed(41)
    sample_index = []
    sample_index_less = []
    for index, item in enumerate(Y):
        if item == target_class: 
            sample_index_less.append(index)
            
        elif item != target_class:
            sample_index.append(index)
    sample_index_less = random.sample(sample_index_less, number_sample)
    sample_index += sample_index_less
    x = X[sample_index]
    y = Y[sample_index]
    return x, y

    
def train_multiclass_full(model, optimizer, criterion, epochs, model_path, train_loader, val_loader):

    running_loss_list = []
    val_running_loss_list = []
    mean_abs_error_list = []
    val_mean_abs_error_list = []

    maxval_f1score = 0.0
    minval_loss = 10000000
    model_return = None
    best_weights = None
    performance_record = {'loss': [], 'val_loss': []}
    for epoch in range(1, epochs): 

        running_loss = []
        val_running_loss = []

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward  + optimize

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        #model_micro.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                #print('outputs: ', outputs)
                #print('labels: ', labels)
                val_loss = criterion(outputs, labels)
                val_running_loss.append(val_loss.item())

        train_f1score= f1score_cnn_fn(model, train_loader, 'train')

        val_f1score= f1score_cnn_fn(model, val_loader, 'val')
  
        loss_mean = np.mean(running_loss)
        val_loss_mean = np.mean(val_running_loss)

        running_loss_list.append(float(f'{loss_mean :.4f}'))
        val_running_loss_list.append(float(f'{val_loss_mean:.4f}'))

        print(f'Epoch{epoch}: loss: {loss_mean:.4f} val_loss: {val_loss_mean:.4f}')

        if val_f1score > maxval_f1score and train_f1score > maxval_f1score and loss_mean < minval_loss and val_loss_mean < minval_loss:
            maxval_f1score = val_f1score
            minval_loss = val_loss_mean
            
            print('val_f1score: ', val_f1score)

            performance_record['loss'].append(running_loss_list[:])

            performance_record['val_loss'].append(val_running_loss_list[:])

            best_weights = copy.deepcopy(model.state_dict())


    return maxval_f1score, performance_record, best_weights



def multiclass_train_mlp_folds(Batch_size, LR, DROP, L1, L2, criterion, train_val_X, train_val_Y ,train_val_groups, base_path, model_path, mlp_model,oversample, epochs=60, n_splits=10):
    for fold in range(n_splits):

        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)
        
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
            

        model_fold = f'Fold{fold}.pth'
        result_fold = f'Fold{fold}_tuning'
        record_fold = f'Fold{fold}_record'

        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)

        result_path1 = os.path.join(PATH0, result_fold)
        model_path1 = os.path.join(PATH0, model_fold)
        record_path1 = os.path.join(PATH0, record_fold)

        RESULTS = []

        model_return = None
        performance_record = {}
        maxval_f1score = 0.0

        for batch_size in Batch_size:

            for lr in LR:
                for drop in DROP:
                    for l1 in L1:
                        for l2 in L2:

                            train_loader = data_loader_mlp(train_X, train_Y, batch_size=batch_size, shuffle=True)
                            val_loader = data_loader_mlp(val_X, val_Y, batch_size=batch_size, shuffle=False)

                            results = {}
                            model = mlp_model(l1, l2, drop).to(device)                     

                            optimizer = optim.Adam(model.parameters(), lr=lr)

                            this_f1score, this_record, best_weights = train_multiclass_full(model, optimizer, criterion, epochs, model_path, train_loader, val_loader)

                            results['lr']=lr
                            results['drop']=drop
                            results['f1score'] = this_f1score
                            results['batch_size'] = batch_size
                            results['l1'] = l1
                            results['l2'] = l2

                            RESULTS.append(results)

                            with open(result_path1, "wb") as fp:   #Pickling
                              pickle.dump(RESULTS, fp) 

                            if this_f1score > maxval_f1score:
                                maxval_f1score = this_f1score

                                performance_record = this_record

                                with open(record_path1, "wb") as fp:   #Pickling
                                  pickle.dump(performance_record, fp)			

                                model.load_state_dict(best_weights)

                                torch.save(model.state_dict(), model_path1) 

                                print('best f1score obtained: ', maxval_f1score)  

        print('best f1score obtained: ', maxval_f1score)
        print(f'fold_{fold} done')



def multiclass_inference_mlp_folds(mlp_model, cnn_class_metrics, base_path, model_path, train_val_X, train_val_Y, test_X, test_Y,train_val_groups, batch_size = 16, drop = 0.0, n_splits=10):
    Folds_acc = {'train':[], 'val':[], 'test':[]}
    Folds_f1score = {'train':[], 'val':[], 'test':[]}
    Folds_conf = {'train':[], 'val':[], 'test':[]}
    for fold in range(n_splits):
 
        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)

        train_loader = data_loader_mlp(train_X, train_Y, batch_size=batch_size, shuffle=True)
        val_loader = data_loader_mlp(val_X, val_Y, batch_size=batch_size, shuffle=False)

        test_loader = data_loader_mlp(test_X, test_Y, batch_size=batch_size, shuffle=False)

        model = mlp_model #(drop=drop)

            
        model_fold = f'Fold{fold}.pth'
        
        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)
    
        model_path1 = os.path.join(PATH0, model_fold)
                
        model.load_state_dict(torch.load(model_path1))
        model.eval()

        train_acc, train_conf, train_f1score = cnn_class_metrics(model, train_loader, 'train')
        val_acc, val_conf, val_f1score = cnn_class_metrics(model, val_loader, 'val')
        test_acc, test_conf, test_f1score = cnn_class_metrics(model, test_loader,'test')

        Folds_acc['train'].append(train_acc)
        Folds_acc['val'].append(val_acc)
        Folds_acc['test'].append(test_acc)
        Folds_f1score['train'].append(train_f1score)
        Folds_f1score['val'].append(val_f1score)
        Folds_f1score['test'].append(test_f1score)
        
        Folds_conf['train'].append(train_conf)
        Folds_conf['val'].append(val_conf)
        Folds_conf['test'].append(test_conf)        
        
    return Folds_acc, Folds_f1score, Folds_conf
        
        

def multiclass_train_folds(Batch_size, LR, DROP, criterion, n_train, n_val, train_val_X, train_val_Y ,train_val_groups, class_vary, base_path, model_path, pretrain_model,model_from_pretrain, PATH=None, less_volume = True, sequence=True, epochs=60, n_splits=10):
    for fold in range(n_splits):
        train_X, train_Y, val_X, val_Y = 0, 0, 0, 0
        if less_volume:
            group, Train_X, Train_Y, Val_X, Val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)

            train_X, train_Y = sample_one_multiclass(Train_X, Train_Y, n_train, class_vary)
            val_X, val_Y = sample_one_multiclass(Val_X, Val_Y, n_val, class_vary)
        elif not less_volume:
            group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)
            

        model_fold = f'Fold{fold}.pth'
        result_fold = f'Fold{fold}_tuning'
        record_fold = f'Fold{fold}_record'

        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)

        result_path1 = os.path.join(PATH0, result_fold)
        model_path1 = os.path.join(PATH0, model_fold)
        record_path1 = os.path.join(PATH0, record_fold)

        RESULTS = []

        model_return = None
        performance_record = {}
        maxval_f1score = 0.5

        for batch_size in Batch_size:

            for lr in LR:
                for drop in DROP:

                    train_loader = data_loader_fn(train_X, train_Y,transform, batch_size=batch_size)
                    val_loader = data_loader_test_fn(val_X, val_Y, transform_test, batch_size=batch_size, multilabel=False)

                    results = {}
                    model = model_from_pretrain(pretrain_model, drop)
                    if sequence:
                        model.load_state_dict(torch.load(PATH))
                        model.fc[3] = nn.Linear(100, 1+class_vary)
                        model.to(device)                        

                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    this_f1score, this_record, best_weights = train_multiclass_full(model, optimizer, criterion, epochs, model_path, train_loader, val_loader)

                    results['lr']=lr
                    results['drop']=drop
                    results['f1score'] = this_f1score
                    results['batch_size'] = batch_size

                    RESULTS.append(results)

                    with open(result_path1, "wb") as fp:   #Pickling
                      pickle.dump(RESULTS, fp) 

                    if this_f1score > maxval_f1score:
                        maxval_f1score = this_f1score

                        performance_record = this_record

                        with open(record_path1, "wb") as fp:   #Pickling
                          pickle.dump(performance_record, fp)			

                        model.load_state_dict(best_weights)

                        torch.save(model.state_dict(), model_path1) 

                        print('best f1score obtained: ', maxval_f1score)  

        print('best f1score obtained: ', maxval_f1score)
        print(f'fold_{fold} done')



def multiclass_inference_folds(pretrain_model, model_from_pretrain, cnn_class_metrics, base_path, model_path, n_train, n_val, train_val_X, train_val_Y, test_X, test_Y,train_val_groups, class_vary, batch_size = 16, less_volume=True, sequence=True, drop = 0.0, n_splits=10):
    Folds_acc ={'train':[], 'val':[], 'test':[]}
    Folds_f1score = {'train':[], 'val':[], 'test':[]}
    Folds_conf ={'train':[], 'val':[], 'test':[]}
    for fold in range(n_splits):
        train_X, train_Y, val_X, val_Y = 0, 0, 0, 0
        if less_volume:
            group, Train_X, Train_Y, Val_X, Val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)

            train_X, train_Y = sample_one_multiclass(Train_X, Train_Y, n_train, class_vary)
            val_X, val_Y = sample_one_multiclass(Val_X, Val_Y, n_val, class_vary)
        elif not less_volume:
            group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=n_splits, test_fold=fold)

        train_loader = data_loader_test_fn(train_X, train_Y, transform_test, batch_size=batch_size, multilabel=False)
        val_loader = data_loader_test_fn(val_X, val_Y, transform_test, batch_size=batch_size, multilabel=False)

        test_loader = data_loader_test_fn(test_X, test_Y, transform_test, batch_size=batch_size, multilabel=False)

        model = model_from_pretrain(pretrain_model, drop)

        if sequence:
            model.fc[3] = nn.Linear(100, 1+class_vary)
            model.to(device)
            
        model_fold = f'Fold{fold}.pth'
        
        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)
    
        model_path1 = os.path.join(PATH0, model_fold)
                
        model.load_state_dict(torch.load(model_path1))
        model.eval()

        train_acc, train_conf, train_f1score = cnn_class_metrics(model, train_loader, 'train')
        val_acc, val_conf, val_f1score = cnn_class_metrics(model, val_loader, 'val')
        test_acc, test_conf, test_f1score = cnn_class_metrics(model, test_loader,'test')

        Folds_acc['train'].append(train_acc)
        Folds_acc['val'].append(val_acc)
        Folds_acc['test'].append(test_acc)
        Folds_f1score['train'].append(train_f1score)
        Folds_f1score['val'].append(val_f1score)
        Folds_f1score['test'].append(test_f1score)
        
        Folds_conf['train'].append(train_conf)
        Folds_conf['val'].append(val_conf)
        Folds_conf['test'].append(test_conf)        
        
    return Folds_acc, Folds_f1score, Folds_conf



    
def train_multilabel_full(model, optimizer, criterion, epochs, model_path, train_loader, val_loader,train_loader2, val_loader2,label_dict):

    running_loss_list = []
    val_running_loss_list = []
    mean_abs_error_list = []
    val_mean_abs_error_list = []

    maxval_f1score = 0.0
    minval_loss = 100000
    model_return = None
    best_weights = None
    performance_record = {'loss': [], 'val_loss': []}
    for epoch in range(1, epochs): 

        running_loss = []
        val_running_loss = []

        #model_micro.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward  + optimize

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        #model_micro.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                #print('outputs: ', outputs)
                #print('labels: ', labels)
                val_loss = criterion(outputs, labels)
                val_running_loss.append(val_loss.item())



       
        train_f1score= f1score_multilabel(model, train_loader2,label_dict, 'train')

        val_f1score= f1score_multilabel(model, val_loader2,label_dict, 'val')
  
        loss_mean = np.mean(running_loss)
        val_loss_mean = np.mean(val_running_loss)

        running_loss_list.append(float(f'{loss_mean :.4f}'))
        val_running_loss_list.append(float(f'{val_loss_mean:.4f}'))

        print(f'Epoch{epoch}: loss: {loss_mean:.4f} val_loss: {val_loss_mean:.4f}')

        if val_f1score > maxval_f1score and train_f1score > maxval_f1score and loss_mean < minval_loss and val_loss_mean < minval_loss:
            maxval_f1score = val_f1score
            minval_loss = val_loss_mean
            
            print('val_f1score: ', val_f1score)



            performance_record['loss'].append(running_loss_list[:])

            performance_record['val_loss'].append(val_running_loss_list[:])

            best_weights = copy.deepcopy(model.state_dict())


    return maxval_f1score, performance_record, best_weights


def multilabel_train_folds(Batch_size, LR, DROP, criterion, n_train, n_val, train_val_X, train_val_Y, label_dict,train_val_groups, class_vary, base_path, model_path, pretrain_model,model_from_pretrain, PATH=None, less_volume = True, sequence=True, epochs=60, n_splits=10):
    for fold in range(n_splits):
        train_X, train_Y, val_X, val_Y = 0, 0, 0, 0
        if less_volume:
            group, Train_X, Train_Y, Val_X, Val_Y = stratified_train_test_group_kfold_multilabel(train_val_X, train_val_Y, label_dict,train_val_groups, n_splits=n_splits, test_fold=fold)

            train_X, train_Y = sample_one_class(Train_X, Train_Y, label_dict, n_train, class_vary)
            val_X, val_Y = sample_one_class(Val_X, Val_Y, label_dict, n_val, class_vary)
        elif not less_volume:
            group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold_multilabel(train_val_X, train_val_Y, label_dict,train_val_groups, n_splits=n_splits, test_fold=fold)
            
        train_loader2 = data_loader_test_fn(train_X, train_Y,transform, batch_size=1)
        val_loader2 = data_loader_test_fn(val_X, val_Y, transform_test, batch_size=1)

        model_fold = f'Fold{fold}.pth'
        result_fold = f'Fold{fold}_tuning'
        record_fold = f'Fold{fold}_record'

        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)

        result_path1 = os.path.join(PATH0, result_fold)
        model_path1 = os.path.join(PATH0, model_fold)
        record_path1 = os.path.join(PATH0, record_fold)

        RESULTS = []

        model_return = None
        performance_record = {}
        maxval_f1score = 0.0

        for batch_size in Batch_size:

            for lr in LR:
                for drop in DROP:

                    train_loader = data_loader_weighted_multilabel(train_X, train_Y, label_dict,transform, batch_size=batch_size)
                    val_loader = data_loader_test_fn(val_X, val_Y, transform_test, batch_size=batch_size)

                    results = {}
                    model = model_from_pretrain(pretrain_model, drop)
                    if sequence:
                        model.load_state_dict(torch.load(PATH))

                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    this_f1score, this_record, best_weights = train_multilabel_full(model, optimizer, criterion, epochs, model_path, train_loader, val_loader,train_loader2, val_loader2,label_dict)

                    results['lr']=lr
                    results['drop']=drop
                    results['f1score'] = this_f1score
                    results['batch_size'] = batch_size

                    RESULTS.append(results)

                    with open(result_path1, "wb") as fp:   #Pickling
                      pickle.dump(RESULTS, fp) 

                    if this_f1score > maxval_f1score:
                        maxval_f1score = this_f1score

                        performance_record = this_record

                        with open(record_path1, "wb") as fp:   #Pickling
                          pickle.dump(performance_record, fp)			

                        model.load_state_dict(best_weights)

                        torch.save(model.state_dict(), model_path1) 

                        print('best f1score obtained: ', maxval_f1score)  

        print('best f1score obtained: ', maxval_f1score)
        print(f'fold_{fold} done')

    
          
def multilabel_inference_folds(pretrain_model, model_from_pretrain, cnn_class_metrics, base_path, model_path, n_train, n_val, train_val_X, train_val_Y, test_X, test_Y,label_dict,train_val_groups, class_vary, batch_size = 1, less_volume=True, drop = 0.0, n_splits=10):
    Folds_acc ={'train':[], 'val':[], 'test':[]}
    Folds_f1score = {'train':[], 'val':[], 'test':[]}
    Folds_conf ={'train':[], 'val':[], 'test':[]}
    for fold in range(n_splits):
        train_X, train_Y, val_X, val_Y = 0, 0, 0, 0
        if less_volume:
            group, Train_X, Train_Y, Val_X, Val_Y = stratified_train_test_group_kfold_multilabel(train_val_X, train_val_Y, label_dict,train_val_groups, n_splits=n_splits, test_fold=fold)

            train_X, train_Y = sample_one_class(Train_X, Train_Y, label_dict, n_train, class_vary)
            val_X, val_Y = sample_one_class(Val_X, Val_Y, label_dict, n_val, class_vary)
        elif not less_volume:
            group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold_multilabel(train_val_X, train_val_Y, label_dict,train_val_groups, n_splits=n_splits, test_fold=fold)

        train_loader = data_loader_test_fn(train_X, train_Y, transform_test, batch_size=batch_size)
        val_loader = data_loader_test_fn(val_X, val_Y, transform_test, batch_size=batch_size)

        test_loader = data_loader_test_fn(test_X, test_Y, transform_test, batch_size=batch_size)

        model = model_from_pretrain(pretrain_model, drop)
        
        model_fold = f'Fold{fold}.pth'
        
        PATH0 = os.path.join(base_path, 'Models')
        PATH0 = os.path.join(PATH0, model_path)
    
        model_path1 = os.path.join(PATH0, model_fold)
                
        model.load_state_dict(torch.load(model_path1))
        model.eval()

        train_acc, train_conf, train_f1score = cnn_class_metrics(model, train_loader, label_dict, 'train')
        val_acc, val_conf, val_f1score = cnn_class_metrics(model, val_loader, label_dict, 'val')
        test_acc, test_conf, test_f1score = cnn_class_metrics(model, test_loader, label_dict,'test')

        Folds_acc['train'].append(train_acc)
        Folds_acc['val'].append(val_acc)
        Folds_acc['test'].append(test_acc)
        Folds_f1score['train'].append(train_f1score)
        Folds_f1score['val'].append(val_f1score)
        Folds_f1score['test'].append(test_f1score)
        
        Folds_conf['train'].append(train_conf)
        Folds_conf['val'].append(val_conf)
        Folds_conf['test'].append(test_conf)        
        
    return Folds_acc, Folds_f1score, Folds_conf




    
def accuracy_multilabel(trained_model, data_loader,  label_dict, data_type):
    correct = 0
    total = 0
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        labels = labels.detach().cpu()[0].tolist()
        
        outputs = trained_model(images)
        outputs = outputs.detach().cpu()[0].tolist()
        #print('labels: ', labels, 'outputs: ', outputs)
        label_class = label2class_multilabel(label_dict, labels, idx=False)
        pred_class = label2class_multilabel(label_dict, outputs, idx=False)
        
        if label_class == pred_class:
            correct += 1
                
        total += 1
         
    accuracy = 100 * correct / total
    #print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')
    
    return accuracy    


        
def f1score_multilabel(trained_model, data_loader, label_dict, data_type):
    trained_model.eval()
    #with torch.no_grad():
    Labels = []
    Predicted = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        labels = labels.detach().cpu()[0].tolist()

        outputs = trained_model(images)        
        outputs = outputs.detach().cpu()[0].tolist()
        label_class = label2class_multilabel(label_dict, labels, idx=False)
        pred_class = label2class_multilabel(label_dict, outputs, idx=False)
        #print('labels: ', labels, 'outputs: ', outputs)
        Labels.append(label_class)        
        Predicted.append(pred_class)

    f1score = f1_score(Labels, Predicted, labels=list(Counter(Labels).keys()), average='macro')
    
    #print(f'f1score of the network on the {data_type}: {f1score :.2f} ')
    return f1score


def confusion_multilabel(trained_model, data_loader, label_dict, data_type):
    trained_model.eval()
    Labels = []
    Predicted = []
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        labels = labels.detach().cpu()[0].tolist()

        outputs = trained_model(images)
        outputs = outputs.detach().cpu()[0].tolist()
        label_class = label2class_multilabel(label_dict, labels, idx=False)
        pred_class = label2class_multilabel(label_dict, outputs, idx=False)

        Labels.append(label_class)        
        Predicted.append(pred_class)
    
    
    cm_test = confusion_matrix(Labels, Predicted)
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test

    

def accuracy_2classes(trained_model, data_loader, data_type):
    Labels = []
    Predicted = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        #_, predicted = torch.max(outputs.data, 1)

        Labels += labels.cpu().tolist()
        Predicted += outputs.round().cpu().tolist()
       
    accuracy = accuracy_score(Labels, Predicted) * 100
    #print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')
    
    return accuracy    
    

def f1score_2classes(trained_model, data_loader, data_type):
    trained_model.eval()
    #with torch.no_grad():
    Labels = []
    Predicted = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        #_, predicted = torch.max(outputs.data, 1)

        Labels += labels.cpu().tolist()
        Predicted += outputs.round().cpu().tolist()

    f1score = f1_score(Labels, Predicted, average='macro')
    #print(f'f1score of the network on the {data_type}: {f1score :.2f} ')
    return f1score


def confusion_2classes(trained_model, data_loader, data_type):
    Labels = []
    Predicted = []
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        #_, predicted = torch.max(outputs.data, 1)
        #cm_test = confusion_matrix(labels.cpu(), predicted.cpu())
        
        Labels += labels.cpu().tolist()
        Predicted += outputs.round().cpu().tolist()

    cm_test = confusion_matrix(Labels, Predicted)
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test

def accuracy_classification(trained_model, data_loader, data_type):
    correct = 0
    total = 0
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    #print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')
    
    return accuracy
    

def accuracy_regression(trained_model, data_loader, data_type):
	correct = 0
	total = 0
	trained_model.eval()
	#with torch.no_grad():
	for data in data_loader:
		images, labels = data
		images, labels = images.cuda(), labels.cuda()

		outputs = trained_model(images)
		labels =[item for item in labels.cpu().numpy().tolist()]
		outputs = [pred2class(item) for item in outputs.cpu().detach().numpy().tolist()]
		for index, item in enumerate(labels):
		    if labels[index]==outputs[index]:
		    	correct += 1
		    total += 1
	accuracy = 100 * correct / total

	#print(f'Accuracy of the network on the {total} {data_type} images: {accuracy :.1f} %')

	return accuracy

#acc = (y_pred.round() == y_batch).float().mean()    
        
def f1score_cnn_fn(trained_model, data_loader, data_type):
    trained_model.eval()
    #with torch.no_grad():
    Labels = []
    Predicted = []
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)

        Labels += labels.cpu().tolist()
        Predicted += predicted.cpu().tolist()

    f1score = f1_score(Labels, Predicted, average='macro')
    #print(f'f1score of the network on the {data_type}: {f1score :.2f} ')
    return f1score


def confusion_cnn_fn(trained_model, data_loader, data_type):
    Labels = []
    Predicted = []
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)
        _, predicted = torch.max(outputs.data, 1)
        #cm_test = confusion_matrix(labels.cpu(), predicted.cpu())
        
        Labels += labels.cpu().tolist()
        Predicted += predicted.cpu().tolist()

    cm_test = confusion_matrix(Labels, Predicted)
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test

    

def reg_confusion_cnn_fn(trained_model, data_loader, data_type):
    correct = 0
    total = 0
    #trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)

        labels =[item for item in labels.cpu().numpy().tolist()]
        outputs = [pred2class(item) for item in outputs.cpu().detach().numpy().tolist()]
        cm_test = confusion_matrix(labels, outputs)
        #print(f'{data_type} confusion matrix: {cm_test}')

    return cm_test
    

def nnrank_confusion_cnn_fn(trained_model, data_loader, data_type):
    #trained_model.eval()
    Labels = []
    Outputs = []
    for data in data_loader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        outputs = trained_model(images)
        outputs = nnrank2label(outputs)
        labels =[np.rint(item) for item in labels.cpu().numpy().tolist()]
        outputs = [np.rint(item) for item in outputs.cpu().detach().numpy().tolist()]
        Labels += labels
        Outputs += outputs

    cm_test = confusion_matrix(Labels, Outputs)
    #print(f'confusion matrix: {cm_test}')

    return cm_test
        

def rmse_cnn_fn(trained_model, data_loader, data_type):
    trained_model.eval()
    #with torch.no_grad():
    for data in data_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = trained_model(images)

        labels =[item for item in labels.cpu().numpy().tolist()]
        outputs = [item for item in outputs.cpu().detach().numpy().tolist()]

        rmse_test = np.sqrt(mean_squared_error(labels, outputs))
        #print(f'{data_type} rmse: {rmse_test}')

    return rmse_test



def cnn_class_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_test_fn(X, Y, transform_test,batch_size=len(Y))
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_classification(trained_model, data_loader, data_type)
        cm_test = confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    
    
    
def cnn_reg_cross_val_final_test(trained_model, X, Y, data_type, root_path):
	best_test = []
	confusion_matrix_test = []
	rmse_test_folds = []
	for fold in range(10):
        
		data_loader = data_loader_test_reg(X, Y, transform_test,batch_size=len(Y))


		PATH = os.path.join('Models', root_path)
		PATH = os.path.join(PATH, f'{fold}_model.pth')
		trained_model.load_state_dict(torch.load(PATH))
		trained_model.eval()

		acc_test = accuracy_regression(trained_model, data_loader, data_type)
		cm_test = reg_confusion_cnn_fn(trained_model, data_loader, data_type)
		rmse_test = rmse_cnn_fn(trained_model, data_loader, data_type)

		best_test.append(acc_test)
		confusion_matrix_test.append(cm_test)
		rmse_test_folds.append(rmse_test)
	
	return best_test, confusion_matrix_test, rmse_test_folds
	
	
def cnn_nnrank_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_test_reg(X, Y, transform_test,batch_size=len(Y))
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_nnrank(trained_model, data_loader, data_type)
        cm_test = nnrank_confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    	
    
    
def mlp_class_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_mlp(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_classification(trained_model, data_loader, data_type)
        cm_test = confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test
    
    
def mlp_nnrank_cross_val_final_test(trained_model, X, Y, data_type, root_path):
    best_test = []
    confusion_matrix_test = []
    for fold in range(10):
        
        data_loader = data_loader_mlp_reg(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)
        
        
        PATH = os.path.join('Models', root_path)
        PATH = os.path.join(PATH, f'{fold}_model.pth')
        trained_model.load_state_dict(torch.load(PATH))
        trained_model.eval()
        
        acc_test = accuracy_nnrank(trained_model, data_loader, data_type)
        cm_test = nnrank_confusion_cnn_fn(trained_model, data_loader, data_type)
        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    return best_test, confusion_matrix_test    	    
    

def mlp_reg_cross_val_final_test(trained_model, X, Y, data_type, root_path):
	best_test = []
	confusion_matrix_test = []
	rmse_test_folds = []
	for fold in range(10):
        
		data_loader = data_loader_mlp_reg(X, Y, batch_size=len(Y), shuffle=False, drop_last=False)


		PATH = os.path.join('Models', root_path)
		PATH = os.path.join(PATH, f'{fold}_model.pth')
		trained_model.load_state_dict(torch.load(PATH))
		trained_model.eval()

		acc_test = accuracy_regression(trained_model, data_loader, data_type)
		cm_test = reg_confusion_cnn_fn(trained_model, data_loader, data_type)
		rmse_test = rmse_cnn_fn(trained_model, data_loader, data_type)

		best_test.append(acc_test)
		confusion_matrix_test.append(cm_test)
		rmse_test_folds.append(rmse_test)
	
	return best_test, confusion_matrix_test, rmse_test_folds             
            

        
def model_test_regression(train_val_X, train_val_Y, train_val_groups,test_X, test_Y, best_fold, model_path):
    best_test = []
    root_mean_squared_error = []
    confusion_matrix_test = []
    
    for fold in range(10):
        
        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=10, test_fold=fold)
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
        
        PATH = os.path.join('Models', model_path)
        loaded_model = pickle.load(open(PATH, 'rb'))
        loaded_model.fit(train_X, train_Y)
        pred_test_Y =loaded_model.predict(test_X)        
        rmse = np.sqrt(mean_squared_error(test_Y, pred_test_Y))
        pred_test_Y =pred2class(pred_test_Y)
        cm_test = confusion_matrix(test_Y, pred_test_Y)
        acc_test = accuracy_score(test_Y, pred_test_Y)
        
        best_test.append(acc_test)
        root_mean_squared_error.append(rmse)
        confusion_matrix_test.append(cm_test)
    
    return best_test, root_mean_squared_error, confusion_matrix_test
    

    
def model_test_classification(train_val_X, train_val_Y, train_val_groups,test_X, test_Y, best_fold, model_path):
    best_test = []
    confusion_matrix_test = []

    for fold in range(10):
        
        group, train_X, train_Y, val_X, val_Y = stratified_train_test_group_kfold(train_val_X, train_val_Y, train_val_groups, n_splits=10, test_fold=fold)
        train_X, train_Y = oversample.fit_resample(train_X, train_Y)
        
        PATH = os.path.join('Models', model_path)
        loaded_model = pickle.load(open(PATH, 'rb'))
        loaded_model.fit(train_X, train_Y)
        pred_test_Y = loaded_model.predict(test_X)
        pred_val_Y = loaded_model.predict(val_X)
        pred_train_Y = loaded_model.predict(train_X)


        cm_test = confusion_matrix(test_Y, pred_test_Y)
        acc_test = accuracy_score(test_Y, pred_test_Y)

        best_test.append(acc_test)
        confusion_matrix_test.append(cm_test)
    
    return best_test, confusion_matrix_test
        
        


