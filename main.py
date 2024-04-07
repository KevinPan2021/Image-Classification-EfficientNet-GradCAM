import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import pandas as pd
import pickle
import platform


# system packages
from model import ResNet18, ResNet50, EfficientNet_b0, EfficientNet_b3
from training import model_training, feedforward
from visualization import plot_confusion_matrix, plot_data_distribution

# supports MacOS and Windows
def GPU_Device():
    system = platform.system()
    if system == "Darwin":
        return 'mps'
    elif system == "Windows":
        return 'GPU'


# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
    
    def __len__(self):
        return len(self.key_to_value)
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key)

    def get_key(self, value):
        return self.value_to_key.get(value)
    
    
# Extract pretrained activations
class SaveFeatures():
    features = None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()
    def remove(self): 
        self.hook.remove()
    
    
# reading imgs and label to a dictionary
def read_imgs(path, transform):
    X, Y = [], []
    for folder in os.listdir(path):
        if folder.endswith('DS_Store'):
            continue
        folder_path = os.path.join(path, folder)
        for imgs in os.listdir(folder_path):
            if not imgs.endswith('.jpg'):
                continue
            img = Image.open(os.path.join(folder_path, imgs)).convert('RGB')
            X.append(transform(img))
            Y.append(folder)
    return X, Y


# read a single image from path
def read_img(path, transform):
    img = Image.open(path) 
    X = [transform(img)]
    Y = [path.split('/')[-2]]
    return X, Y
    


# convert list of tensor to torch matrix
def convert_to_tensor(X, Y, class_ind_pair):
    X = torch.stack(X)
    Y = torch.stack([torch.tensor(class_ind_pair.get_key(y)) for y in Y])
    return X, Y


# preform horizontal flipping to expand the data size and data balancing
def data_augmentation(X, Y, counts):
    # determine the targeted data size after augmenting for each class
    count_arr = np.array(list(counts.values()))
    targeted_counts = int(np.percentile(count_arr, 5)*2) # double the 5th percentile
    
    # precalculate the total number of augmnentations (faster than torch.cat)
    total_aug = 0
    for key, value in counts.items():
        total_aug += min(max(0, targeted_counts - value), value)
        
    total_X = torch.empty(total_aug+X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    total_Y = torch.empty(total_aug+Y.shape[0])
    
    num_aug = 0
    for key, value in counts.items():
        count = min(max(0, targeted_counts - value), value)
        
        if count <= 0:
            continue
        
        ind = np.random.choice(np.argwhere(Y==key)[0], size=count, replace=False)
               
        # horizontial flipping X
        sampled_X = torch.flip(X[ind, :, :, :], dims=(3,))
        sampled_Y = torch.zeros(len(ind)) + key
        
        # concatenate to original X and Y
        total_X[num_aug: num_aug+len(ind), :, :, :] = sampled_X
        total_Y[num_aug: num_aug+len(ind)] = sampled_Y
        
        num_aug += len(ind)
    
    total_X[num_aug:, :, :, :] = X
    total_Y[num_aug:] = Y
        
    return total_X, total_Y


def main():    
    # read labels from csv and create class to label pairs
    df = pd.read_csv('data/sports.csv')
    labels = df['labels']
    classes = sorted(list(set(labels)))
    nclasses = len(classes)
    class_ind_pair = BidirectionalMap()
    for ind, name in enumerate(classes):
        class_ind_pair.add_mapping(ind, name)
    # Save the instance to a pickle file
    with open("class_ind_pair.pkl", "wb") as f:
        pickle.dump(class_ind_pair, f)
        
        
    # loading model
    #model = ResNet18(nclasses)
    model = ResNet50(nclasses)
    #model = EfficientNet_b0(nclasses)
    #model = EfficientNet_b3(nclasses)
    model = model.to(GPU_Device())
    
    # image transform
    if type(model).__name__ == 'EfficientNet_b3':
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    
    # reading train data
    trainX, trainY = read_imgs('data/train', transform)
    
    # reading validation data
    validX, validY = read_imgs('data/valid', transform) 
    
    # convert to tensor
    trainX, trainY = convert_to_tensor(trainX, trainY, class_ind_pair)
    print('train', trainX.shape, trainY.shape)
    validX, validY = convert_to_tensor(validX, validY, class_ind_pair)
    print('valid', validX.shape, validY.shape)
    
    
    # plot the data distribution
    counts = plot_data_distribution(trainY, class_ind_pair)
    
    # data augmentation
    np.random.seed(42)
    trainX, trainY = data_augmentation(trainX, trainY, counts)
    
    # replot data distribution
    plot_data_distribution(trainY, class_ind_pair)
    
    # convert to loader object
    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(validX, validY), batch_size=32, shuffle=True)
    
    # model training
    model_training(model, nclasses, train_loader, valid_loader, GPU_Device())
    
    # loading the best preforming model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # model test preformance
    testX, testY = read_imgs('data/test', transform) 
    testX, testY = convert_to_tensor(testX, testY, class_ind_pair)
    test_loader = DataLoader(TensorDataset(testX, testY), batch_size=32)
    confusion_mat, test_accuracy, test_loss = feedforward(model, test_loader)
    print('finished inference')
    print('test acc', test_accuracy)
    
    plot_confusion_matrix(confusion_mat, class_ind_pair)

    
if __name__ == '__main__':
    main()