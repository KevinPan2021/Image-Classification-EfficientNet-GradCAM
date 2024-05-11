import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import pickle
from collections import Counter

# system packages
from efficientnet import EfficientNet_b0
from training import model_training, feedforward
from visualization import plot_confusion_matrix, plot_data_distribution, plot_image
        

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
    

# custom dataset class, read images from folder
class CustomDataset(Dataset):
    def __init__(self, root_dir, class_ind_pair, transform=None):
        self.root_dir = root_dir
        self.class_ind_pair = class_ind_pair
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        
        self.image_paths = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            label = self.class_ind_pair.get_key(folder)
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                self.image_paths.append((img_path, label))
                
                
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
    
    def get_label_distribution(self):
        counter = Counter()
        for path in self.image_paths:
            counter.update({path[-1]: 1})
        return counter
    



# supports CUDA, MPS, and CPU
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'



# return the image transformation (with/without) data augmentation
def get_transform(aug=False):
    # image transform with augmentation
    if aug:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # image transform without augmentation
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform



def main():    
    # dataset
    dataset = '../Datasets/100 Sports Image Classification'
    
    # read labels from csv and create class to label pairs
    df = pd.read_csv(f'{dataset}/sports.csv')
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
    model = EfficientNet_b0(nclasses)
    model = model.to(compute_device())

    # reading train data
    train_dataset = CustomDataset(f'{dataset}/train', class_ind_pair, get_transform(aug=True))

    # reading validation data
    valid_dataset = CustomDataset(f'{dataset}/valid', class_ind_pair, get_transform()) 

    # visualize some data
    for i in range(0, len(train_dataset), len(train_dataset)//6):
        x, y = train_dataset[i]
        plot_image(i, x, class_ind_pair.get_value(y)) 

    # plot the data distribution
    plot_data_distribution(train_dataset.get_label_distribution(), class_ind_pair)

    # convert to loader object
    train_loader = DataLoader(
        train_dataset, batch_size=128, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=256, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    
    # model training
    model_training(model, nclasses, train_loader, valid_loader)
    
    # loading the best preforming model
    model.load_state_dict(torch.load(f'{type(model).__name__}.pth'))
    
    # model test preformance
    test_dataset = CustomDataset(f'{dataset}/test', class_ind_pair, get_transform())
    
    test_loader = DataLoader(
        test_dataset, batch_size=256, num_workers=4, pin_memory=True, 
        persistent_workers=True, shuffle=False
    )
    
    print('Test Preformance')
    confusion_mat, test_accuracy, test_loss = feedforward(model, test_loader, confusion=True)
    
    plot_confusion_matrix(confusion_mat, class_ind_pair)

    
if __name__ == '__main__':
    main()