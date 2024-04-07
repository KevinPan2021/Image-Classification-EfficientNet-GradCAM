import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from collections import Counter


# plot the data distribution (occurrence)
def plot_data_distribution(Y, class_ind_pair):
    Y = Y.numpy()
    counts = Counter(Y)
    
    fig, ax = plt.subplots(figsize = (30, 15))
    ax.set_xlim(-1, len(counts))
    
    indices = np.arange(len(counts))
    occurrence = [counts[i] for i in indices]
    cls_names = [class_ind_pair.get_value(i) for i in indices]
    ax.bar(indices, occurrence, 0.7, color = "firebrick")
    
    # Rotate x-axis labels
    ax.set_xticks(range(len(cls_names)))  # Set tick positions
    ax.set_xticklabels(cls_names, rotation = 90)
    
    
    ax.set(xticks = indices, xticklabels = cls_names)
    ax.set_ylabel("Counts", fontsize=15)
    ax.set_title('data distribution', fontsize=25)

    for i, v in enumerate(occurrence): 
        ax.text(i - 0.5, v + 2, str(v), color = "royalblue", fontsize=10)
    return counts
    


# plot the loss and acc curves
def plot_training_curves(train_loss, valid_loss, train_acc, valid_acc):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('binary cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('accuracy curves')
    plt.xlabel('epochs')
    plt.ylabel('accuracy in %')
    plt.legend()
    
    

# plot the confusion matrix from 2d matrix
def plot_confusion_matrix(confusion_mat, class_ind_pair):        
    category = []
    for i in range(len(class_ind_pair.key_to_value)):
    
        category.append(class_ind_pair.get_value(i))
        
    # plot the confusion matrix
    df_cm = pd.DataFrame(confusion_mat, index = category, columns = category)
    fig, ax = plt.subplots(figsize = (24,20))
    
    ax.set_xlabel('prediction')
    ax.set_ylabel('ground truth')
    ax.set_title('ConfusionMatrix', fontsize=30)
    sn.heatmap(df_cm, annot=True, fmt='g')
    
    plt.tight_layout()
    
