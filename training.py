import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

from visualization import plot_training_curves


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


@torch.no_grad
def feedforward(model, data_loader, confusion=False):
    # Set model to evaluation mode
    model.eval()

    running_loss = 0.0
    labels_list = []
    pred_list = []

    # use focal loss to mitigate data imbalance
    criterion = FocalLoss()

    device = next(model.parameters()).device
    
    with tqdm(total=len(data_loader)) as pbar:
        # Iterate over the dataset
        for i, (X, Y) in enumerate(data_loader):
            # move to device
            X = X.to(device)
            Y = Y.to(device)
    
            # Forward pass
            outputs = model(X)
    
            loss = criterion(outputs, Y)
    
            # Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            labels_list.extend(Y.tolist())
            pred_list.extend(predicted.tolist())
            
            compare = [label == predicted for label, predicted in zip(labels_list, pred_list)]

            # Update tqdm description with loss and accuracy
            pbar.set_postfix({'Loss': running_loss/(i+1), 'Acc': 100*sum(compare)/len(compare)})
            pbar.update(1)
    
    running_loss /= len(data_loader)
    acc = sum(compare)/len(compare)
    if confusion:
        confusion_mat = confusion_matrix(labels_list, pred_list)
        return confusion_mat, acc, running_loss
    else:
        return acc, running_loss



def backpropagation(model, data_loader, optimizer, scaler, confusion=False):
    # Set model to training mode
    model.train()

    running_loss = 0.0
    labels_list = []
    pred_list = []

    # use focal loss to mitigate data imbalance
    criterion = FocalLoss()
    device = next(model.parameters()).device
    
    # Iterate over the dataset
    # Wrap your training loop with tqdm for progress tracking
    with tqdm(total=len(data_loader)) as pbar:
        for i, (X, Y) in enumerate(data_loader):
            # move to device
            X = X.to(device)
            Y = Y.to(device)
    
            # mixed precision training
            with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                outputs = model(X)
                loss = criterion(outputs, Y)
    
            # Update training statistics# Update statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            labels_list.extend(Y.tolist())
            pred_list.extend(predicted.tolist())
            
            compare = [label == predicted for label, predicted in zip(labels_list, pred_list)]

            # Reset gradients
            optimizer.zero_grad()
    
            # Backpropagate the loss
            scaler.scale(loss).backward()
    
            # Optimization step
            scaler.step(optimizer)
    
            # Updates the scale for next iteration.
            scaler.update()
            
            # Update tqdm description with loss and accuracy
            pbar.set_postfix({'Loss': running_loss/(i+1), 'Acc': 100*sum(compare)/len(compare)})
            pbar.update(1)
            
            
    # Calculate training accuracy and loss
    running_loss /= len(data_loader)
    acc = sum(compare)/len(compare)
    if confusion:
        confusion_mat = confusion_matrix(labels_list, pred_list)
        return confusion_mat, acc, running_loss
    else:
        return acc, running_loss



def model_training(model, nclasses, train_loader, valid_loader):
    # Define hyperparameters
    learning_rate = 5e-4
    weight_decay = 0
    num_epochs = 60

    # optimizer
    optimizer = optim.Adam(
        model.parameters(), weight_decay=weight_decay, lr=learning_rate)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # calculate the initial statistics (random)
    print(f"Epoch {0}/{num_epochs}")
    train_acc, train_loss = feedforward(model, train_loader)
    valid_acc, valid_loss = feedforward(model, valid_loader)
    train_loss_curve, train_acc_curve = [train_loss], [train_acc]
    valid_loss_curve, valid_acc_curve = [valid_loss], [valid_acc]
    

    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_acc, train_loss = backpropagation(model, train_loader, optimizer, scaler)
        valid_acc, valid_loss = feedforward(model, valid_loader)

        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        valid_loss_curve.append(valid_loss)
        valid_acc_curve.append(valid_acc)


        # evaluate the current preformance
        if valid_loss < best_valid_loss + threshold:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}.pth')
        else:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break

    # plotting the training curves
    plot_training_curves(train_loss_curve, valid_loss_curve, train_acc_curve, valid_acc_curve)
