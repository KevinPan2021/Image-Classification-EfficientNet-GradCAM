import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from visualization import plot_training_curves

def feedforward(model, data_loader, GPU_Device):
    epoch_loss = 0.0
    labels_list = []
    predicted_list = []
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()   
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        
        # Iterate over the dataset
        for X, Y in data_loader:
            # move to device
            X = X.to(GPU_Device)
            Y = Y.to(GPU_Device)
            
            # Forward pass
            outputs = model(X)
            
            loss = criterion(outputs, Y)
            
            # Update test statistics
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            labels_list.append(Y.cpu())
            predicted_list.append(predicted.cpu())
            
            
    labels_list = torch.cat(labels_list, dim=0)
    predicted_list = torch.cat(predicted_list, dim=0)
    
    test_correct = predicted_list.eq(labels_list).sum().item()
    confusion_mat = confusion_matrix(labels_list, predicted_list)
    
    # Calculate test accuracy and loss
    epoch_acc = 100 * test_correct / len(data_loader.dataset)
    epoch_loss /= len(data_loader)
    return confusion_mat, epoch_acc, epoch_loss




def backpropagation(model, data_loader, optimizer, scheduler, GPU_Device):
    epoch_acc = 0.0
    epoch_loss = 0.0
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set model to training mode
    model.train()
    
    # Iterate over the dataset
    for X, Y in tqdm(data_loader):
        
        # move to device
        X = X.to(GPU_Device)
        Y = Y.to(GPU_Device)
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Update training statistics
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += predicted.eq(Y).sum().item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    
    # Update the learning rate
    scheduler.step()
    
    # Calculate training accuracy and loss
    epoch_acc = 100 * epoch_acc / len(data_loader.dataset)
    epoch_loss /= len(data_loader)
    
    return epoch_acc, epoch_loss

   

def model_training(model, nclasses, train_loader, valid_loader, GPU_Device):    
    # Define hyperparameters
    learning_rate = 1e-3
    weight_decay = 0
    num_epochs = 30
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)
    
    # learning rate scheduler - gradually decrease learning rate over time
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # calculate the initial statistics (random)
    _, train_acc, train_loss = feedforward(model, train_loader, GPU_Device)
    _, valid_acc, valid_loss = feedforward(model, valid_loader, GPU_Device)
    train_loss_curve, train_acc_curve = [train_loss], [train_acc]
    valid_loss_curve, valid_acc_curve = [valid_loss], [valid_acc]
    print(f"Epoch {0}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {valid_loss:.4f} | Test Acc: {valid_acc:.2f}%")
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # Training loop
    for epoch in range(num_epochs):
        train_acc, train_loss = backpropagation(model, train_loader, optimizer, scheduler, GPU_Device)
        _, valid_acc, valid_loss = feedforward(model, valid_loader, GPU_Device)
        
        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        valid_loss_curve.append(valid_loss)
        valid_acc_curve.append(valid_acc)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {valid_loss:.4f} | Test Acc: {valid_acc:.2f}%")
        
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