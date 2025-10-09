
from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    The Dice coefficient measures the overlap between predicted and ground truth segmentations,
    with values closer to 1 indicating better performance.
    '''
    dice_val = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_val(predicted, target).item()

    return value

def calculate_weights(val1, val2):
    '''
    Computes class weights for handling imbalanced datasets
    takes number of background and foreground pixels to return weights for the cross
    entropy loss value
    '''
    count = np.array([val1,val2])
    somma = count.sum()
    weights = count/somma
    weights = 1/weights
    somma = weights.sum()
    weights = weights/somma
    return torch.tensor(weights, dtype = torch.float32)

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1, device = torch.device('cuda:0')):
    '''
    Core training loop for the medical image segmentation model.
    model: The neural network model to train
    data_in: Tuple containing (train_loader, test_loader) - PyTorch DataLoaders
    loss: Loss function (DiceLoss)
    optim: Optimizer ()
    max_epochs: Total number of training epochs
    model_dir: Directory path to save model checkpoints and metrics
    test_interval: How often to run validation (default is every epoch)
    device: Computing device (default is GPU)
    '''

    # Sets up tracking variables for:
    # best validation Dice score and the epoch it occurred
    # Lists to store training/validation losses and metrics for plotting later
    # Unpacks the data loaders
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_val = []
    save_metric_train = []
    save_metric_val = []
    train_loader, val_loader = data_in

    for epoch in range(max_epochs):
        print('--'*10)
        print(f'epoch {epoch + 1}/{max_epochs}')
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        # train phase
        for batch_data in train_loader:
            train_step += 1

            #prepare Data
            volume = batch_data['image']
            label = batch_data['label']
            label = label != 0 # convert multi class labels to binary (background=0, foreground=1) (Boolean Tensor)
            volume, label = (volume.to(device), label.to(device))
            # forward pass
            optim.zero_grad() 
            outputs = model(volume) # get prediction
            train_loss = loss(outputs,label) # calc loss
            # backward pasee
            train_loss.backward() #compute gradients
            optim.step() #update model param

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)

        # avg metric over all batches
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')
        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        # validation phase
        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad(): # no gradient computation: + efficient
                val_epoch_loss = 0
                val_metric = 0
                epoch_metric_val = 0
                val_step = 0

                for val_data in val_loader:
                    val_step += 1
                    # data prep
                    val_volume = val_data["image"]
                    val_label = val_data["label"]
                    val_label = val_label != 0
                    val_volume, val_label = (val_volume.to(device), val_label.to(device),)
                    # forward pass only (no backpropagation)
                    val_outputs = model(val_volume)
                    val_loss = loss(val_outputs, val_label)
                    val_epoch_loss += val_loss.item()
                    val_metric = dice_metric(val_outputs, val_label)
                    epoch_metric_val += val_metric

                # average validation metrics
                val_epoch_loss /= val_step
                print(f'validation_loss_epoch: {val_epoch_loss:.4f}')
                save_loss_val.append(val_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_val.npy'), save_loss_val)

                epoch_metric_val /= val_step
                print(f'validation_dice_epoch: {epoch_metric_val:.4f}')
                save_metric_val.append(epoch_metric_val)
                np.save(os.path.join(model_dir, 'metric_val.npy'), save_metric_val)

                #save best model (Only saves models that improve validation performance)
                # to det best model uses dice coefficient
                if epoch_metric_val > best_metric:
                    best_metric = epoch_metric_val
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, 'best_metric_model.pth'
                    ))

                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    
def show_patient(data, SLICE_NUMBER=1, train=True, test=False):
    '''
    This function is to show one patient from your datasets.
    data: takes patients from the data loader
    SLICE_NUMBER shows the slice of the patient volume
    train: display patient of training data
    test: display  patient of test data
    '''
    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)
    if train:
        plt.figure('Visual Train', (12,6))
    
        plt.subplot(1,2,1)
        plt.title(f'image {SLICE_NUMBER}')
        plt.imshow(view_train_patient['image'][0 , 0 , : , : ,SLICE_NUMBER], cmap = 'gray')
        plt.subplot(1,2,2)
        plt.title(f'label {SLICE_NUMBER}')
        plt.imshow(view_train_patient['image'][0 , 0 , : , : ,SLICE_NUMBER])
        
        plt.show()
    if test:
        plt.figure('Visual Test', (12,6))
    
        plt.subplot(1,2,1)
        plt.title(f'image {SLICE_NUMBER}')
        plt.imshow(view_test_patient['image'][0 , 0 , : , : ,SLICE_NUMBER], cmap = 'gray')
        plt.subplot(1,2,2)
        plt.title(f'label {SLICE_NUMBER}')
        plt.imshow(view_test_patient['image'][0 , 0 , : , : ,SLICE_NUMBER])
        
        plt.show()

def calculate_pixels(data):
    val = np.zeros((1,2))

    for batch in tqdm(data):
        batch_label = batch['image'] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count)==1:
            counr = np.append(count,0)
        val += count

        print('the last values:',val)
        return val    
