import monai.networks.nets as nets
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import numpy as np
import torch
from Preprocessing_kfoldCV import define_files, get_transforms, create_kfold_split
from Util import calculate_weights, train

import os
#import json

data_path = 'D:/personal projects/Image segmentation with Pythorch and MONAI/Img'
# where to save model, save_loss, Dice loss 
model_dir = 'D:/personal projects/Image segmentation with Pythorch and MONAI/MODEL'


device = torch.device('cuda:0')

def create_model(model_name, device="cuda"):
    if model_name == "attention_unet":
        model = nets.AttentionUnet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif model_name == "unet_plus_plus":
        model = nets.UnetPlusPlus(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2, #keep residaual units
            norm=Norm.BATCH, #keep batch norm
        )
    elif model_name == 'standard_unet':
            model = nets.UNet(
            dimensions = 3, 
            in_channels = 2, # each slice has 2 channel 
            out_channels = 2, # 2 because is the number of classes we have, foreground+background
            channels = (16, 32, 64, 128, 256), # how many fitters in the convolution block
            strides = (2,2,2,2),
            num_res_units = 2,
            norm = Norm.BATCH,
            dropout=0.2,  # Add dropout for regularization
        )
    return model.to(device)

def create_loss_function(device):
    ''' create Loss function '''
    # loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
    loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True) 
    
    return loss_function

def run_kfold_training(model_name='standard_unet', max_epochs =50, learning_rate=1e-5):
    '''Main function, will run the Kfold cross validation training'''
    
    # get files
    train_files, test_files = define_files(data_path)
    # create k fold split
    split = create_kfold_split(train_files, n_splits=5)

    # var to store the results from each fold
    fold_results = []

    # model = create_model('attention_unet', device)
    # print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    # optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

    # Train for each fold
    for fold, (train_idxs, val_idxs) in enumerate(split):
        print('#'*20)
        print(f'Training fold {fold + 1}')
        print('#'*20)
        
        # create model directory for this fold
        fold_model_dir = os.path.join(model_dir, f'fold{fold+1}')
        os.makedirs(fold_model_dir, exist_ok=True)

        # create model for this fold
        model = create_model(model_name, device)
        print(f'model: {model_name}')
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

        # def optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay=1e-5,
            amsgrad=True
        )
        loss_function = create_loss_function(device)

        #create loaders for this specific fold
        fold_train_files = [train_files[i] for i in train_idxs]
        fold_val_files = [train_files[i] for i in val_idxs]

        print(f"Fold {fold + 1} - Training samples: {len(fold_train_files)}")
        print(f"Fold {fold + 1} - Validation samples: {len(fold_val_files)}")
        
        train_loader, val_loader, test_loader = get_transforms(
            fold_train_files, 
            fold_val_files, 
            test_files, 
            cache = True, 
            augment = True)
        
        # we can train the model
        print(f'start training fold {fold+1}')
        try:
            train(
                model=model,
                data_in=(train_loader, val_loader),
                loss=loss_function,
                optim=optimizer,
                max_epochs=max_epochs,
                model_dir=fold_model_dir,
                test_interval=1,
                device=device
            )
            #Load metric and fold results
            train_losses = np.load(os.path.join(fold_model_dir,'loss_train.npy'))
            val_losses = np.load(os.path.join(fold_model_dir,'loss_val.npy'))
            train_metrics = np.load(os.path.join(fold_model_dir,'metric_train.npy'))
            val_metrics = np.load(os.path.join(fold_model_dir,'metric_val.npy'))
            
            # create dictionary to store fold results
            fold_result = {
                 'fold': fold+1,
                 'best_val_dice': val_metrics.max(),
                 'final_train_loss': train_losses[-1],
                 'final_val_loss': val_losses[-1],
                 'final_train_dice': train_metrics[-1],
                 'final_val_dice': val_metrics[-1]
            }
            #add it to the list
            fold_results.append(fold_result)
            print(f'Fold {fold+1} completed succesfuly')
            print(f"Best validation dice coeff: {fold_result['best_val_dice']:.4f}")
        except Exception as e:
             print(f'Error training fold {fold+1}: {str(e)}')
             continue
        
        print(rf'Fold {fold+1} training complete (^.^) \n')

    # summary of all folds
    print('='*25)
    print('K FOLD CROSS VALIDATION SUMMARY')
    print('='*25)


    if fold_results:
         val_dices = [result['best_val_dice'] for result in fold_results]
         mean_dice = np.mean(val_dices)
         std_dice = np.std(val_dices)

         print(f"Mean validation Dice across folds: {mean_dice:.4f} ± {std_dice:.4f}")
         print("\nDetailed results by fold:")

         for result in fold_results:
              print(f"FOLD {result['fold']}: best val dice = {result['best_val_dice']:,.4f}")
        
        # save summary results
         summary_path = os.path.join(model_dir, 'kfold_summary.npy')
         np.save(summary_path, fold_results)
         print(f'\nsave results to {summary_path}')

         return fold_results
    else:
          print('Training unsuccesful :-( ')
          return None


##########################################            
def train_single_fold(fold_idx=0, model_name="standard_unet", max_epochs=100, learning_rate=1e-5):
    """Train a single fold for testing purposes"""
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Get files and create split
    train_files, test_files = define_files(data_path)
    splits = create_kfold_split(train_files, n_splits=5)
    
    if fold_idx >= len(splits):
        raise ValueError(f"Fold index {fold_idx} out of range. Available folds: 0-{len(splits)-1}")
    

    train_idxs, val_idxs = splits[fold_idx]
    
    # Create model and training components
    model = create_model(model_name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    loss_function = create_loss_function(device)
    
    # Prepare data
    fold_train_files = [train_files[i] for i in train_idxs]
    fold_val_files = [train_files[i] for i in val_idxs]
    
    train_loader, val_loader, test_loader = get_transforms(
        fold_train_files, fold_val_files, test_files, cache=True, augment=True
    )

    #         # Add this debugging code after get_transforms()
    # print("=== DEBUGGING DATA SHAPES ===")
    # for i, batch in enumerate(train_loader):
    #     print(f"Batch {i}:")
    #     print(f"Image shape: {batch['image'].shape}")
    #     print(f"Label shape: {batch['label'].shape}")
    #     print(f"Image dtype: {batch['image'].dtype}")
    #     print(f"Label dtype: {batch['label'].dtype}")
    #     if i >= 2:  
    #         break
    # print("=== END DEBUGGING ===")

    # Train
    fold_model_dir = os.path.join(model_dir, f'single_fold_{fold_idx}')
    os.makedirs(fold_model_dir, exist_ok=True)
    
    train(
        model=model,
        data_in=(train_loader, val_loader),
        loss=loss_function,
        optim=optimizer,
        max_epochs=max_epochs,
        model_dir=fold_model_dir,
        test_interval=1,
        device=device
    )
################################à

if __name__ == '__main__':
     print('starting Kfold CV training: \n')
     results = run_kfold_training(          
          model_name = 'attention_unet',
          max_epochs = 300, #300
          learning_rate = 1e-4)

    # train_single_fold(fold_idx=0, model_name="attention_unet", max_epochs=10)
