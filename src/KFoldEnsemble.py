import torch
import os
# import glob
from train import create_model  # only import related to the function in this project (defines the model used)

class KFoldEnsemble:
    def __init__(self, model_dir, model_name, device, num_folds=5):
        self.models = []
        self.device = device
        self.num_folds = num_folds
        
        # Load all fold models
        for fold in range(1, num_folds + 1):
            # Try different possible model paths
            possible_paths = [
                os.path.join(model_dir, f'fold{fold}', 'best_metric_model.pth'),
                os.path.join(model_dir, f'single_fold_{fold-1}', 'best_metric_model.pth')
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path and os.path.exists(model_path):
                print(f"Loading model from fold {fold}: {model_path}")
                model = create_model(model_name, device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                
                model.eval()  # Set to evaluation mode
                self.models.append(model)
            else:
                print(f"Warning: Model not found for fold {fold}")
    
    def predict(self, volume):
        """
        Make prediction using average ensemble
        volume: input tensor of shape [batch, channels, depth, height, width]
        returns: averaged prediction
        """
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(volume)
                # Apply sigmoid if your loss uses sigmoid (like DiceLoss with sigmoid=True)
                pred = torch.sigmoid(pred)
                predictions.append(pred)
            
            # Average all predictions
            if predictions:
                ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
                return ensemble_pred
            else:
                raise ValueError("No models loaded for ensemble")
    
    def __len__(self):
        return len(self.models)
