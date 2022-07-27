import torch
####################################
#   Create Checkpoint Builder
####################################

def checkpoint_builder(model, optimizer, epoch, save_checkpoint_path):
    """
    desc:
        creates checkpoint for the model
    params:
        model: trained model
        optimizer
        criterion: loss fn
        epoch: checkpoint epoch
        model_params: hyperparams of model
        data_module: object of DataModule => contains the datasets and vocabulary
        save_checkpoint_path: path to save checkpoint. Include checkpoint name
    """
    
    torch.save({'model_state_dict': model.state_dict(),
                        'optim_state_dict':optimizer.state_dict(),
                        
                        'epoch': epoch, 
                       }, save_checkpoint_path)

