import yaml
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


def get_hyper():
    with open('hyperparameter.yaml', 'r') as file:
        para = yaml.safe_load(file)
    return para

def create_checkpoint(epoch,model,optimizer):
    checkpoint = {
        'epoch': epoch,
        'model': model(),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }
    torch.save(checkpoint, 'checkpoint{}.pkl'.format(epoch))
    
    
    
def main():
    hyperparameter = get_hyper()
    

if __name__ == '__main__':
    main()