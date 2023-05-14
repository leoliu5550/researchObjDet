#region import_part
import yaml,os,logging,logging.config
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
#endregion

#region logging


logging.config.fileConfig('cfg/logger.conf')
# root_logger = logging.getLogger('root')
# root_logger.debug('MainProg:Test Root Logger...')
logger = logging.getLogger('main')
# logger.info('Test Main Logger')
# logger.info("test")
#endregion

CFG_PATH = 'cfg/hyperparameter.yaml'

def get_hyper():
    with open(CFG_PATH, 'r') as file:
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