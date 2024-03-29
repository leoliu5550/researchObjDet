#region import_part
import dynamic_yaml
import logging
import logging,logging.config
import torch
from model.backbone import backbonebase
#endregion

#region logging
logging.config.fileConfig('./cfg/logger.conf')
# root_logger = logging.getLogger('root')
# root_logger.debug('MainProg:Test Root Logger...')
logger = logging.getLogger('main')
# logger.info('Test Main Logger')
# logger.info("test")
#endregion

CFG_PATH = './cfg/hyperparameter.yaml'

def get_hyper():
    with open(CFG_PATH, 'r') as file:
        para = dynamic_yaml.load(file)
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
    logger.debug(hyperparameter)
    model = backbonebase()
    logger.debug(model[:-2])

if __name__ == '__main__':
    main()