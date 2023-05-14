#region import_part
import yaml,os,logging,datetime
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
#endregion

#region logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# set two handlers
log_file = "{}.log".format(__name__)
# rm_file(log_file)
fileHandler = logging.FileHandler(os.path.join('log', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.log' ), mode = 'w')
fileHandler.setLevel(logging.DEBUG)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

# set formatter
formatter = logging.Formatter('[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# add
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

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