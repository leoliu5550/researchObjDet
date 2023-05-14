# print(yaml.dump(data))
# print('---')
# print(type(data))
# print(data)
# print('---')
# print(data['name'])
# print(data['age'])
# import torchvision
# print(dir(torchvision.models))

# print(getattr(torchvision.models, name))


import os,logging

import logging,logging.config
import temp2 as mod

logging.config.fileConfig('cfg/logger.conf')
root_logger = logging.getLogger('root')

root_logger.debug('MainProg:Test Root Logger...')
logger = logging.getLogger('main')
logger.info('Test Main Logger')

mod.testLogger()#子模块


