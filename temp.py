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



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set two handlers
log_file = "{}.log".format(__name__)
# rm_file(log_file)
fileHandler = logging.FileHandler(os.path.join('log', log_file), mode = 'w')
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

logger.info("test95195195151")