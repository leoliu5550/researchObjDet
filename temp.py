import yaml


with open('hyperparameter.yaml', 'r') as file:
    data = yaml.safe_load(file)
print(data)



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