import wandb
import random


wandb.init(
    # set the wandb project where this run will be logged
    project="DETR_Research",
    
    # track hyperparameters and run metadata
    config={
    "name":"losslos",
    "learning_rate": 0.02,
    "architecture": "TestRun",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()