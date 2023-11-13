from diffusers.optimization import *
from torch.optim import *
import torch

num_train_steps = 100 * 100
num_warmup_steps = 100 * 10
optimizer = SGD([torch.randn(0, 100, 100)], lr=0.1)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_train_steps)
print(lr_scheduler.state_dict())


