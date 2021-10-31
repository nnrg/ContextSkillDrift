import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

MAX_EPOCHS = 200

class NoopScheduler:
    def step(self):
        pass

def get_optimizer_and_scheduler(net: torch.nn.Module):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-2)
    # scheduler = StepLR(optimizer, step_size=MAX_EPOCHS // 2, gamma=0.1)
    scheduler = NoopScheduler()
    return optimizer, scheduler

