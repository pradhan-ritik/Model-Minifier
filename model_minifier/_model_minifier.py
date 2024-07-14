import torch
import torch.nn as nn
import torch.optim
import numpy as np


class Epochs(int):
    def __init__(self, epoch_num: int) -> None:
        super().__init__()

    def _end(self, last1, last2):
        return self <= last2

class Still_Improving(float):
    def __init__(self, improving_precision: float) -> None:
        super().__init__()

    def _end(self, last1, last2):
        return self <= abs(last1 - last2)

def minify_model(X_train: np.array, base_model: nn.Module, new_model: nn.Module, loss_fn: nn.modules.loss._Loss = nn.MSELoss, optimizer: torch.optim.Optimizer = torch.optim.Adam, optimizer_parameters: dict = dict(lr=0.01), train_limit: Epochs | Still_Improving = Still_Improving(0.0001), verbose: bool = True) -> None:
    c_base_model = base_model
    c_base_model.train(False)
    y = c_base_model(X_train)
    loss_fn = loss_fn()
    optimizer = optimizer(new_model.parameters(), **optimizer_parameters)
    is_epochs_limit = isinstance(train_limit, Epochs)

    last1, last2 = 1, 0
    while train_limit._end(last1, last2):
        if is_epochs_limit:
            last2 += 1
        
        else:
            last1 = last2
            if last2 == 0:
                last1 = 1
            

        optimizer.zero_grad()
        y_pred = new_model.forward(X_train)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if verbose:
            if is_epochs_limit:
                print(f"epoch {last2}/{train_limit} loss: {loss.item()}")

            else:
                last2 = loss.item()
                print(f"loss: {last2} improvement: {last1-last2}")


