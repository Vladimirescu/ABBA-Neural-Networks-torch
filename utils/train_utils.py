import math
import importlib
import torch
import torch.nn as nn


def get_object(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    class_object = getattr(module, class_name)
    
    return class_object


def get_agg_ops(ops_dict):
    """
    Return a list of aggregation operator objects from dictionary config.
    """    
    agg_ops = []
    
    for obj_config in ops_dict:
        params = obj_config.params
        cls = get_object(obj_config.target)
        agg_ops.append(cls(**params))
        
    return agg_ops
        

def schedule_lip(epoch, epochs_warmup, lazy_start_epochs, lip, lip_start, typ="exp"):
    if epoch <= lazy_start_epochs:
        return lip_start

    if typ == "exp":
        return max(lip, lip_start * math.exp(- (epoch - lazy_start_epochs - 1) * math.log(lip_start / lip) / epochs_warmup))
    else:
        """Linear decrease"""
        return max(lip, lip_start * (1 - (epoch - lazy_start_epochs - 1) / epochs_warmup) + lip)


class ModelBuffer(nn.Module):
    """
    Buffer for keeping the previous #buffer_size models' state_dicts and optimizer state_dicts,
    in order to restore when training becomes unstable.
    """
    def __init__(self, buffer_size, model, optimizer, device, *args, **kwargs):

        self.buffer_size = buffer_size
        self.buffer_states = [
            {
                "state_dict": model.state_dict(),
                "opt_dict": optimizer.state_dict(),
            } 
            for _ in range(self.buffer_size)
        ]

        self.buffer_losses = [
            torch.tensor(torch.inf, device=device) for _ in range(self.buffer_size)
        ]

        self.current_index = 0
        self.current_state_dict = None

    def update(self, model, optimizer, loss):

        self.buffer_states[self.current_index]["state_dict"] = model.state_dict()
        self.buffer_states[self.current_index]["opt_dict"] = optimizer.state_dict()
        self.buffer_losses[self.current_index] = loss
        self.current_state_dict = self.buffer_states[self.current_index]["state_dict"]

        self.current_index = (self.current_index + 1) % self.buffer_size

    def restore_prev_best(self, model, optimizer):

        best = min([(i, loss) for i, loss in enumerate(self.buffer_losses)], key=lambda x: x[1])
        best_index = best[0]

        model.load_state_dict(self.buffer_states[best_index]["state_dict"])
        optimizer.load_state_dict(self.buffer_states[best_index]["opt_dict"])

        print(f"Restored best state from buffer, with loss={best[1]}.")

        return best[1]