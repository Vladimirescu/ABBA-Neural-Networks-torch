import torch
import torch.nn as nn
from tqdm import tqdm
import math
import wandb
import os

from .constraints import constrain_abba_conv, ABBAConvConstraint, ABBADenseConstraint, StandardDenseConstraint, StandardConvConstraint
from models import get_intermediate_outputs, FullyDenseABBA, FullyConvABBA, ConvDenseABBA
from layers import MyBatchNorm1D, MyBatchNorm2D
from utils.conv_utils import get_bn_factors, get_lips
from utils.train_utils import ModelBuffer, schedule_lip


class CustomOptimizer(torch.optim.Optimizer):
    def __init__(self, base_opt, model, 
                 lr_standard=0.001, 
                 lr_abba=0.0001,
                 weight_decay=0):
        # Define some default values for compatibility
        defaults = dict(lr=lr_standard, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        super(CustomOptimizer, self).__init__(model.parameters(), defaults)

        # Initialize optimizer for each group with different learning rates
        self.model = model

        if base_opt == "adam":
            self.opt = torch.optim.Adam([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}
            ], lr=lr_standard, betas=(0.9, 0.999), weight_decay=weight_decay)
        elif base_opt == "sgd":
            self.opt = torch.optim.SGD([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}
            ], lr=lr_standard, weight_decay=weight_decay, momentum=0.99, nesterov=True)
        elif base_opt == "rmsprop":
            self.opt = torch.optim.RMSprop([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}
            ], lr=lr_standard, weight_decay=weight_decay)
        elif base_opt == "adadelta":
            self.opt = torch.optim.Adadelta([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}
            ], lr=lr_standard, rho=0.9, weight_decay=weight_decay)        
        elif base_opt == "adamax":
            self.opt = torch.optim.Adamax([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}                
            ], lr=lr_standard, betas=(0.5, 0.9), weight_decay=weight_decay)
        elif base_opt == "adamw":
            self.opt = torch.optim.AdamW([
                {"params": model.params["abba"], "lr": lr_abba},
                {"params": model.params["others"]}  
            ], lr=lr_standard, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            raise ValueError(f"Not implemented base optimizer {base_opt}.")

    def step(self, closure=None):
        # Perform optimization steps for each group of parameters
        self.opt.step(closure)

        with torch.no_grad():
            for p in self.model.params["abba"]:
                p.data.clamp_(min=1e-9)


def get_constraint_objects(model, device):
    
    batch_norm_layers = []
    dense_constraint_standard = None
    conv_constraint_standard = None
    dense_constraint = None
    conv_constraints = []

    try:
        obj = model.module
    except:
        obj = model

    if isinstance(obj, ConvDenseABBA):
        conv_constraints = [
            ABBAConvConstraint(x, device=device) for x in model.conv_net.abba_layers
        ]

        if len(model.conv_net.standard_layers) > 0:
            conv_constraint_standard = StandardConvConstraint(model.conv_net.standard_layers, device=device)

        for layer in model.conv_net.abba_model:
            if isinstance(layer, MyBatchNorm2D):
                batch_norm_layers.append(layer)

        dense_constraint = ABBADenseConstraint(model.dense_net.abba_layers, device=device)

        if len(model.dense_net.standard_layers) > 0:
            dense_constraint_standard = StandardDenseConstraint(model.dense_net.standard_layers, device=device, typ="clip")

        for layer in model.dense_net.abba_model:
            if isinstance(layer, MyBatchNorm1D):
                batch_norm_layers.append(layer)

    elif isinstance(obj, FullyDenseABBA):
        dense_constraint = ABBADenseConstraint(model.abba_layers, device=device)

        if len(model.standard_layers) > 0:
            dense_constraint_standard = StandardDenseConstraint(model.standard_layers, device=device, typ="clip")

        for layer in model.abba_model:
            if isinstance(layer, MyBatchNorm1D):
                batch_norm_layers.append(layer)

    elif isinstance(obj, FullyConvABBA):
        conv_constraints = [
            ABBAConvConstraint(x, device=device) for x in model.abba_layers
        ]

        if len(model.standard_layers) > 0:
            conv_constraint_standard = StandardConvConstraint(model.standard_layers, device=device)

        for layer in model.abba_model:
            if isinstance(layer, MyBatchNorm2D):
                batch_norm_layers.append(layer)

    return dense_constraint, dense_constraint_standard, conv_constraints, conv_constraint_standard, batch_norm_layers


def adaptive_constraint(
        constrain_conv,
        constrain_dense,
        lip_total,
        conv_constraints=[],
        dense_constraint=None,
        dense_constraint_standard=None,
        conv_constraint_standard=None,
        n_it_dense=15,
        n_it_conv=15,
        cnst=1e-1,
        bn_layers=[],
        min_lip_conv_prod=1,
        max_lip_standard=1
    ):
    """
    Adaptive computation of $l_2$ norms. 

    :param constrain_conv: bool, whether to constrain conv part
    :param constrain_dense: bool, whether to constrain dense part
    :param dense_constraint_standard: StandardDenseConstraint object
    :param conv_constraint_standard: StandardConvConstraint object
    :param cnst: tolerance for the total norm
    :param warmup_epochs: int, number of epochs to perform warmup: slowly decrease the start imposed
     norm up to the required maximum
    :param lip_start: Norm to start with
    :param lip_max: Norm to end with
    :param min_lip_conv_prod: minimu allowed value for the bound to impose on the conv product
    """

    """Build operators based on current weights"""
    for c_c in conv_constraints:
        c_c.construt_abba_conv()
    dense_constraint.construct_abba_denses()
    dense_constraint.construct_before_after_operators(0)

    """Constrain all standard layers to have Lip = 1"""
    if conv_constraint_standard:
        conv_constraint_standard.run(
            lips=1.0
        )
    if dense_constraint_standard:
        dense_constraint_standard.run(
            lips=1.0
        )

    total_prod = 1.0
    """"""

    if constrain_dense:
        dense_global = dense_constraint.current_global
    else:
        dense_global = 1

    if constrain_conv:
        conv_lips = torch.cat([c_c.current_lip.unsqueeze(0) for c_c in conv_constraints])
        conv_global = torch.prod(conv_lips)
    else:
        conv_global = 1
    
    with torch.no_grad():
        bn_lip = get_bn_factors(bn_layers)    

    current_global = dense_global * conv_global * total_prod * bn_lip
    non_abba_lip = total_prod * bn_lip
    
    """Constrain only when lower than imposed"""
    if current_global > lip_total + cnst:
        if constrain_dense and constrain_conv:
            """Adaptive constraint based on prev Lip"""
            dense_global_ = (lip_total / non_abba_lip)**0.5 * (dense_global / (conv_global + 1e-7))**0.5
            conv_global_ = (lip_total / non_abba_lip)**0.5 * (conv_global / (dense_global + 1e-7))**0.5

            # if conv_global_ < min_lip_conv_prod:
            #     conv_global_ = min_lip_conv_prod
            #     dense_global_ = lip_total / (non_abba_lip * conv_global_)

            # if dense_global_ < 1:
            #     dense_global_ = 1
            #     conv_global_ = lip_total / non_abba_lip

            conv_lips_ = conv_lips * (conv_global_ / conv_global)**(1.0 / len(conv_lips))

            """Make all layers equal Lip"""
            # dense_global_ = lip_total ** 0.5
            # conv_lips_ = torch.ones(len(conv_lips)) * (lip_total ** 0.5) ** (1 / len(conv_lips))
            """"""

            with torch.no_grad():
                dense_constraint.run(lip=dense_global_, n_it=n_it_dense)
                constrain_abba_conv(conv_constraints, conv_lips_, n_it=n_it_conv)
        elif constrain_dense and not constrain_conv:
            dense_global_ = (lip_total / non_abba_lip) / conv_global

            with torch.no_grad():
                dense_constraint.run(lip=dense_global_, n_it=n_it_dense)
        elif not constrain_dense and constrain_conv:
            conv_global_ = (lip_total / non_abba_lip) / dense_global
            conv_lips_ = conv_lips * (conv_global_ / conv_global)**(1.0 / len(conv_lips))

            with torch.no_grad():
                constrain_abba_conv(conv_constraints, conv_lips_, n_it=n_it_conv)
        else:
            pass
    

def abba_regularizer(model):
    reg = 0
    for layer in model.conv_net.abba_layers:
        reg += (torch.nn.functional.cosine_similarity(layer.A.view(-1), layer.B.view(-1), dim=0) + 1) / 2

    for layer in model.dense_net.abba_layers:
        reg += (torch.nn.functional.cosine_similarity(layer.A.view(-1), layer.B.view(-1), dim=0) + 1) / 2
    
    return reg


def train_loop(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epochs=100,
        val_loader=None,
        dense_constr=False,
        conv_constr=False,
        lip=10,
        lip_start=1000,
        fabric=None,
        constr_freq: int=1,
        constr_verbose_freq: int=10,
        test_freq: int=10,
        warmup_epochs: int=10,
        lazy_start_epochs: int=0,
        buffer_size: int=0,
        loss_margin_increase: float=0.03,
        scheduler: torch.optim.lr_scheduler=None,
        log_wandb: bool=False,
        save_path: str=None,
        gradient_clip: float=None,
        project: str="abba_torch",
        log_activations: bool=False,
        is_binary_cls: bool=False,
    ):
    """
    Training loop for constrained networks.

    :param dense_constr/conv_constr: bool, whether to constrain the dense/conv part
    :param lip: float > 0, maximum imposed norm
    :param lip_start: float, >lip, value to start the Lipschitz scheduler
    :param fabric: lightning.Fabric object or None. If given, training will be performed using the
                   PyTorch Lightning configuration
    :param constr_freq: int > 1, frequency of batches to constrain the model at
    :param constr_verbose_freq: int > 1, frequency of epochs to display Lipschitz stats
    :param test_freq: int > 1, frequency of epochs to print norm stats
    :param warmup_wepochs: int, number of epochs for gradually decreasing the imposed 
                           norm from lip_start to lip
    :param lazy_start_epochs: int, number of initial epochs to run the model constrained at lip_start 
    :param buffer_size: int, length of the buffer, representing the number of previous epochs for which to keep
                        the state_dict and optimizer_dict. If the current epoch presented unstable behaviour, 
                        the best model among previous buffer_size epochs is restored in its place
    :param loss_margin_increase: float, only applicable if buffer_size > 0. Represents the maximum acceptable loss
                        increase over which the best previous model is restored. This defines the "unstable behaviour"
    :param gradient_clip: float, value to clip gradients to
    :param project: str, name of the project used to log into wandb some metrics
    :paramm is_binary_cls: bool, whether binary classification is performed
    """

    if log_wandb:
        wandb.init(
            project=project,
            config=locals()
        )
        wandb.watch(model)

    model_buffer = None
    prev_loss_total = torch.inf
    if buffer_size > 0:
        model_buffer = ModelBuffer(buffer_size, model, optimizer, device)

    dense_constraint, dense_constraint_standard, conv_constraints, conv_constraint_standard, batch_norm_layers = get_constraint_objects(model, device=device)

    ### If log_wandb, these logged_metrics will be sent to W&B
    save_model = False
    best_loss = torch.inf
    acc_at_best = 0

    logged_metrics = {}
    for epoch in range(1, epochs + lazy_start_epochs + 1):
        model.train()
        loss_total = 0
        correct_total = 0
        total = 0

        """Warmup"""
        lip_total = schedule_lip(epoch, warmup_epochs, lazy_start_epochs, lip, lip_start, typ="exp")
        print(f"\nImposing Lip max {lip_total}.\n")
        logged_metrics["lip_constr"] = lip_total

        """Start testing at each epoch and (optional) save model"""
        if lip_total == lip:
            save_model = True if save_path else False
            print("From now saving")
            test_freq = 1

        for i, batch in tqdm(enumerate(train_loader)):
            ### Then optimize

            optimizer.zero_grad()
            x, y = batch

            if not fabric:
                x, y = x.to(device), y.to(device)

            y_pred = model(x)

            if is_binary_cls:
                loss = criterion(y_pred, y.float())
            else:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(y_pred, y.long())
                else:
                    loss = criterion(y_pred, y.float())

            if fabric:
                fabric.backward(loss)
            else:
                loss.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_value_(model.params["abba"], gradient_clip)
                torch.nn.utils.clip_grad_value_(model.params["others"], gradient_clip)

            optimizer.step()   

            ### Like a forward_pre_hook. Don't constrain after the final batch -> may override the final verbosed results
            if i % constr_freq == 0 and (conv_constr or dense_constr):
                adaptive_constraint(
                    conv_constr, 
                    dense_constr, 
                    lip_total,
                    conv_constraints, 
                    dense_constraint, 
                    dense_constraint_standard,
                    conv_constraint_standard,
                    bn_layers=batch_norm_layers
                )

            if is_binary_cls:
                cls_pred = (y_pred > 0.5).float()
            else:
                cls_pred = torch.argmax(y_pred, dim=-1)

            loss_total += loss
            correct_total += torch.sum(cls_pred == y)
            total += x.shape[0]

        if epoch % constr_verbose_freq == 0 and (conv_constr or dense_constr):
            lips = get_lips(model, batch_norm_layers, 
                            simplified=False, return_lips=True, compute_global_conv=True)

            standard_dense_lips, abba_dense_lips, abba_dense_prod, abba_dense_global, \
            standard_conv_lips, abba_conv_lips, abba_conv_global, bn_lip, total_prod, total_prod_no_bn = lips

            logged_metrics["lips/lip_total_prod"] = total_prod
            logged_metrics["lips/lip_total_prod_noBN"] = total_prod_no_bn
            for i in range(len(abba_conv_lips)):
                logged_metrics[f"lips/conv_lips_{i}"] = abba_conv_lips[i]
            logged_metrics["lips/conv_global_lip"] = abba_conv_global
            logged_metrics["lips/dense_global_lip"] = abba_dense_global
            for i in range(len(abba_dense_lips)):
                logged_metrics[f"lips/dense_lips_{i}"] = abba_dense_lips[i]
            
            logged_metrics["bn_lip"] = bn_lip


        loss_total = loss_total / len(train_loader)
        print(f"Epoch {epoch}/{epochs + lazy_start_epochs}: train_loss: {loss_total} train_acc: {correct_total / total}")

        logged_metrics["train/acc"] = correct_total / total
        logged_metrics["train/loss"] = loss_total

        """update Buffer and restore if neccessary"""
        logged_metrics["buffer/changed"] = 0
        if model_buffer:
            model_buffer.update(model, optimizer, loss_total)

            if loss_total > prev_loss_total + loss_margin_increase:
                restored_loss = model_buffer.restore_prev_best(model, optimizer)
                prev_loss_total = restored_loss

                logged_metrics["buffer/changed"] = 1
            else:
                prev_loss_total = loss_total

        """Update lr"""
        if scheduler:
            if isinstance(scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, 
                                      torch.optim.lr_scheduler.ChainedScheduler)):
                scheduler.step(loss_total)
            else:
                scheduler.step()

            for i, param_group in enumerate(optimizer.param_groups):
                logged_metrics[f"lr/param_group_{i}"] = param_group['lr']

        """test iterate"""
        if epoch % test_freq == 0 and val_loader is not None:
            model.eval()
            loss_total = 0
            correct_total = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch

                    if not fabric:
                        x, y = x.to(device), y.to(device)

                    y_pred = model(x)

                    if is_binary_cls:
                        loss = criterion(y_pred, y.float()).float()
                        cls_pred = (y_pred > 0.5).float()
                    else:
                        loss = criterion(y_pred, y.long()).float()
                        cls_pred = torch.argmax(y_pred, dim=-1)                        

                    loss_total += loss
                    correct_total += torch.sum(cls_pred == y)
                    total += x.shape[0]

            print("\n")
            print(f"Epoch {epoch}: val_loss: {loss_total / len(val_loader)} val_acc: {correct_total / total}")
            print("\n")

            logged_metrics["test/acc"] = correct_total / total
            logged_metrics["test/loss"] = loss_total / len(val_loader)

            """(optional) save"""
            if logged_metrics["test/loss"] < best_loss and save_model:
                best_loss = logged_metrics["test/loss"]
                
                acc_at_best = logged_metrics["test/acc"]

                dir_name, file_name = os.path.split(save_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)

                if fabric:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

            print(f"Current best Acc / Loss: {acc_at_best} / {best_loss}")

        if log_wandb:
            parameters_dict = {}
            for name, param in model.named_parameters():
                parameters_dict[f"weights/{name}"] = param.detach().cpu().numpy()

            """Log some activations"""
            if log_activations:
                with torch.no_grad():
                    if val_loader is not None:
                        activations = get_intermediate_outputs(model, val_loader, device)
                    else:
                        activations = get_intermediate_outputs(model, train_loader, device)

                print("#### Activations ####")
                for act in activations.keys():
                    print(f"{act} mean={activations[act].mean()} std={activations[act].std()}")
                    parameters_dict[f"activations/{act}"] = activations[act].cpu().numpy()

            logged_metrics.update(parameters_dict)
            wandb.log(logged_metrics)