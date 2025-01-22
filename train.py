import lightning as L
import torch
from torchsummary import summary
import argparse
import wandb
from omegaconf import OmegaConf

from utils.train_utils import get_agg_ops
from models import ConvDenseABBA
from training import train_loop, CustomOptimizer
from data import *


if __name__ == "__main__":
    bounds = (-1, 1)
    
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('dataset', 
                        choices=["blood", "pneumo", "derma"],
                        help="Dataset name. Should have corresponding .yaml in ./configs.")
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save_p', type=str, default=None)
    args = parser.parse_args()

    fabric = L.Fabric(accelerator='auto', devices=1)
    fabric.launch()

    config = OmegaConf.load(f"./configs/{args.dataset}.yaml")

    # Get Model
    agg_ops = get_agg_ops(config.network_agg_ops)
    model = ConvDenseABBA(agg_ops, **config.network)

    # Get Data
    if args.dataset == "blood":
        train_loader, test_loader = blood_mnist(128, bounds=bounds)

    elif args.dataset == "pneumo":
        train_loader, test_loader = pnreumo_mnist(128, bounds=bounds)
        
    elif args.dataset == "derma":
        train_loader, test_loader = derma_mnist(127, bounds=bounds)

    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    train_loader = fabric.setup_dataloaders(train_loader)
    test_loader = fabric.setup_dataloaders(test_loader)

    try:
        summary(model, (config.network.in_ch, *config.network.hw), device="cpu")
    except:
        pass

    optimizer = CustomOptimizer(base_opt=config.optimizer.name, 
                                model=model, 
                                lr_standard=config.optimizer.lr_standard,
                                lr_abba=config.optimizer.lr_abba,
                                weight_decay=config.optimizer.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.)
    scheduler = None
    model, optimizer = fabric.setup(model, optimizer)

    # Train
    train_loop(model, 
                train_loader=train_loader,
                val_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=fabric.device,
                fabric=fabric,
                scheduler=scheduler,
                log_wandb=args.wandb,
                save_path=args.save_p,
                **config.train)

    if args.wandb:
        wandb.finish()