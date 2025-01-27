import lightning as L
import torch
from torchsummary import summary
import argparse
from torcheval.metrics import MulticlassAUROC
from omegaconf import OmegaConf

from models import ConvDenseABBA, get_intermediate_outputs
from training import CustomOptimizer
from data import *
from attacks import acc_perturb_curve
from utils.conv_utils import get_lips
from utils.train_utils import get_agg_ops


def test(model, test_loader, n_classes, criterion, attack, bounds=(-1, 1)):
    features = get_intermediate_outputs(model, test_loader)
    
    for k in features.keys():
        print(f"{k} mean={features[k].mean()} std={features[k].std()}")

    model.eval()
    loss_total = 0
    correct_total = 0
    total = 0
    
    if n_classes == 2:
        auroc = MulticlassAUROC(num_classes=n_classes, device="cuda", average=None)
    else:
        auroc = MulticlassAUROC(num_classes=n_classes, device="cuda", average="macro")

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to("cuda"), y.to("cuda")

            y_pred = model(x)

            auroc.update(torch.nn.functional.softmax(y_pred, dim=-1), y)

            loss = criterion(y_pred, y.long()).float()
            cls_pred = torch.argmax(y_pred, dim=-1)   

            loss_total += loss
            correct_total += torch.sum(cls_pred == y)
            total += x.shape[0]

    print("\n")
    print(f"val_loss: {loss_total / len(test_loader)} val_acc: {correct_total / total}")
    print(f"val_AUROC: {auroc.compute()}")
    print("\n")

    get_lips(model, batch_norm_layers=[], simplified=False, compute_global_conv=True)
    # get_lips(model, batch_norm_layers=[], simplified=True)

    eps_base = [0.4 * i for i in range(11)]
    if bounds == (-1, 1):
        eps = [2 * x for x in eps_base]
    else:
        eps = eps_base
    
    accs = acc_perturb_curve(model, test_loader, bounds=bounds, epsilons=eps, attack=attack)
    accs = accs.tolist()

    print(f"{'eps':<10}{'Acc [%]':<10}")
    print("-" * 20)
    for e, a in zip(eps_base, accs):
        print(f"{e:<10.2f}{a * 100:.2f}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument('dataset', 
                        choices=["blood", "pneumo", "derma"],
                        help="Dataset name. Should have corresponding .yaml in ./configs.")
    parser.add_argument("chckpt_path", type=str)
    parser.add_argument("--attack", type=str, default="ddn")
    args = parser.parse_args()

    fabric = L.Fabric(accelerator='auto', devices=1)
    fabric.launch()

    bounds = (-1, 1)

    config = OmegaConf.load(f"./configs/{args.dataset}.yaml")
    
    agg_ops = get_agg_ops(config.network_agg_ops)
    model = ConvDenseABBA(agg_op=agg_ops, **config.network)

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

    # dummy optimizer for Fabric
    optimizer = CustomOptimizer(base_opt="adam", 
                                model=model, 
                                lr_standard=1e-3,
                                lr_abba=1e-3,
                                weight_decay=0)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.)

    scheduler = None

    model, optimizer = fabric.setup(model, optimizer)

    model.load_state_dict(
        torch.load(args.chckpt_path)
    )

    test(model, test_loader, config.network.n_classes, criterion, args.attack)