import foolbox as fb
import torch
from tqdm import tqdm


def acc_perturb_curve(model, dataloader, bounds, epsilons, attack="ddn", device="cuda"):
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)
    if attack == "ddn":
        attack = fb.attacks.DDNAttack(steps=300)
    elif attack == "deepfool":
        attack = fb.attacks.L2DeepFoolAttack(steps=300)
    elif attack == "fmn":
        attack = fb.attacks.L2FMNAttack(steps=300)
    else:
        raise NotImplementedError(f"Attack {attack} not yet supported.")

    Success = None
    for i, batch in tqdm(enumerate(dataloader)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        _, advs, success = attack(fmodel, x, y.squeeze(), epsilons=epsilons)

        if Success is None:
            Success = success
        else:
            Success = torch.cat((Success, success), dim=-1)

        accs = torch.mean((~Success).float(), dim=-1)

    return accs


def approx_lip(model, dataloader, bounds, eps=4, attack="ddn", device="cuda"):
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=bounds, device=device)
    if attack == "ddn":
        attack = fb.attacks.DDNAttack(steps=300)
    elif attack == "deepfool":
        attack = fb.attacks.L2DeepFoolAttack(steps=300)
    elif attack == "fmn":
        attack = fb.attacks.L2FMNAttack(steps=300)
    else:
        raise NotImplementedError(f"Attack {attack} not yet supported.")
    
    max_lip = 0
    for i, batch in tqdm(enumerate(dataloader)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        _, advs, _ = attack(fmodel, x, y.squeeze(), epsilons=eps)

        logits_x = model(x)
        logits_x_adv = model(advs)

        mask = logits_x.argmax(1) == y

        x, advs, logits_x, logits_x_adv = x[mask], advs[mask], logits_x[mask], logits_x_adv[mask]

        x_norms = torch.norm(x - advs, p=2, dim=(1, 2, 3))
        log_norms = torch.norm(logits_x - logits_x_adv, p=2, dim=-1)

        max_lip = max(torch.max(log_norms / (x_norms + 1e-7)), max_lip)

    return max_lip