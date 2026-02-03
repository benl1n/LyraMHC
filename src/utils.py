import os
import random
import numpy as np
import torch


def set_reproducibility(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data(train_name, task, data_path):
    if train_name == "Anthem_train" and task == "pMHC":
        train_file = os.path.join(data_path, "train_data.txt")
        test_file = os.path.join(data_path, "test_data.txt")
        file_path = os.path.join(data_path, "MHC_pseudo.txt")
        return train_file, test_file, file_path

    elif train_name == "transpMHC_train" and task == "pMHC":
        train_file = os.path.join(data_path, "pmhc_train.csv")
        test_file = os.path.join(data_path, "pmhc_test.csv")
        return train_file, test_file

    elif train_name == "transpMHC_train" and task == "TCR":
        train_file = os.path.join(data_path, "pmhc_tcr_train.csv")
        test_file = os.path.join(data_path, "pmhc_tcr_test.csv")
        return train_file, test_file


def get_model_save_path(cfg, save_dir, i, name, prefix="best"):
    if cfg.train.task == "TCR" and cfg.train.name == "transpMHC_train":
        filename = f"{prefix}_TCR_model_{i}.pt"
    elif cfg.train.task == "pMHC" and cfg.train.name == "transpMHC_train":
        filename = f"{prefix}_pMHC_model_{i}.pt"
    else:
        filename = f"{prefix}_model_{i}.pt"
    path = os.path.join(save_dir, name)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def create_optimizer_with_llrd(model, cfg):

    base_lr = cfg.dataset.train.start_lr
    weight_decay = cfg.train.get("weight_decay", 0.01)
    layer_decay = cfg.train.get("layer_decay", 0.8)  # 每一层向上衰减的系数

    no_decay = ["bias", "LayerNorm.weight", "BatchNorm1d.weight", "bn.weight"]

    try:
        layers = model.hla_encoder.encoder.blocks
        num_layers = len(layers)
    except AttributeError:
        print("Warning: Could not find model blocks for LLRD, falling back to basic grouping.")
        num_layers = 0

    def get_layer_lr(layer_idx):

        return base_lr * (layer_decay ** (num_layers - layer_idx))

    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        lr = base_lr

        if "encoder" in name and "blocks." in name:
            try:
                layer_idx = int(name.split("blocks.")[1].split(".")[0])
                lr = get_layer_lr(layer_idx)
            except (IndexError, ValueError):
                lr = base_lr * (layer_decay ** (num_layers + 1))

        elif "fusion_stack" in name:
            lr = base_lr * (layer_decay ** 1)

        elif "tcr_encoder" in name or "predictor" in name:
            lr = base_lr

        is_added = False
        for group in param_groups:
            if group["lr"] == lr and group["weight_decay"] == wd:
                group["params"].append(param)
                is_added = True
                break

        if not is_added:
            param_groups.append({
                "params": [param],
                "lr": lr,
                "weight_decay": wd,
                "name": f"lr_{lr:.2e}_wd_{wd}"
            })

    optimizer = torch.optim.AdamW(param_groups)

    print(f"LLRD: Created {len(param_groups)} parameter groups.")
    return optimizer