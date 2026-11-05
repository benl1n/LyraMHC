# src/utils.py
import os
import random
import numpy as np
import torch

def set_reproducibility(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)



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
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)


