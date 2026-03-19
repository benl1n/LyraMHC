# trainers/transpMHC_trainer.py
import datetime
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, \
    matthews_corrcoef
from src.callbacks import ModelCheckPointCallBack, EarlyStopCallBack
from src.encode import ENCODING_METHOD_MAP
from src.logger import log_to_file, setup_logging
from src.metrics import get_metrics
from src.models.components.weight_initial import weight_initial

from src.utils import count_parameters, set_reproducibility
from src.trainers.train_base import BaseTrainer
from src.data_provider.transpMHC_data_provider import PeptideHLADataset
from src.utils import get_data, get_model_save_path
from src.registry import TRAINER_REGISTRY


@TRAINER_REGISTRY.register("transpMHC_train")
class ModernTrainer(BaseTrainer):

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.device = cfg.train.device if torch.cuda.is_available() else "cpu"

        super().__init__(model=model, config=cfg, device=self.device)

    def parse_batch(self, batch, task, device):
        if task == "pMHC":
            hla, pep, label = batch
            return {
                "inputs": (hla.to(device), pep.to(device)),
                "label": label.to(device)
            }
        elif task == "TCR":
            hla, pep, tcr, label = batch
            return {
                "inputs": (hla.to(device), pep.to(device), tcr.to(device)),
                "label": label.to(device)
            }
        else:
            raise ValueError(f"Unknown task: {task}")

    def train_epoch(self):
        pass

    def train(self, checkpoints_dir, fold, train_loader, val_loader):

        cfg = self.cfg
        device = self.device

        log_to_file('Begin training model #', fold)
        model = self.model
        weight_initial(model)
        model.to(device)
        log_to_file('Trainable params', count_parameters(model))

        optimizer = optim.Adam(model.parameters(), lr=cfg.dataset.train.start_lr,
                               weight_decay=cfg.dataset.train.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, threshold=cfg.dataset.train.loss_delta, patience=cfg.dataset.train.scheduler_patience,
            cooldown=cfg.dataset.train.cooldown, min_lr=cfg.dataset.train.min_lr,
            factor=cfg.dataset.train.factor
        )

        save_path_best = get_model_save_path(cfg, checkpoints_dir, fold, cfg.dataset.train.name, prefix="best")

        model_check_callback = ModelCheckPointCallBack(
            model, save_path_best, period=cfg.dataset.train.period, delta=cfg.dataset.train.loss_delta
        )
        early_stop_callback = EarlyStopCallBack(patience=cfg.dataset.train.callback_patience,
                                                delta=cfg.dataset.train.loss_delta)

        log_to_file('Start training', datetime.datetime.now())

        # pMHC task ---------------------------------------------------------------------
        if cfg.train.task == "pMHC":
            for epoch in tqdm(range(cfg.dataset.train.epochs), desc=f"Training Fold {fold}"):
                epoch_start_time = datetime.datetime.now()
                model.train()
                epoch_loss = 0.0
                for hla, pep, label, sample in train_loader:
                    hla, pep, label, sample = hla.to(device), pep.to(device), label.to(device), sample
                    optimizer.zero_grad()
                    pred = model(hla, pep)
                    loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), cfg.dataset.train.grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()

                time_delta = datetime.datetime.now() - epoch_start_time
                model.eval()
                val_loss = 0.0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for hla, pep, label, sample in val_loader:
                        hla, pep, label, sample = hla.to(device), pep.to(device), label.to(device), sample
                        pred = model(hla, pep)
                        loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                        val_loss += loss.item()
                        all_preds.extend(pred.view(-1).cpu().tolist())
                        all_labels.extend(label.view(-1).cpu().tolist())

                avg_train_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                y_true = np.array(all_labels)
                y_prob = np.array(all_preds)
                y_pred = (y_prob > 0.5).astype(int)

                model_check_callback.check(epoch, avg_val_loss)

                log_to_file("Training process",
                            "[base_model{0:1d}]-[model{1:1d}]-[Epoch {2:04d}] - time: {3:4d} s, train_loss: {4:0.5f}, val_loss: {5:0.5f}".format(
                                fold, fold, epoch, time_delta.seconds, avg_train_loss,
                                avg_val_loss))

                if early_stop_callback.check(epoch, avg_val_loss):
                    log_to_file("EarlyStop", f"Stopped at epoch {epoch}")
                    break

                scheduler.step(avg_val_loss)

                last_path = get_model_save_path(cfg, checkpoints_dir, self.cfg.dataset.params.model_count - 1,
                                                cfg.dataset.train.name, prefix="last")
                torch.save(model.state_dict(), last_path)
                get_metrics(y_true, y_prob, y_pred, avg_val_loss)


        # TCR task ---------------------------------------------------------------------
        if cfg.train.task == "TCR":
            model.load_pretrained_weights(cfg.experiment.best_model_path, freeze=False)
            for epoch in tqdm(range(cfg.dataset.train.epochs), desc=f"Training Fold {fold}"):
                epoch_start_time = datetime.datetime.now()
                model.train()
                epoch_loss = 0.0
                for hla, pep, tcr, label, sample in train_loader:
                    hla, pep, tcr, label = hla.to(device), pep.to(device), tcr.to(device), label.to(device)
                    optimizer.zero_grad()
                    pred = model(hla, pep, tcr)
                    loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                    loss.backward()
                    nn.utils.clip_grad_value_(model.parameters(), cfg.dataset.train.grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()

                time_delta = datetime.datetime.now() - epoch_start_time
                model.eval()
                val_loss = 0.0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for hla, pep, tcr, label, sample in val_loader:
                        hla, pep, tcr, label = hla.to(device), pep.to(device), tcr.to(device), label.to(device)
                        pred = model(hla, pep, tcr)
                        loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                        val_loss += loss.item()
                        all_preds.extend(pred.view(-1).cpu().tolist())
                        all_labels.extend(label.view(-1).cpu().tolist())

                avg_train_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                y_true = np.array(all_labels)
                y_prob = np.array(all_preds)
                y_pred = (y_prob > 0.5).astype(int)

                model_check_callback.check(epoch, avg_val_loss)

                log_to_file("Training process",
                            "[base_model{0:1d}]-[model{1:1d}]-[Epoch {2:04d}] - time: {3:4d} s, train_loss: {4:0.5f}, val_loss: {5:0.5f}".format(
                                fold, fold, epoch, time_delta.seconds, avg_train_loss,
                                avg_val_loss))

                if early_stop_callback.check(epoch, avg_val_loss):
                    log_to_file("EarlyStop", f"Stopped at epoch {epoch}")
                    break

                scheduler.step(avg_val_loss)

                last_path = get_model_save_path(cfg, checkpoints_dir, self.cfg.dataset.params.model_count - 1,
                                                cfg.dataset.train.name, prefix="last")
                torch.save(model.state_dict(), last_path)
                get_metrics(y_true, y_prob, y_pred, avg_val_loss)

    def fit(self):
        cfg = self.cfg
        checkpoints_path = setup_logging(HydraConfig.get().runtime.output_dir)  #

        train_path, _ = get_data(cfg.train.name,
                                 cfg.train.task,
                                 cfg.train.data_path)

        df = pd.read_csv(train_path)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)

        y = df['label'].astype(int).tolist()

        skf = StratifiedKFold(n_splits=cfg.dataset.params.model_count, shuffle=True, random_state=cfg.train.seed)

        encoding_func = ENCODING_METHOD_MAP[cfg.train.encoding_method]

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            g = torch.Generator()
            g.manual_seed(cfg.train.seed)
            worker_init_fn = set_reproducibility(cfg.train.seed)

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_dataset = PeptideHLADataset(
                train_df, encoding_func, cfg
            )
            val_dataset = PeptideHLADataset(
                val_df, encoding_func, cfg
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.dataset.train.batch_size,
                shuffle=True,
                num_workers=0,
                worker_init_fn=worker_init_fn,
                generator=g

            )
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.dataset.train.batch_size, shuffle=False, num_workers=0
            )

            log_to_file(f"Fold {fold}", f"Train {len(train_dataset)}, Val {len(val_dataset)}")

            self.train(checkpoints_path, fold, train_loader, val_loader)