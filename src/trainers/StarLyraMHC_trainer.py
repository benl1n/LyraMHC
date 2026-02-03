import os
import datetime
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data.backward_compatibility import worker_init_fn
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
from src.encode import ENCODING_METHOD_MAP, one_hot_PLUS_blosum_encode, one_hot_encode, physical_encode, blosum_encode, \
    blosum80_encode, EDSSMat62_encode, N_blosum_encode
from src.logger import log_to_file, setup_logging
from src.models.components.LyraMHC import weight_initial, LyraMHC
from src.utlis import count_parameters, create_optimizer_with_llrd, set_reproducibility
from src.trainers.train_base import BaseTrainer
from src.data_provider.StarFusionDataProvider import MultiPeptideHLADataset
from src.utlis import get_data, get_model_save_path
from torch.optim.lr_scheduler import ReduceLROnPlateau

class StarLyraMHC_Trainer(BaseTrainer):

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

        # --- 1. 优化器配置 ---
        # 魏征提示：由于参数量翻倍（4个编码器），建议适当降低 LR 或增大 Weight Decay 防止过拟合
        # 这里先维持你的设定，但请密切关注 train_loss 是否下降过快
        start_lr = cfg.dataset.train.get('start_lr', 3e-4)
        # 魏征建议：Weight Decay 调大一点点 (1e-5 -> 1e-4) 以应对参数激增
        optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=1e-4)

        total_epochs = cfg.dataset.train.epochs

        # --- 2. 调度器：Plateau 是最稳的选择 ---
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )

        save_path_best = get_model_save_path(cfg, checkpoints_dir, fold, cfg.dataset.train.name, prefix="best")
        model_check_callback = ModelCheckPointCallBack(model, save_path_best, period=1, delta=1e-5)
        early_stop_callback = EarlyStopCallBack(patience=cfg.dataset.train.callback_patience, delta=1e-5)

        log_to_file('Start training', datetime.datetime.now())

        # pMHC task ---------------------------------------------------------------------
        if cfg.train.task == "pMHC":
            for epoch in tqdm(range(total_epochs), desc=f"Training Fold {fold}"):

                epoch_start_time = datetime.datetime.now()
                curr_real_lr = optimizer.param_groups[0]['lr']

                # 日志净化：不再记录 KeepProb
                log_to_file("LR-Action", f"Epoch {epoch}: LR={curr_real_lr:.2e}")

                # --- 训练阶段 ---
                model.train()
                epoch_loss = 0.0

                for hla_list, pep_list, label, sample in train_loader:
                    hla_list = [h.to(device) for h in hla_list]
                    pep_list = [p.to(device) for p in pep_list]
                    label = label.to(device)

                    # 魏征修正：移除所有掩码生成逻辑
                    # view_mask = None (默认全量)

                    optimizer.zero_grad()
                    # 直接前向传播，不传 view_mask
                    pred = model(hla_list, pep_list)

                    loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                    loss.backward()

                    # 梯度裁剪依然必要
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.dataset.train.grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()

                # --- 验证阶段 ---
                model.eval()
                val_loss = 0.0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for hla_list, pep_list, label, sample in val_loader:
                        hla_list = [h.to(device) for h in hla_list]
                        pep_list = [p.to(device) for p in pep_list]
                        label = label.to(device)

                        pred = model(hla_list, pep_list)

                        loss = nn.BCELoss()(pred.view(-1, 1), label.view(-1, 1))
                        val_loss += loss.item()
                        all_preds.extend(pred.view(-1).cpu().tolist())
                        all_labels.extend(label.view(-1).cpu().tolist())

                time_delta = datetime.datetime.now() - epoch_start_time
                avg_train_loss = epoch_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                # --- 调度器步进 ---
                scheduler.step(avg_val_loss)

                # --- 指标计算 ---
                y_true = np.array(all_labels)
                y_prob = np.array(all_preds)
                y_pred = (y_prob > 0.5).astype(int)

                model_check_callback.check(epoch, avg_val_loss)

                log_to_file("Training process",
                            "[Fold {0}]-[Epoch {1:04d}] - time: {2:4d} s, train_loss: {3:0.5f}, val_loss: {4:0.5f}".format(
                                fold, epoch, time_delta.seconds, avg_train_loss, avg_val_loss))

                metrics = {}
                try:
                    metrics['AUROC'] = roc_auc_score(y_true, y_prob)
                    metrics['AUPRC'] = average_precision_score(y_true, y_prob)
                except:
                    metrics['AUROC'] = float('nan')
                metrics['ACC'] = accuracy_score(y_true, y_pred)
                metrics['BACC'] = balanced_accuracy_score(y_true, y_pred)
                metrics['F1'] = f1_score(y_true, y_pred)
                metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
                metrics['val_Loss'] = avg_val_loss

                log_to_file("metric:", metrics)

                if early_stop_callback.check(epoch, avg_val_loss):
                    log_to_file("EarlyStop", f"Stopped at epoch {epoch}")
                    break

                last_path = get_model_save_path(cfg, checkpoints_dir, self.cfg.dataset.params.model_count - 1,
                                                cfg.dataset.train.name, prefix="last")
                torch.save(model.state_dict(), last_path)


        # TCR task ---------------------------------------------------------------------
        if cfg.train.task == "TCR":
            model.load_pretrained_weights(cfg.experiment.best_model_path, freeze=False)
            optimizer = create_optimizer_with_llrd(model, cfg)
            for epoch in tqdm(range(cfg.dataset.train.epochs), desc=f"Training Fold {fold}"):
                epoch_start_time = datetime.datetime.now()
                model.train()
                epoch_loss = 0.0
                for hla, pep, tcr, label in train_loader:
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
                    for hla, pep, tcr, label in val_loader:
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

                metrics = {}
                try:
                    metrics['AUROC'] = roc_auc_score(y_true, y_prob)
                except:
                    metrics['AUROC'] = float('nan')
                try:
                    metrics['AUPRC'] = average_precision_score(y_true, y_prob)
                except:
                    metrics['AUPRC'] = float('nan')
                metrics['ACC'] = accuracy_score(y_true, y_pred)
                metrics['BACC'] = balanced_accuracy_score(y_true, y_pred)
                metrics['F1'] = f1_score(y_true, y_pred)
                metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
                metrics['val_Loss'] = avg_val_loss
                print(metrics)
                log_to_file("metric:", metrics)

    def fit(self):
        cfg = self.cfg
        checkpoints_path = setup_logging(HydraConfig.get().runtime.output_dir)

        train_path, _, = get_data(cfg.train.name,
                                  cfg.train.task,
                                  cfg.train.data_path)

        df = pd.read_csv(train_path)

        y = df['label'].astype(int).tolist()

        skf = StratifiedKFold(n_splits=cfg.dataset.params.model_count, shuffle=True, random_state=cfg.train.seed)

        ENCODING_METHOD_MAP = {
            'one_hot': one_hot_encode,  # 20
            'physical': physical_encode,  # 11
            'blosum': blosum_encode,  # 23
            'EDSSMat62': EDSSMat62_encode # 24
        }

        target_methods = [
            'one_hot', 'physical', 'blosum', 'EDSSMat62'
        ]

        encoding_func = [ENCODING_METHOD_MAP[m] for m in target_methods]

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            set_reproducibility(cfg.train.seed)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_dataset = MultiPeptideHLADataset(
                train_df, encoding_func, cfg
            )
            val_dataset = MultiPeptideHLADataset(
                val_df, encoding_func, cfg
            )

            train_loader = DataLoader(
                train_dataset, batch_size=cfg.dataset.train.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, batch_size=cfg.dataset.train.batch_size, shuffle=False, num_workers=0
            )

            log_to_file(f"Fold {fold}", f"Train {len(train_dataset)}, Val {len(val_dataset)}")

            self.train(checkpoints_path, fold, train_loader, val_loader)
