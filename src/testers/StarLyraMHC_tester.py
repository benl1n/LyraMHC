import os
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, \
    matthews_corrcoef

from src.encode import one_hot_PLUS_blosum_encode, one_hot_encode, blosum_encode, physical_encode, blosum80_encode, \
    EDSSMat62_encode, N_blosum_encode
from src.logger import log_to_file, setup_logging
from src.testers.test_base import Basetester
# 注意：这里必须导入和 Trainer 一致的多视图 Dataset
from src.data_provider.StarFusionDataProvider import MultiPeptideHLADataset
from src.utlis import get_data, get_model_save_path


class StarLyraMHC_Tester(Basetester):

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.run_dir = setup_logging(HydraConfig.get().runtime.output_dir)
        self.device = cfg.train.device if torch.cuda.is_available() else "cpu"
        super().__init__(model=model, config=cfg, device=self.device)

    def test(self, checkpoint_path, cfg, data_loader, fold):

        model = self.model
        model.eval()

        all_preds, all_labels = [], []

        # 魏征提示：TCR 任务和 pMHC 任务在解析 batch 时必须严格区分
        if cfg.train.task == "pMHC":
            with torch.no_grad():
                for hla_list, pep_list, label, sample in tqdm(data_loader, desc=f"Testing Fold {fold}"):

                    hla_list = [h.to(self.device) for h in hla_list]
                    pep_list = [p.to(self.device) for p in pep_list]
                    label = label.to(self.device)


                    pred = model(hla_list, pep_list)

                    all_preds.extend(pred.view(-1).cpu().tolist())
                    all_labels.extend(label.view(-1).cpu().tolist())

        elif cfg.train.task == "TCR":
            # 如果后期扩展 TCR，逻辑需在此同步更新
            with torch.no_grad():
                for hla, pep, tcr, label, sample in tqdm(data_loader, desc=f"Testing TCR Fold {fold}"):
                    hla, pep, tcr, label = hla.to(self.device), pep.to(self.device), tcr.to(self.device), label.to(
                        self.device)
                    pred = model(hla, pep, tcr)
                    all_preds.extend(pred.view(-1).cpu().tolist())
                    all_labels.extend(label.view(-1).cpu().tolist())

        # 指标计算逻辑
        y_true = np.array(all_labels)
        y_prob = np.array(all_preds)
        y_pred = (y_prob > 0.5).astype(int)

        metrics = {
            "AUC": roc_auc_score(y_true, y_prob),
            "AUPR": average_precision_score(y_true, y_prob),
            "ACC": accuracy_score(y_true, y_pred),
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        return metrics, y_prob

    def fit(self):

        cfg = self.cfg
        _, test_path = get_data(cfg.train.name, cfg.train.task, cfg.train.data_path)
        df = pd.read_csv(test_path)

        # 1. 定义与 Trainer 必须完全一致的 7 路编码映射
        ENCODING_METHOD_MAP = {
            'one_hot': one_hot_encode,  # 20
            'physical': physical_encode,  # 11
            'blosum': blosum_encode,  # 23
            'EDSSMat62': EDSSMat62_encode # 24
        }
        target_methods = [
            'one_hot', 'physical', 'blosum', 'EDSSMat62'
        ]
        encoding_funcs = [ENCODING_METHOD_MAP[m] for m in target_methods]

        # 2. 使用 MultiPeptideHLADataset
        test_dataset = MultiPeptideHLADataset(df, encoding_funcs, cfg)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.dataset.train.batch_size,
            shuffle=False,
            num_workers=0
        )

        best_score = -float("inf")
        best_fold = -1
        best_state_dict = None
        results = []

        # 3. 遍历 Fold 进行交叉验证评估
        for fold in range(cfg.dataset.params.model_count):
            log_to_file("Test", f"Loading Weights for Fold {fold}")

            # 定位 Trainer 保存的 best 权重
            checkpoints_path = get_model_save_path(
                cfg, self.run_dir, fold, cfg.dataset.train.name, prefix="best"
            )

            if not os.path.exists(checkpoints_path):
                print(f"Warning: {checkpoints_path} not found, skipping...")
                continue

            # 加载状态机
            state_dict = torch.load(checkpoints_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

            # 执行测试
            metrics, _ = self.test(checkpoints_path, cfg, test_loader, fold)
            metrics["fold"] = fold

            score = (metrics["AUC"] + metrics["AUPR"]) / 2
            metrics["score"] = score
            results.append(metrics)

            log_to_file("Fold result", {**metrics, "score": score})

            if score > best_score:
                best_score = score
                best_fold = fold
                best_state_dict = state_dict

        if best_state_dict is not None:
            best_model_path = os.path.join(self.run_dir, "best_model.pt")
            torch.save(best_state_dict, best_model_path)
            log_to_file("Overall Best model", {
                "best_fold": best_fold,
                "best_score": best_score,
                "model_path": best_model_path
            })