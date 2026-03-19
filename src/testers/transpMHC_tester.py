# testers/transpMHC_tester.py
import os
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, \
    matthews_corrcoef

from src.encode import ENCODING_METHOD_MAP
from src.logger import log_to_file, setup_logging
from src.registry import TESTER_REGISTRY

from src.testers.test_base import Basetester
from src.data_provider.transpMHC_data_provider import PeptideHLADataset
from src.utils import get_data, get_model_save_path


@TESTER_REGISTRY.register("transpMHC_train")
class TranspMHC_Tester(Basetester):

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.run_dir = setup_logging(HydraConfig.get().runtime.output_dir)
        device = cfg.train.device if torch.cuda.is_available() else "cpu"

        self.target_hla = {"A*01:01", "A*02:01", "A*24:02", "B*08:01", "B*18:01"}
        # Explicitly define the required length range
        self.target_lengths = set(range(8, 16))  # 8 to 15 inclusive

        super().__init__(model=model, config=cfg, device=device)

    def get_real_length(self, pep_tensor):
        """
        Calculates the actual length of the peptide from a 20x15 one-hot tensor.
        Assumes padding columns are all zeros.
        pep_tensor shape: [20, 15]
        """
        # Sum across the amino acid dimension (dim=0)
        # If a column is all zeros (padding), the sum is 0.
        # If it's an amino acid, the sum is 1.
        mask = torch.sum(pep_tensor, dim=0) > 0
        return int(torch.sum(mask).item())

    def save_peptide_features(self, pep_origin, pep_encoded, hla_type, length, save_root):
        hla_dir = hla_type.replace("*", "_").replace(":", "_")
        # Organize by HLA -> Length to keep structure clean
        save_dir = os.path.join(save_root, hla_dir, f"len_{length}")
        os.makedirs(save_dir, exist_ok=True)

        # Transpose or keep as is? User said 20x15.
        # Usually CSVs are easier to read as (Length x Features), but I will respect your 20x15 structure.
        # Index names added for clarity.
        aa_labels = [f"AA_{i}" for i in range(pep_origin.size(0))]  # 0-19
        pos_labels_orgin = [f"Pos_{i}" for i in range(pep_origin.size(1))]  # 0-14
        pos_labels_encode = [f"Pos_{i}" for i in range(pep_encoded.size(1))]

        df_origin = pd.DataFrame(
            pep_origin.detach().cpu().numpy(),
            index=aa_labels,
            columns=pos_labels_orgin
        )

        df_encoded = pd.DataFrame(
            pep_encoded.detach().cpu().numpy(),
            index=[f"AA_{i}" for i in range(pep_encoded.size(0))],
            columns=pos_labels_encode
        )

        df_origin.to_csv(os.path.join(save_dir, "pep_origin.csv"))
        df_encoded.to_csv(os.path.join(save_dir, "pep_encoded.csv"))

    def test(self, checkpoint, cfg, data_loader, fold):
        model = self.model
        model.eval()

        all_preds, all_labels = [], []
        all_peptides, all_hlas = [], []

        # Track saved combinations: Set of tuples (HLA, Length)
        saved_combinations = set()
        total_required = len(self.target_hla) * len(self.target_lengths)

        if cfg.train.task == "pMHC":
            with torch.no_grad():
                for hla, pep, label, sample in tqdm(data_loader, desc="Testing"):
                    hla, pep, label = hla.to(self.device), pep.to(self.device), label.to(self.device)

                    pred = model(hla, pep)

                    all_preds.extend(pred.view(-1).cpu().tolist())
                    all_labels.extend(label.view(-1).cpu().tolist())

                    hla_seq, peptide_seq = sample

                    all_hlas.extend(hla_seq)
                    all_peptides.extend(peptide_seq)

                    # if len(saved_combinations) < total_required:
                    #     features = model(hla, pep, return_features=True)
                    #
                    #     # Iterate through the batch
                    #     for i in range(label.size(0)):
                    #         y = label[i].item()
                    #         hla_type = sample[i]
                    #
                    #         # Filter 1: Must be Positive
                    #         if y != 1:
                    #             continue
                    #
                    #         # Filter 2: Must be a Target HLA
                    #         if hla_type not in self.target_hla:
                    #             continue
                    #
                    #         pep_origin = features["pep_origin"][i]  # Shape [20, 15]
                    #         pep_encoded = features["pep_encoded"][i]
                    #
                    #         # Filter 3: calculate real length and check range 8-15
                    #         real_len = self.get_real_length(pep_origin)
                    #         if real_len not in self.target_lengths:
                    #             continue
                    #
                    #         # Filter 4: Check if we already have this (HLA, Length) pair
                    #         combo_key = (hla_type, real_len)
                    #         if combo_key in saved_combinations:
                    #             continue
                    #
                    #
                    #
                    #         run_dir = HydraConfig.get().runtime.output_dir
                    #         feature_root = os.path.join(run_dir, "features", f"fold_{fold}")
                    #
                    #         self.save_peptide_features(
                    #             pep_origin,
                    #             pep_encoded,
                    #             hla_type,
                    #             real_len,
                    #             feature_root
                    #         )
                    #
                    #         saved_combinations.add(combo_key)
                    #         print(
                    #             f"[Captured] Fold={fold} | HLA={hla_type} | Len={real_len} | Progress: {len(saved_combinations)}/{total_required}")
                    #
                    #         if len(saved_combinations) >= total_required:
                    #             print(
                    #                 "All target HLA-Length combinations captured. Stopping feature extraction (continuing metrics).")
                    #             break
                    df_out = pd.DataFrame({
                        "HLA_sequence": all_hlas,
                        "pred": all_preds,
                        "label": all_labels,
                        "peptide": all_peptides
                    })
                    out_path = os.path.join(self.run_dir, f"fold_{fold}_predictions.csv")
                    df_out.to_csv(out_path, index=False)
                    print(f" Fold {fold} predictions saved to: {out_path}")

            # Metric Calculation
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

            print(f"Fold {fold} Metrics: {metrics}")

        elif cfg.train.task == "TCR":
            with torch.no_grad():
                for hla, pep, tcr, label, sample in tqdm(data_loader, desc="Testing"):
                    hla, pep, tcr, label, sample = hla.to(self.device), pep.to(self.device), tcr.to(
                        self.device), label.to(self.device), sample
                    pred = model(hla, pep, tcr)

                    all_preds.extend(pred.view(-1).cpu().tolist())
                    all_labels.extend(label.view(-1).cpu().tolist())

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

        _, test_path = get_data(cfg.train.name,
                                cfg.train.task,
                                cfg.train.data_path)
        df = pd.read_csv(test_path)

        encoding_func = ENCODING_METHOD_MAP[cfg.train.encoding_method]

        test_dataset = PeptideHLADataset(
            df,
            encoding_func,
            cfg
        )

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

        for fold in range(cfg.dataset.params.model_count):
            log_to_file("Test", f"Loading fold {fold}")

            model = self.model
            model.to(self.device)

            checkpoints_path = get_model_save_path(
                cfg, self.run_dir, fold, cfg.dataset.train.name, prefix="best"
            )

            state_dict = torch.load(checkpoints_path, map_location=self.device)
            model.load_state_dict(state_dict)

            metrics, _ = self.test(checkpoints_path, cfg, test_loader, fold)
            metrics["fold"] = fold
            results.append(metrics)

            score = (metrics["AUC"] + metrics["AUPR"]) / 2
            metrics["score"] = score

            log_to_file(
                "Fold result",
                {**metrics, "score": score}
            )

            if score > best_score:
                best_score = score
                best_fold = fold
                best_state_dict = state_dict

        best_model_path = os.path.join(self.run_dir, "best_model.pt")
        torch.save(best_state_dict, best_model_path)

        log_to_file(
            "Best model",
            {
                "best_fold": best_fold,
                "best_score": best_score,
                "model_path": best_model_path,
            }
        )
