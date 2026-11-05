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
        if cfg.train.train_step == "True":
            self.run_dir = setup_logging(HydraConfig.get().runtime.output_dir)
        else:
            self.run_dir = cfg.train.test_path
        self.csv_dir = setup_logging(HydraConfig.get().runtime.output_dir)
        device = cfg.train.device if torch.cuda.is_available() else "cpu"

        self.target_hla = {"A*01:01", "A*02:01", "A*24:02", "B*08:01", "B*18:01"}
        # Explicitly define the required length range
        self.target_lengths = set(range(8, 16))  # 8 to 15 inclusive

        super().__init__(model=model, config=cfg, device=device)

    def test(self, checkpoint, cfg, data_loader, fold):
        model = self.model
        model.eval()

        task = cfg.train.task
        if task not in {"pMHC", "TCR"}:
            raise ValueError(f"Unsupported task: {task}")

        all_preds, all_labels = [], []
        all_peptides, all_hlas, all_tcrs = [], [], []

        for batch in tqdm(data_loader, desc=f"Testing fold {fold}"):
            if task == "pMHC":
                hla, pep, label, sample = batch

                hla = hla.to(self.device)
                pep = pep.to(self.device)

                pred = model(hla, pep)

                hla_seq, peptide_seq = sample
            else:
                hla, pep, tcr, label, sample = batch

                hla = hla.to(self.device)
                pep = pep.to(self.device)
                tcr = tcr.to(self.device)

                pred = model(hla, pep, tcr)

                hla_seq, peptide_seq, tcr_seq = sample
                all_tcrs.extend(tcr_seq)

            all_preds.extend(pred.reshape(-1).cpu().tolist())
            all_labels.extend(label.reshape(-1).cpu().tolist())
            all_hlas.extend(hla_seq)
            all_peptides.extend(peptide_seq)

            del pred, hla, pep
            if task == "TCR":
                del tcr

        if not all_preds:
            raise ValueError("The test loader produced no predictions.")

        output = {
            "HLA_sequence": all_hlas,
            "pred": all_preds,
            "label": all_labels,
            "peptide": all_peptides,
        }

        if task == "TCR":
            output["tcr"] = all_tcrs

        df_out = pd.DataFrame(output)

        filename = (
            f"fold_{fold}_TCR_predictions.csv"
            if task == "TCR"
            else f"fold_{fold}_predictions.csv"
        )
        out_path = os.path.join(self.run_dir, filename)
        df_out.to_csv(out_path, index=False)

        y_true = np.asarray(all_labels)
        y_prob = np.asarray(all_preds)
        y_pred = (y_prob > 0.5).astype(int)

        metrics = self._compute_metrics(y_true, y_prob, y_pred)

        print(f"Fold {fold} predictions saved to: {out_path}")
        print(f"Fold {fold} metrics: {metrics}")

        return metrics, df_out

    def fit(self):
        cfg = self.cfg
        model_count = cfg.dataset.params.model_count

        if model_count < 1:
            raise ValueError("model_count 1")

        _, test_path = get_data(
            cfg.train.name,
            cfg.train.task,
            cfg.train.data_path,
        )
        df = pd.read_csv(test_path)

        encoding_func = ENCODING_METHOD_MAP[cfg.train.encoding_method]
        test_dataset = PeptideHLADataset(df, encoding_func, cfg)

        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
        )

        fold_results = []
        fold_probs = []
        checkpoint_paths = []
        df_base = None

        for fold in range(model_count):
            log_to_file("Test", f"Loading fold {fold}")

            checkpoint_path = get_model_save_path(
                cfg,
                self.run_dir,
                fold,
                cfg.dataset.train.name,
                prefix="best",
            )

            model = self.model
            model.to(self.device)

            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            del state_dict
            model.eval()

            with torch.inference_mode():
                metrics, df_fold = self.test(
                    checkpoint_path,
                    cfg,
                    test_loader,
                    fold,
                )



            probs = df_fold["pred"].to_numpy(dtype=float)

            if not np.isfinite(probs).all():
                raise ValueError(f"fold {fold}NaN or Inf")

            if ((probs < 0) | (probs > 1)).any():
                raise ValueError(
                    f"fold {fold}pred_prob"
                )

            if df_base is None:
                df_base = df_fold.copy()
            else:
                if not np.array_equal(
                        df_base["label"].to_numpy(),
                        df_fold["label"].to_numpy(),
                ):
                    raise ValueError(f"fold {fold}")

                if "sample_id" in df_base.columns:
                    if "sample_id" not in df_fold.columns:
                        raise ValueError(f"fold {fold}")

                    if not np.array_equal(
                            df_base["sample_id"].to_numpy(),
                            df_fold["sample_id"].to_numpy(),
                    ):
                        raise ValueError(f"fold {fold}")

            fold_probs.append(probs)
            checkpoint_paths.append(checkpoint_path)

            fold_metrics = dict(metrics)
            fold_metrics["fold"] = fold
            fold_results.append(fold_metrics)

            log_to_file("Fold result", fold_metrics)

        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(
            os.path.join(self.csv_dir, "fold_metrics.csv"),
            index=False,
        )

        metric_names = ["AUC", "AUPR", "ACC", "BACC", "F1", "MCC"]
        summary_rows = []

        for name in metric_names:
            values = results_df[name].to_numpy(dtype=float)
            summary_rows.append({
                "metric": name,
                "mean": float(np.mean(values)),
                "std": (
                    float(np.std(values, ddof=1))
                    if model_count > 1 else np.nan
                ),
            })


        avg_probs = np.stack(fold_probs, axis=0).mean(axis=0)
        y_true = df_base["label"].to_numpy()
        threshold = 0.5
        y_pred = (avg_probs >= threshold).astype(int)

        ensemble_metrics = self._compute_metrics(
            y_true, avg_probs, y_pred
        )

        pd.DataFrame([ensemble_metrics]).to_csv(
            os.path.join(self.run_dir, "ensemble_metrics.csv"),
            index=False,
        )
        log_to_file("Ensemble result", ensemble_metrics)

        for fold, probs in enumerate(fold_probs):
            df_base[f"fold_{fold}_prob"] = probs

        df_base["ensemble_prob"] = avg_probs
        df_base["ensemble_pred"] = y_pred
        df_base.to_csv(
            os.path.join(self.run_dir, "ensemble_predictions.csv"),
            index=False,
        )

        pd.DataFrame({
            "fold": np.arange(model_count),
            "checkpoint_path": setup_logging(HydraConfig.get().runtime.output_dir),
            "weight": np.full(model_count, 1.0 / model_count),
            "threshold": threshold,
        }).to_csv(
            os.path.join(self.run_dir, "ensemble_manifest.csv"),
            index=False,
        )

        print("\nEnsemble metrics:")
        print(ensemble_metrics)

        return {
            "fold_metrics": results_df,
            "ensemble_metrics": ensemble_metrics,
        }

    def _compute_metrics(self, y_true, y_prob, y_pred):
        return {
            "AUC": roc_auc_score(y_true, y_prob),
            "AUPR": average_precision_score(y_true, y_prob),
            "ACC": accuracy_score(y_true, y_pred),
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
