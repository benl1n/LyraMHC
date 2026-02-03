import torch
from torch.utils.data import Dataset


class PeptideHLADataset(Dataset):
    def __init__(self, df, encode_func,cfg):
        self.df = df.copy()
        self.encode_func = encode_func
        self.pep_max_len = cfg.dataset.params.max_len_pep
        self.hla_max_len = cfg.dataset.params.max_len_hla
        self.tcr_max_len = cfg.dataset.params.max_len_tcr

        self.task = cfg.train.task

        self.pep_cache, self.hla_cache, self.tcr_cache = {}, {}, {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep = str(row['peptide']).strip()[:self.pep_max_len]
        hla = str(row['HLA_sequence']).strip()[:self.hla_max_len]
        label = float(row['label'])
        sample = str(row['hla']), pep

        if pep not in self.pep_cache:
            pep_tensor, _ = self.encode_func(pep, self.pep_max_len)
            self.pep_cache[pep] = pep_tensor
        if hla not in self.hla_cache:
            hla_tensor, _ = self.encode_func(hla, self.hla_max_len)
            self.hla_cache[hla] = hla_tensor

        # ------------------------------------------------------------------------------
        if self.task == "TCR":
            tcr = str(row['tcr']).strip()[:self.tcr_max_len]
            if tcr not in self.tcr_cache:
                tcr_tensor, _ = self.encode_func(tcr, self.tcr_max_len)
                self.tcr_cache[tcr] = tcr_tensor

            return self.hla_cache[hla], self.pep_cache[pep], self.tcr_cache[tcr], torch.tensor(label, dtype=torch.float32), sample

        return  self.hla_cache[hla], self.pep_cache[pep], torch.tensor(label, dtype=torch.float32), sample