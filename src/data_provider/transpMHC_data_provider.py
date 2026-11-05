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

        self.peptides = (
            df["peptide"]
            .astype(str)
            .str.strip()
            .str.slice(stop=self.pep_max_len)
            .tolist()
        )
        self.hlas = (
            df["HLA_sequence"]
            .astype(str)
            .str.strip()
            .str.slice(stop=self.hla_max_len)
            .tolist()
        )

        hla_names = df["hla"].astype(str).tolist()
        self.samples = list(zip(hla_names, self.peptides))
        self.labels = torch.tensor(
            df["label"].astype(float).to_numpy(),
            dtype=torch.float32,
        )

        self.pep_cache = {}
        self.hla_cache = {}
        self.tcr_cache = {}

        for pep in dict.fromkeys(self.peptides):
            pep_tensor, _ = self.encode_func(pep, self.pep_max_len)
            self.pep_cache[pep] = pep_tensor

        for hla in dict.fromkeys(self.hlas):
            hla_tensor, _ = self.encode_func(hla, self.hla_max_len)
            self.hla_cache[hla] = hla_tensor

        if self.task == "TCR":
            self.tcrs = (
                df["tcr"]
                .astype(str)
                .str.strip()
                .str.slice(stop=self.tcr_max_len)
                .tolist()
            )

            for tcr in dict.fromkeys(self.tcrs):
                tcr_tensor, _ = self.encode_func(tcr, self.tcr_max_len)
                self.tcr_cache[tcr] = tcr_tensor

            self.samples = list(zip(hla_names, self.peptides, self.tcrs))
        else:
            self.samples = list(zip(hla_names, self.peptides))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hla_tensor = self.hla_cache[self.hlas[idx]]
        pep_tensor = self.pep_cache[self.peptides[idx]]
        label = self.labels[idx]
        sample = self.samples[idx]

        if self.task == "TCR":
            tcr_tensor = self.tcr_cache[self.tcrs[idx]]
            return hla_tensor, pep_tensor, tcr_tensor, label, sample

        return hla_tensor, pep_tensor, label, sample