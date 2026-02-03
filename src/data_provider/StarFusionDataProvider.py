import torch
from torch.utils.data import Dataset


class MultiPeptideHLADataset(Dataset):
    def __init__(self, df, encoding_funcs_list, cfg):

        self.df = df.copy()

        # 魏征注：核心改动点。不再是一个函数，而是一组函数。
        self.encoding_funcs = encoding_funcs_list
        self.num_views = len(self.encoding_funcs)

        self.pep_max_len = cfg.dataset.params.max_len_pep
        self.hla_max_len = cfg.dataset.params.max_len_hla
        self.tcr_max_len = cfg.dataset.params.max_len_tcr

        self.task = cfg.train.task

        self.pep_cache = {}
        self.hla_cache = {}
        self.tcr_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pep = str(row['peptide']).strip()[:self.pep_max_len]
        hla = str(row['HLA_sequence']).strip()[:self.hla_max_len]
        label = float(row['label'])

        # 你的 sample 标识符逻辑保持不变
        sample = str(row['hla']), pep

        if pep not in self.pep_cache:
            encoded_views = []
            for func in self.encoding_funcs:

                result = func(pep, self.pep_max_len)

                if isinstance(result, tuple):
                    tensor = result[0]
                else:
                    tensor = result

                encoded_views.append(tensor)
            self.pep_cache[pep] = encoded_views


        if hla not in self.hla_cache:
            encoded_views = []
            for func in self.encoding_funcs:
                result = func(hla, self.hla_max_len)
                if isinstance(result, tuple):
                    tensor = result[0]
                else:
                    tensor = result
                encoded_views.append(tensor)
            self.hla_cache[hla] = encoded_views


        pep_data = self.pep_cache[pep]
        hla_data = self.hla_cache[hla]


        if self.task == "TCR":
            tcr = str(row['tcr']).strip()[:self.tcr_max_len]
            if tcr not in self.tcr_cache:
                encoded_views = []
                for func in self.encoding_funcs:
                    result = func(tcr, self.tcr_max_len)
                    if isinstance(result, tuple):
                        tensor = result[0]
                    else:
                        tensor = result
                    encoded_views.append(tensor)
                self.tcr_cache[tcr] = encoded_views

            tcr_data = self.tcr_cache[tcr]

            return hla_data, pep_data, tcr_data, torch.tensor(label, dtype=torch.float32), sample

        return hla_data, pep_data, torch.tensor(label, dtype=torch.float32), sample