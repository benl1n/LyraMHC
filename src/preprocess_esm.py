import os
import torch
import esm
import numpy as np
from tqdm import tqdm
import math
os.environ["TORCH_HOME"] = "D:/Ben_Plan/paper_factorary/LyraMHC_Project/ESM"
DATA_PATH = "D:/Ben_Plan/paper_factorary/LyraMHC_Project/data/processed/Anthem_train/"
SAVE_DIR = "D:/Ben_Plan/paper_factorary/LyraMHC_Project/ESM_Features/"
HLA_FILE = os.path.join(DATA_PATH, "proteins_esm.txt")
TRAIN_FILE = os.path.join(DATA_PATH, "train_data.txt")  # 假设你的训练文件名

MAX_LEN_HLA = 273
MAX_LEN_PEP = 15
ESM_DIM = 1280
# 64
BATCH_SIZE = 16


# ==========================================

def load_raw_data():
    print("Reading raw data...")
    hla_dict = {}
    with open(HLA_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            info = line.strip().split(' ')
            if len(info) < 2: continue
            hla_dict[info[0]] = info[1][:MAX_LEN_HLA]

    samples = []
    with open(TRAIN_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            info = line.strip().split('\t')
            if info[0] in hla_dict and len(info[1]) <= MAX_LEN_PEP:
                samples.append((info[0], info[1]))

    print(f"Loaded {len(hla_dict)} HLA sequences and {len(samples)} valid samples.")
    return hla_dict, samples


def run_preprocessing():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading ESM-2 esm2_t33_650M_UR50D on {device}...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    hla_dict, samples = load_raw_data()

    print("Step 1: Extracting HLA features (Dictionary)...")
    hla_esm_features = {}
    with torch.no_grad():
        for name, seq in tqdm(hla_dict.items()):
            _, _, tokens = batch_converter([("tmp", seq)])
            results = model(tokens.to(device), repr_layers=[33])
            hla_esm_features[name] = results["representations"][33][0, 1:-1, :].cpu().numpy().astype(np.float32)

    np.save(os.path.join(SAVE_DIR, "hla_esm_dict.npy"), hla_esm_features)
    print("HLA features saved.")

    print("Step 2: Extracting Peptide features (Memmap)...")
    num_samples = len(samples)
    memmap_path = os.path.join(SAVE_DIR, "pep_esm_640.dat")

    fp = np.memmap(memmap_path, dtype='float32', mode='w+', shape=(num_samples, MAX_LEN_PEP, ESM_DIM))

    for i in tqdm(range(0, num_samples, BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, num_samples)
        batch_data = samples[i:batch_end]
        pep_seqs = [s[1] for s in batch_data]

        _, _, tokens = batch_converter([("tmp", p) for p in pep_seqs])
        with torch.no_grad():
            results = model(tokens.to(device), repr_layers=[33])
            reprs = results["representations"][33].cpu().numpy()  # [B, L+2, 640]

            for j, seq in enumerate(pep_seqs):
                actual_len = len(seq)
                # 截取有效部分 [1 : 1+actual_len]
                feat = reprs[j, 1: 1 + actual_len, :]
                # 写入 memmap，不足 15 的地方自动保持为 0 (因为创建时是全 0)
                fp[i + j, :actual_len, :] = feat
                fp[i + j, actual_len:, :] = 0

    fp.flush()
    del fp

    import json
    with open(os.path.join(SAVE_DIR, "sample_order.json"), "w") as f:
        json.dump(samples, f)

    print(f"\nAll Done! Features stored in {SAVE_DIR}")
    print(f"Memmap file size: {os.path.getsize(memmap_path) / (1024 ** 3):.2f} GB")


if __name__ == "__main__":
    run_preprocessing()