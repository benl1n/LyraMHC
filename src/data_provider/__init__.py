



def get_dataset(cfg):
    name = cfg.dataset.name

    if name == "Anthem":
        # 这里的参数直接从 cfg.dataset 读取，不再需要在这个函数里硬编码
        return AnthemDataset(
            file_path=cfg.dataset.train_file,
            max_len_hla=cfg.dataset.max_len_hla,

        )
    elif name == "TranspMHC":
        return TranspMHC_Dataset(
            file_path=cfg.dataset.train_file,

        )
    else:
        raise ValueError(f"Dataset {name} not supported!")