# LyraMHC

## 1. Overview
This source code is from **"LyraMHC: A Unified State-Space Framework for Peptide-MHC Binding and TCR Recognition"**.
LyraMHC is  a unified and highly decoupled framework that integrates state-space models (SSMs) with a bidirectional gated cross-attention (BGCA) mechanism for both pMHC and pMHC-TCR prediction. This framework support seven selectable encoding schemes, ranging from one-hot to blosum. 

## 2. Datasets

This study utilizes two publicly available benchmark datasets:

### (1) Anthem dataset

Available at:
https://github.com/s7776d/CapsNet-MHC/tree/main/dataset

## (1)  TranspMHC / TransTCR Dataset (Niu et al.)

Available at:
https://github.com/sherry-0805/TranspMHC-TransTCR

## 3. Dependencies

Install all required packages via:


```
pip install -r requirement.txt
```


## 4.Usage

### 4.1 Training and Testing

To train and evaluate the model:

```
python main.py
```

Logs and checkpoints will be saved to:

```
outputs/${train.name}/${experiment.name}/
```

## 5.Configuration

Dataset settings are defined in:
```
configs/dataset/Anthem.yaml
configs/dataset/transpMHC.yaml
```
Model architecture and hyperparameters are specified in:
```
configs/model/lyraMHC.yaml
```

The main experiment pipeline is controlled via:

```
configs/config.yaml
```

### 5.1 pMHC Task

**Anthem Dataset**

```
# configs/config.yaml
defaults:
    ...
  - dataset: Anthem
train:
    ...
  # train.name: Anthem_train / transpMHC_train
  name: "Anthem_train"
```

TranspMHC Dataset (Niu's Dataset)

```
# configs/config.yaml
defaults:
    ...
  - dataset: transpMHC
train:
    ...
  # train.name: Anthem_train / transpMHC_train
  name: "transpMHC_train"
```

### 5.2 pMHC-TCR Task

To enable TCR–pMHC prediction, modify:

```
# configs/config.yaml
experiment:
  ...
  best_model_path: "outputs/transpMHC_train/blosum/2026-01-05_19-30-47/checkpoints/best_model.pt"

train:
  ...
  encoding_method: "blosum"
  encoding_method2: "blosum"
  name: "transpMHC_train"
  task: "TCR"
```

Note: The pretrained pMHC model is required for transfer learning in the pMHC-TCR task.







