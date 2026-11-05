# LyraMHC

## 1. Overview
This source code is from **"LyraMHC: A Unified State-Space Framework for Peptide-MHC Binding and TCR Recognition"**.
LyraMHC is  a unified and highly decoupled framework that integrates state-space models (SSMs) with a bidirectional gated cross-attention (BGCA) mechanism for both pMHC and pMHC-TCR prediction. This framework support seven selectable encoding schemes, ranging from one-hot to blosum. 

![Model Architecture](https://raw.githubusercontent.com/benl1n/LyraMHC/main/framework.png)

## 2. Datasets

This study utilizes two publicly available benchmark datasets:

### (1) Anthem dataset

Available at:
https://github.com/s7776d/CapsNet-MHC/tree/main/dataset

### (2)  TranspMHC / TransTCR Dataset (Niu et al.)

Available at:
https://github.com/sherry-0805/TranspMHC-TransTCR

## 3. Dependencies

Install all required packages via:


```
pip install -r requirements.txt
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

**Training and inference**

For example, to evaluate the BLOSUM80 pMHC model on Niu's dataset:

```yaml
train:
  name: "transpMHC_train"
  task: "pMHC"
  encoding_method: "blosum80"
  encoding_method2: "blosum80"
  train_step: "False"
  test_path: "outputs/transpMHC_train/Blosum80/2026-10-09_15-48-10/checkpoints"
```

### 5.2 pMHC-TCR Task

To enable TCR–pMHC prediction, modify:

```
# configs/config.yaml
defaults:
    ...
  - dataset: transpMHC
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

**Training and inference**

Set `train.train_step` to `"True"` to train the TCR–pMHC model using pretrained pMHC weights. To evaluate a trained TCR–pMHC model, set it to `"False"` and specify the TCR–pMHC checkpoint directory through `train.test_path`. For example:



```yaml
train:
  name: "transpMHC_train"
  task: "TCR"
  encoding_method: "blosum80"
  encoding_method2: "blosum80"
  train_step: "False"
  test_path: "outputs/transpMHC_train/Blosum80_TCR/2026-10-09_22-27-56/checkpoints"
```

The encoding settings must match those used to train the saved TCR–pMHC model.

**Note:** `experiment.best_model_path` specifies the pretrained pMHC weights used for transfer learning, whereas `train.test_path` specifies the trained model checkpoints used for inference. All examples show only the relevant configuration settings; retain the remaining settings in `configs/config.yaml`.
