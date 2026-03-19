# trainers/capsnet_trainer.py
import datetime
import torch
from hydra.core.hydra_config import HydraConfig
from torch import nn, optim
from torch.optim import lr_scheduler

from src.callbacks import ModelCheckPointCallBack, EarlyStopCallBack
from src.data_provider.capsnet_data_provider import DataProvider
from src.encode import ENCODING_METHOD_MAP
from src.logger import log_to_file, setup_logging
from src.models.components.weight_initial import weight_initial

from src.registry import TRAINER_REGISTRY
from src.trainers.train_base import BaseTrainer
from src.utils import get_data, get_model_save_path, count_parameters, set_reproducibility


@TRAINER_REGISTRY.register("Anthem_train")
class CapsNet_Trainer(BaseTrainer):

    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        super().__init__(model=model, config=cfg, device=cfg.train.device)

    def batch_train(self, model, device, data):
        hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, ic50, samples = data

        pred_ic50 = model(hla_a.to(device),
                          pep.to(device))
        loss = nn.BCELoss()(pred_ic50.to(self.cfg.train.cpu_device), ic50.view(ic50.size(0), 1))

        return loss

    def batch_validation(self, model, device, data):
        hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, ic50, samples = data
        with torch.no_grad():
            pred_ic50 = model(hla_a.to(device),
                              pep.to(device))
            loss = nn.BCELoss()(pred_ic50.to(self.cfg.train.cpu_device), ic50.view(ic50.size(0), 1))
            pred_ic50 = pred_ic50.view(len(pred_ic50)).tolist()
            return loss, pred_ic50, samples

    def train(self, checkpoints_dir, cfg, data_provider, p):
        device = cfg.train.device
        log_to_file('Device', device)
        log_to_file('PyTorch version', torch.__version__)
        log_to_file('based on base_model #', p)

        for i in range(cfg.dataset.params.model_count):
            log_to_file('begin training model #', i)
            model = self.model
            weight_initial(model)
            model.to(device)

            log_to_file('Trainable params count', count_parameters(model))
            print(model.parameters())
            optimizer = optim.Adam(model.parameters(), lr=cfg.dataset.train.start_lr,
                                   weight_decay=cfg.dataset.train.weight_decay)
            log_to_file("Optimizer", "Adam")
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=cfg.dataset.train.loss_delta,
                                                       patience=cfg.dataset.train.scheduler_patience,
                                                       cooldown=cfg.dataset.train.cooldown,
                                                       min_lr=cfg.dataset.train.min_lr,
                                                       factor=cfg.dataset.train.scheduler_factor)

            save_path_best = get_model_save_path(cfg, checkpoints_dir, i, cfg.dataset.train.name, prefix="best")

            model_check_callback = ModelCheckPointCallBack(
                model,
                save_path_best,
                period=cfg.dataset.train.period,
                delta=cfg.dataset.train.loss_delta,
            )
            early_stop_callback = EarlyStopCallBack(patience=cfg.dataset.train.callback_patience,
                                                    delta=cfg.dataset.train.loss_delta)

            epoch_loss = 0
            validation_loss = 0
            data_provider.new_epoch()

            steps = data_provider.train_steps()
            log_to_file('Start training1', datetime.datetime.now())

            for epoch in range(cfg.dataset.train.epochs):
                epoch_start_time = datetime.datetime.now()
                model.train(True)
                for _ in range(steps):
                    data = data_provider.batch_train(i)
                    # print("***")
                    loss = self.batch_train(model, device, data)

                    loss.backward()
                    # clip grads
                    nn.utils.clip_grad_value_(model.parameters(), cfg.dataset.train.grad_clip)
                    # update params
                    optimizer.step()
                    # record loss
                    epoch_loss += loss.item()
                    # reset grad
                    optimizer.zero_grad()
                    # time compute
                time_delta = datetime.datetime.now() - epoch_start_time

                model.eval()

                val_sample = []
                val_pred = []
                for _ in range(data_provider.val_steps()):
                    data = data_provider.batch_val(i)
                    t_loss, t_pred, t_samples = self.batch_validation(model, device, data)
                    val_sample.append(t_samples)
                    val_pred.append(t_pred)
                    validation_loss += t_loss
                # log
                log_to_file("Training process",
                            "[base_model{0:1d}]-[model{1:1d}]-[Epoch {2:04d}] - time: {3:4d} s, train_loss: {4:0.5f}, val_loss: {5:0.5f}".format(
                                p, i, epoch, time_delta.seconds, epoch_loss / steps,
                                                                 validation_loss / data_provider.val_steps()))
                # call back
                model_check_callback.check(epoch, validation_loss / data_provider.val_steps())
                if early_stop_callback.check(epoch, validation_loss / data_provider.val_steps()):
                    break
                # LR schedule
                scheduler.step(loss.item())
                # reset loss
                epoch_loss = 0
                validation_loss = 0
                # reset data provider
                data_provider.new_epoch()

                # save last epoch model
                save_path_last = get_model_save_path(cfg, checkpoints_dir, i, cfg.dataset.train.name, prefix="last")
                torch.save(model.state_dict(), save_path_last)

    def fit(self):

        cfg = self.cfg
        encoding_func = ENCODING_METHOD_MAP[cfg.train.encoding_method]
        encoding_func2 = ENCODING_METHOD_MAP[cfg.train.encoding_method2]

        train_file, test_file, _ = get_data(cfg.dataset.train.name, cfg.train.task, cfg.train.data_path)

        data_provider = []
        for p in range(cfg.dataset.params.base_model_count):
            temp_provider = DataProvider(train_file,
                                         test_file,
                                         cfg.dataset.train.name,
                                         cfg.train.task,
                                         cfg.train.data_path,
                                         encoding_func,
                                         encoding_func2,
                                         cfg.dataset.train.batch_size,
                                         max_len_hla=cfg.dataset.params.max_len_hla,
                                         max_len_pep=cfg.dataset.params.max_len_pep,
                                         model_count=cfg.dataset.params.model_count,

                                         )
            data_provider.append(temp_provider)

        log_to_file('Traning samples', len(data_provider[0].train_samples[0]))
        log_to_file('Val samples', len(data_provider[0].validation_samples[0]))
        log_to_file('Traning steps', data_provider[0].train_steps())
        log_to_file('Val steps', data_provider[0].val_steps())
        log_to_file('Batch size', data_provider[0].batch_size)
        log_to_file('max_len_hla', data_provider[0].max_len_hla)
        log_to_file('max_len_pep', data_provider[0].max_len_pep)

        checkpoints_path = setup_logging(HydraConfig.get().runtime.output_dir)
        for p in range(cfg.dataset.params.base_model_count):
            set_reproducibility(cfg.train.seed)
            self.train(checkpoints_path, cfg, data_provider[p], p)
