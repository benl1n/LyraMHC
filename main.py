import hydra
from omegaconf import DictConfig

from src.models.build import build_model
from src.testers import get_tester
from src.utlis import set_reproducibility
from src.logger import log_to_file
from src.trainers import get_trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log_to_file("Experiment Name", cfg.experiment.name)
    log_to_file("Description", cfg.experiment.description)
    log_to_file("Tags:", cfg.experiment.tags)

    model = build_model(cfg)
    if cfg.train.train_step == "True":
        trainer = get_trainer(cfg, model)
        trainer.fit()

    try:
        tester = get_tester(cfg, model)
        if tester is None:
            log_to_file("Tester", "No test stage for this dataset, skipped.")
        else:
            tester.fit()
    except Exception as e:
        log_to_file("Tester Error", str(e))

    if __name__ == '__main__':
        main()