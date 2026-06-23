import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from feral_segmentor.data.augmentations import FunctionAugmentation
from feral_segmentor.pipeline import segment


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    # placeholder chain: real config -> Augmentation chain construction not yet chosen
    chain = FunctionAugmentation(fn=lambda x: x)
    with mlflow.start_run():
        mlflow.log_params(chain.to_params() | OmegaConf.to_container(cfg, resolve=True))
        mlflow.set_tag("dataset_variant", cfg.augmentation.name)
        segment(cfg.data.root)


if __name__ == "__main__":
    main()
