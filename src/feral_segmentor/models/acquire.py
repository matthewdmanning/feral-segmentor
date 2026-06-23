"""Hydra entrypoint: acquire a model's weights per its configured source."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from feral_segmentor.config.store import register_configs
from feral_segmentor.models.source import build_model_source

# Schemas must be registered before Hydra composes the config (i.e. before the
# @hydra.main-decorated body runs), so register at import time as well.
register_configs()


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    register_configs()
    build_model_source(cfg.model).acquire(cfg.model)


if __name__ == "__main__":
    main()
