import os
from pathlib import Path
from typing import Dict, List, Optional

import fiftyone.zoo as foz
import hydra
import torchvision.datasets as tv_datasets
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from feral_segmentor.utils import get_logger

log = get_logger(__name__)


def pull_model(cfg: DictConfig) -> list[Path]:
    """Download {repo_id, files} listed in a model config from the HF Hub into weights_dir."""
    weights_dir = Path(cfg.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in cfg.files:
        log.info("Fetching %s from %s", filename, cfg.repo_id)
        downloaded = hf_hub_download(
            repo_id=cfg.repo_id, filename=filename, local_dir=weights_dir
        )
        paths.append(Path(downloaded))
    return paths


# ---------------------------------------------------------------------
# Hugging Face downloader
# ---------------------------------------------------------------------


def download_hf_cat_dataset(
    dataset_name: str = "rafaelpadilla/coco2017",
    output_dir: str = "./data/hf_coco_cats",
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Path:
    """
    Download a HuggingFace dataset and filter cat instances.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset name.
    output_dir : str
        Where to save images.
    split : str
        Dataset split.
    max_samples : Optional[int]
        Optional limit of samples.

    Returns
    -------
    Path
        Path to downloaded dataset.
    """
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    ds = load_dataset(dataset_name, split=split)

    count = 0
    for item in tqdm(ds, desc="Downloading HF cats"):
        objects = item.get("objects")

        # COCO label 17 = cat
        if objects and 17 in objects.get("label", []):
            image = item["image"]

            save_path = output_path / f"{item['image_id']}.jpg"
            image.save(save_path)

            count += 1
            if max_samples and count >= max_samples:
                break

    return output_path


# ---------------------------------------------------------------------
# COCO downloader (via FiftyOne)
# ---------------------------------------------------------------------


def download_coco_cats(
    output_dir: str = "./data/coco_cats",
    split: str = "train",
    label_types: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """
    Download COCO dataset subset with only cat instances.

    Uses FiftyOne dataset zoo.

    Parameters
    ----------
    output_dir : str
        Save directory.
    split : str
        Dataset split: train / validation / test.
    label_types : Optional[List[str]]
        "detections" or "segmentations".
    max_samples : Optional[int]
        Limit dataset size.

    Returns
    -------
    Path
        Download location.
    """
    output_path = Path(output_dir)
    label_types = label_types or ["detections", "segmentations"]
    os.makedirs(output_path, exist_ok=True)
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=label_types,
        classes=["cat"],  # filter at download time
        max_samples=max_samples,
        dataset_dir=str(output_path),
    )

    return Path(dataset.info["dataset_dir"])


# ---------------------------------------------------------------------
# Oxford-IIIT Pet (segmentation)
# ---------------------------------------------------------------------


def download_oxford_pet_cats(
    output_dir: str = "./data/oxford_pet",
    split: str = "trainval",
) -> Path:
    """
    Download Oxford-IIIT Pet dataset (includes segmentation masks).

    Filters cat images.

    Parameters
    ----------
    output_dir : str
        Output directory.
    split : str
        Dataset split.

    Returns
    -------
    Path
        Path where dataset is stored.
    """
    output_path = Path(output_dir)
    os.makedirs(output_path, exist_ok=True)
    dataset = tv_datasets.OxfordIIITPet(
        root=output_dir,
        split=split,
        target_types=["category", "segmentation"],
        download=True,
    )

    cat_dir = output_path / "cats"

    for idx in tqdm(range(len(dataset)), desc="Filtering Oxford cats"):
        img, (label, seg) = dataset[idx]

        # Binary category: 0 = cat, 1 = dog (per torchvision docs)
        if label == 0:
            img_path = cat_dir / f"{idx}.jpg"
            seg_path = cat_dir / f"{idx}_mask.png"

            if isinstance(img, Image.Image):
                img.save(img_path)
            if isinstance(seg, Image.Image):
                seg.save(seg_path)

    return cat_dir


# ---------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------


def download_all_cat_datasets(
    base_dir: str = "./datasets",
    max_samples: Optional[int] = 500,
) -> Dict[str, Path]:
    """
    Download multiple cat datasets with segmentation/detection annotations.

    Parameters
    ----------
    base_dir : str
        Root directory for all datasets.
    max_samples : Optional[int]
        Per-dataset sample limit.

    Returns
    -------
    Dict[str, Path]
        Mapping dataset names to directories.
    """
    base = Path(base_dir)

    paths: Dict[str, Path] = {}
    os.makedirs(base, exist_ok=True)

    paths["hf_coco"] = download_hf_cat_dataset(
        output_dir=str(base / "hf_coco"),
        max_samples=max_samples,
    )

    paths["coco"] = download_coco_cats(
        output_dir=str(base / "coco"),
        max_samples=max_samples,
    )

    paths["oxford_pet"] = download_oxford_pet_cats(
        output_dir=str(base / "oxford_pet"),
    )

    return paths


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pull_model(cfg.model)


if __name__ == "__main__":
    main()
