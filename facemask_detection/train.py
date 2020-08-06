import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import apex
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from iglovikov_helper_functions.dl.pytorch.lightning import find_average
from pytorch_lightning.logging import WandbLogger
from torch.utils.data import DataLoader

from facemask_detection.dataloader import FaceMaskDataset


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class FaceMask(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = object_from_dict(self.hparams["model"])

        if hparams["sync_bn"]:
            self.model = apex.parallel.convert_syncbn_model(self.model)

        self.loss = object_from_dict(self.hparams["loss"])

    def forward(self, batch: Dict) -> torch.Tensor:
        return self.model(batch)

    def setup(self, stage: int = 0) -> None:
        positive_samples: List[Tuple[Path, int]] = []
        negative_samples: List[Tuple[Path, int]] = []

        for file_name in sorted((Path(self.hparams["data_path"]) / "masks").glob("*.jpg")):
            positive_samples += [(file_name, 1)]

        for file_name in sorted((Path(self.hparams["data_path"]) / "non-masks").glob("*.jpg")):
            negative_samples += [(file_name, 0)]

        self.train_samples = (
            positive_samples[: int(0.8 * len(positive_samples))] + negative_samples[: int(0.8 * len(negative_samples))]
        )
        self.val_samples = (
            positive_samples[int(0.8 * len(positive_samples)) :] + negative_samples[int(0.8 * len(negative_samples)) :]
        )

    def train_dataloader(self):
        train_aug = from_dict(self.hparams["train_aug"])

        result = DataLoader(
            FaceMaskDataset(self.train_samples, train_aug),
            batch_size=self.hparams["train_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.hparams["val_aug"])

        return DataLoader(
            FaceMaskDataset(self.val_samples, val_aug),
            batch_size=self.hparams["val_parameters"]["batch_size"],
            num_workers=self.hparams["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def configure_optimizers(self):
        optimizer = object_from_dict(
            self.hparams["optimizer"], params=[x for x in self.model.parameters() if x.requires_grad]
        )

        scheduler = object_from_dict(self.hparams["scheduler"], optimizer=optimizer)
        self.optimizers = [optimizer]

        return self.optimizers, [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])

        total_loss = self.loss(logits, batch["target"])

        tqdm_dict = {"train_loss": total_loss, "lr": self._get_current_lr()}

        return OrderedDict({"loss": total_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

    def _get_current_lr(self) -> torch.Tensor:
        lr = [x["lr"] for x in self.optimizers[0].param_groups][0]
        return torch.Tensor([lr])[0].cuda()

    def validation_step(self, batch, batch_idx):
        targets = batch["target"]
        logits = self.forward(batch["image"])

        return OrderedDict({"val_loss": self.loss(logits, targets), "logits": logits, "targets": targets})

    def validation_epoch_end(self, outputs: List) -> Dict[str, Any]:
        result_probs: List[float] = []
        result_targets: List[float] = []

        for output in outputs:
            result_probs += torch.sigmoid(output["logits"]).cpu().numpy().flatten().tolist()
            result_targets += output["targets"].cpu().numpy().flatten().tolist()

        accuracy = np.mean(np.array(result_targets) == (np.array(result_probs) > 0.5).astype(int))
        loss = find_average(outputs, "val_loss")

        logs = {"val_loss": loss, "epoch": self.trainer.current_epoch, "accuracy": accuracy}

        return {"val_loss": loss, "log": logs}


def main():
    args = get_args()

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    pipeline = FaceMask(hparams)

    Path(hparams["checkpoint_callback"]["filepath"]).mkdir(exist_ok=True, parents=True)

    trainer = object_from_dict(
        hparams["trainer"],
        logger=WandbLogger(hparams["experiment_name"]),
        checkpoint_callback=object_from_dict(hparams["checkpoint_callback"]),
    )

    trainer.fit(pipeline)


if __name__ == "__main__":
    main()
