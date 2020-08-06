import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import yaml
from albumentations.core.serialization import from_dict
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from facemask_detection.dataloader import FaceMaskTestDataset
from facemask_detection.utils import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-c", "--config_path", type=Path, help="Path to config.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save jsons.", required=True)
    arg("-b", "--batch_size", type=int, help="batch_size", default=2)
    arg("-j", "--num_workers", type=int, help="num_workers", default=12)
    arg("-w", "--weight_path", type=str, help="Path to weights.", required=True)
    arg("--world_size", default=-1, type=int, help="number of nodes for distributed training")
    arg("--local_rank", default=-1, type=int, help="node rank for distributed training")
    arg("--fp16", action="store_true", help="Use fp6")

    return parser.parse_args()


def main():
    args = get_args()
    torch.distributed.init_process_group(backend="nccl")

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams.update({"json_path": args.output_path, "local_rank": args.local_rank, "fp16": args.fp16})

    device = torch.device("cuda", args.local_rank)

    model = object_from_dict(hparams["model"])
    model = model.to(device)

    if args.fp16:
        model = model.half()

    corrections: Dict[str, str] = {"model.": ""}
    checkpoint = load_checkpoint(file_path=args.weight_path, rename_in_layers=corrections)
    model.load_state_dict(checkpoint["state_dict"])

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    file_paths = []

    for regexp in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
        file_paths += sorted(x for x in tqdm(args.input_path.rglob(regexp)))

    dataset = FaceMaskTestDataset(file_paths, transform=from_dict(hparams["test_aug"]))

    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=sampler is None,
        drop_last=False,
        sampler=sampler,
    )

    prediction = predict(dataloader, model, hparams, device)

    prediction_list = [torch.zeros_like(prediction) for _ in range(dist.get_world_size())]
    dist.all_gather(prediction_list, prediction)

    if dist.get_rank() == 0:
        with torch.no_grad():
            predictions = torch.cat(prediction_list, dim=1).reshape(-1).cpu().numpy()[: len(dataset)]

        df = pd.DataFrame({"file_path": file_paths, "predictions": predictions})

        df.to_csv(args.output_path, index=False)


def predict(dataloader, model, hparams, device):
    model.eval()

    if hparams["local_rank"] == 0:
        loader = tqdm(dataloader)
    else:
        loader = dataloader

    result = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"]  # images that are rescaled and padded

            if hparams["fp16"]:
                images = images.half()

            prediction = model(images.to(device))

            result += [torch.sigmoid(prediction)]

    return torch.cat(result)


if __name__ == "__main__":
    main()
